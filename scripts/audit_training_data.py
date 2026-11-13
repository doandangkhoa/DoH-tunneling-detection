"""Audit source shift, validation support, CNN padding and observable TLS types."""
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.runtime-deps'))
import numpy as np
import pandas as pd

def audit():
    master=pd.read_parquet(ROOT/'data/interim/master_dataset.parquet').set_index('sample_id')
    packets=pd.read_parquet(ROOT/'data/interim/packet_metadata.parquet')
    qa=ROOT/'data/interim/qa'
    source_rows=[];sequence_rows=[];split_info={};metadata=[]
    for split in ['train','val','test']:
        with np.load(ROOT/f'data/processed/cnn1d/{split}.npz',allow_pickle=False) as z:
            ids=z['sample_id'];mask=z['mask'];y=z['y']
            lengths=mask.sum(axis=1)
            assert np.array_equal(y,master.loc[ids,'label'])
        meta=master.loc[ids,['label','source','capture_id','split_group','split']].reset_index()
        meta['tunneling_tool']=meta.source.where(meta.label==1)
        metadata.append(meta)
        for (source,label),sub in meta.groupby(['source','label']):
            source_rows.append({'split':split,'source':source,'label':int(label),'flows':len(sub),
                                'captures':sub.capture_id.nunique(),'groups':sub.split_group.nunique(),
                                'share_within_class':len(sub)/(y==label).sum()})
        for sid,n in zip(ids,lengths):
            sequence_rows.append({'sample_id':sid,'split':split,'source':master.at[sid,'source'],
                                  'label':int(master.at[sid,'label']),'valid_packets':int(n),
                                  'padding_fraction':float(1-n/32)})
        benign=meta[meta.label==0];n=len(benign)
        split_info[split]={'flows':len(ids),'benign':n,'malicious':int((y==1).sum()),
                           'mean_valid_packets':float(lengths.mean()),'median_valid_packets':float(np.median(lengths)),
                           'p10_valid_packets':float(np.quantile(lengths,.1)),
                           'padding_fraction':float(1-lengths.sum()/(len(ids)*32)),
                           'short_sequence_fraction':float((lengths<32).mean()),
                           'benign_captures':benign.capture_id.nunique(),'benign_groups':benign.split_group.nunique(),
                           'empirical_fpr_step':1/n if n else None,
                           'zero_fp_95_upper_iid_only':1-.05**(1/n) if n else None}
        # Metadata is explicitly separate from X and may be joined for evaluation only.
        meta.to_parquet(ROOT/f'data/processed/xgboost/common/{split}_metadata.parquet',index=False)
        meta.to_parquet(ROOT/f'data/processed/cnn1d/{split}_metadata.parquet',index=False)
        if_meta=meta.loc[meta.label==0] if split=='train' else meta
        if_meta.to_parquet(ROOT/f'data/processed/isolation_forest/common/{split}_metadata.parquet',index=False)
    metadata=pd.concat(metadata,ignore_index=True)
    sources=pd.DataFrame(source_rows)
    malicious_train=sources[(sources.split=='train') & (sources.label==1)]
    dominant=malicious_train.loc[malicious_train.flows.idxmax()]
    dominant_text=f"{dominant.source} chiếm ưu thế trong malicious train ({int(dominant.flows):,} flow; {dominant.share_within_class:.2%}). "
    sources.to_csv(qa/'common_source_distribution.csv',index=False)
    seq=pd.DataFrame(sequence_rows)
    types=pd.crosstab(packets.sample_id,packets.tls_observation)
    seq=seq.merge(types,left_on='sample_id',right_index=True,how='left').fillna(0)
    seq.to_csv(qa/'sequence_statistics.csv',index=False)
    grouped=seq.groupby(['split','source','label']).agg(flows=('sample_id','size'),
        mean_valid_packets=('valid_packets','mean'),mean_padding_fraction=('padding_fraction','mean'))
    grouped.to_csv(qa/'sequence_statistics_by_source.csv')
    p=packets.merge(metadata[['sample_id','split','source','label']],on='sample_id',validate='many_to_one')
    tls=p.groupby(['split','source','label','tls_observation']).size().rename('packets').reset_index()
    tls.to_csv(qa/'tls_observation_counts.csv',index=False)
    p['no_payload']=p.tls_observation.eq('no_tcp_payload')
    payload_summary=p.groupby(['split','source','label']).agg(valid_packets=('sample_id','size'),
        no_tcp_payload_packets=('no_payload','sum'),no_tcp_payload_fraction=('no_payload','mean')).reset_index()
    payload_summary.to_csv(qa/'tcp_payload_by_source.csv',index=False)
    status={'splits':split_info,'tls_observation_counts':{str(k):int(v) for k,v in packets.tls_observation.value_counts().items()},
            'tls_scope':'record headers visible at packet boundaries; no TCP reassembly/decryption; outer type 23 is not proof of application phase',
            'cnn_mask_runtime_status':'UNVERIFIABLE: model.py and train.py contain TODO only',
            'comparison_scope':'three detection methods with their respective input representations; not isolated architecture comparison'}
    (qa/'training_audit.json').write_text(json.dumps(status,indent=2,allow_nan=False)+'\n')
    print(json.dumps(status,indent=2))

if __name__=='__main__': audit()
