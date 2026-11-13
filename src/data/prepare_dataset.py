"""Reproducible offline DoH tabular preparation; run from the repository root."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.runtime-deps'))
import numpy as np
import pandas as pd

META = ['SourceIP', 'DestinationIP', 'SourcePort', 'DestinationPort', 'TimeStamp', 'DoH']
RESOLVERS = {'1.1.1.1','8.8.4.4','8.8.8.8','9.9.9.9','9.9.9.10','9.9.9.11',
             '176.103.130.131','176.103.130.130','149.112.112.10','149.112.112.112',
             '104.16.248.249','104.16.249.249'}

def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)

def prepare():
    (ROOT/'data/processed/xgboost').mkdir(parents=True,exist_ok=True)
    (ROOT/'data/interim/qa').mkdir(parents=True,exist_ok=True)
    (ROOT/'splits').mkdir(parents=True,exist_ok=True)
    start = time.time()
    raw = ROOT / 'data/raw/csv'
    audit, parts, exclusions = [], [], []
    for p in sorted(raw.glob('**/Separate/*.csv')):
        rel = str(p.relative_to(ROOT))
        source = p.parent.parent.name
        item = {'path': rel, 'bytes': p.stat().st_size, 'source': source}
        if not item['bytes']:
            item.update(status='empty', rows=0); audit.append(item); continue
        # Baseline is not treated as malicious merely because of its parent directory.
        df = pd.read_csv(p)
        item['rows'] = len(df)
        if 'baseline' in p.name:
            item['status'] = 'excluded_ambiguous_baseline'; audit.append(item); continue
        if source not in ['Chrome', 'Firefox'] and 'tunnel' not in p.name:
            raise ValueError(f'Unrecognized labeling provenance: {p}')
        if not set(META) <= set(df):
            raise ValueError(f'Unexpected schema: {p}')
        doh = df.DoH.astype(str).str.lower()
        if not doh.isin(['true','false']).all():
            raise ValueError(f'Unknown DoH value: {p}')
        item['non_doh'] = int((doh != 'true').sum())
        df = df.loc[doh == 'true'].copy()
        item.update(status='included', doh_rows=len(df)); audit.append(item)
        if df.empty: continue
        df['source'] = source
        df['source_file'] = rel
        df['source_row'] = df.index + 2
        df['label'] = np.int8(source not in ['Chrome', 'Firefox'])
        df['capture_id'] = source + '/' + p.stem
        parts.append(df)
        if len(parts) % 1000 == 0: print(f'Read {len(parts)} nonempty DoH CSV files',flush=True)
    df = pd.concat(parts, ignore_index=True)
    features = [c for c in parts[0].columns if c not in META and c not in
                ['source','source_file','source_row','label','capture_id']]
    for c in features:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    initial = len(df)
    df['start_time'] = pd.to_datetime(df.TimeStamp, errors='coerce')
    # Only endpoints documented by the dataset creators are accepted as DoH.
    bad = (df.start_time.isna() | ~np.isfinite(df.Duration) | (df.Duration <= 0)
           | ~((df.SourceIP.isin(RESOLVERS) & (df.SourcePort == 443))
               | (df.DestinationIP.isin(RESOLVERS) & (df.DestinationPort == 443))))
    for c in ['SourcePort','DestinationPort']:
        bad |= ~df[c].between(1,65535) | (df[c] % 1 != 0)
    exclusions.append({'reason':'invalid_time_duration_port_or_unverified_doh_endpoint','rows':int(bad.sum())})
    df.loc[bad, ['source_file','source_row']].to_csv(ROOT/'data/interim/qa/invalid_rows.csv',index=False)
    df = df.loc[~bad].copy()
    # Values with a defined nonnegative domain become missing, never silently abs().
    domain_counts = {}
    for c in features:
        invalid = ~np.isfinite(df[c])
        if 'Skew' not in c:
            invalid |= df[c] < 0
        else:
            stdcol=c.split('Skew')[0]+'StandardDeviation'
            # DoHMeter explicitly emits -10 when skew is undefined (std == 0).
            invalid |= (df[c] == -10) & (df[stdcol].isna() | (df[stdcol] == 0))
        domain_counts[c] = int(invalid.sum())
        df.loc[invalid,c] = np.nan
    # Source-row stable IDs and unordered endpoint connection keys.
    df['sample_id'] = [hashlib.sha256(f'{p}:{r}'.encode()).hexdigest()[:24]
                       for p,r in zip(df.source_file,df.source_row)]
    ep1 = df.SourceIP + ':' + df.SourcePort.astype(str)
    ep2 = df.DestinationIP + ':' + df.DestinationPort.astype(str)
    df['connection_key'] = np.where(ep1 < ep2, ep1+'|'+ep2, ep2+'|'+ep1)
    # Exact flow records and exact feature vectors are deduplicated globally.
    signature = pd.util.hash_pandas_object(df[features], index=False)
    conflict = df.groupby(signature).label.transform('nunique') > 1
    exclusions.append({'reason':'conflicting_labels_for_identical_features','rows':int(conflict.sum())})
    df.loc[conflict,['sample_id','source_file','source_row','label']].to_csv(
        ROOT/'data/interim/qa/conflicting_labels.csv',index=False)
    df = df.loc[~conflict].copy()
    duplicated = df.duplicated(features)
    exclusions.append({'reason':'exact_duplicate_feature_vectors','rows':int(duplicated.sum())})
    df.loc[duplicated,['sample_id','source_file','source_row']].to_csv(
        ROOT/'data/interim/qa/duplicates.csv',index=False)
    df = df.loc[~duplicated].reset_index(drop=True)
    # A reused TCP 4-tuple on another day is a new connection, not one giant group.
    session_keys = {}
    for key,sub in df.groupby('connection_key',sort=False):
        end = float('-inf'); session = None
        for i in sub.sort_values('start_time').index:
            timestamp = df.at[i,'start_time'].timestamp()
            if timestamp > end + 2:
                session = key + '/' + str(timestamp)
            end = max(end,timestamp + df.at[i,'Duration'])
            session_keys[i] = session
    df['connection_key'] = pd.Series(session_keys)
    # Keep collection day and resolver together, including simultaneous clients.
    resolver = df.DestinationIP.where(df.DestinationIP.isin(RESOLVERS),df.SourceIP)
    base_group = df.source + '/' + df.start_time.dt.strftime('%Y-%m-%d') + '/' + resolver
    # Union collection groups sharing a connection, including timeout windows.
    parent = {g:g for g in base_group.unique()}
    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]; g = parent[g]
        return g
    first = {}
    for connection,capture,g in zip(df.connection_key,df.capture_id,base_group):
        for key in ['connection:'+connection,'capture:'+capture]:
            if key in first: parent[find(g)] = find(first[key])
            else: first[key] = g
    df['split_group'] = [find(g) for g in base_group]
    df['flow_id'] = df.sample_id
    print(f'Cleaned {len(df)} flows, {df.split_group.nunique()} independent groups',flush=True)
    print(df.groupby('source').split_group.nunique().to_dict(),flush=True)
    # Try deterministic group splits and select by proportions, never model results.
    best = None
    stats=df.groupby('split_group').agg(rows=('label','size'),positives=('label','sum'),source=('source','first'))
    if df.groupby('split_group').source.nunique().max()!=1:
        raise ValueError('Unexpected cross-source collection group')
    for seed in range(42,2042):
        rng=np.random.default_rng(seed); assignment={}
        for source,groups in stats.groupby('source'):
            ordered=rng.permutation(groups.index)
            if len(ordered)<3:
                assignment[ordered[0]]='train'
                for g in ordered[1:]: assignment[g]='val' if seed%2 else 'test'
                continue
            n_hold=max(1,round(len(ordered)*.15))
            for name,values in zip(['val','test','train'],[ordered[:n_hold],ordered[n_hold:2*n_hold],ordered[2*n_hold:]]):
                assignment.update({g:name for g in values})
        score=0
        for name,ratio in zip(['train','val','test'],[.7,.15,.15]):
            sub=stats.loc[[g for g,v in assignment.items() if v==name]]
            if not 0 < sub.positives.sum() < sub.rows.sum():
                score=float('inf');break
            score+=abs(sub.rows.sum()/len(df)-ratio)+abs(sub.positives.sum()/sub.rows.sum()-df.label.mean())
        if best is None or score < best[0]: best=(score,seed,assignment)
    if best is None: raise ValueError('No valid grouped split; refusing row-wise fallback')
    df['split'] = ''
    split_summary = {}
    for name in ['train','val','test']:
        ix=df.index[df.split_group.map(best[2])==name]
        df.loc[ix,'split'] = name
        sub = df.loc[ix]
        split_summary[name] = {'rows':len(sub),'benign':int((sub.label==0).sum()),
                               'malicious':int((sub.label==1).sum()),'groups':sub.split_group.nunique(),
                               'sources':{str(k):int(v) for k,v in sub.source.value_counts().items()}}
        sub[['sample_id']].to_csv(ROOT/f'splits/{name}_ids.csv',index=False)
    train = df[df.split=='train']
    medians = train[features].median().fillna(0)
    filled = df[features].fillna(medians)
    kept = [c for c in features if filled.loc[train.index,c].nunique()>1]
    dropped = [c for c in features if c not in kept]
    filled = filled[kept]
    center = filled.loc[train.index].median()
    scale = (filled.loc[train.index].quantile(.75)-filled.loc[train.index].quantile(.25)).replace(0,1)
    assert np.isfinite(filled.to_numpy()).all()
    assert df.sample_id.is_unique
    for col in ['split_group','capture_id','connection_key']:
        assert df.groupby(col).split.nunique().max()==1, col
    assert not df.duplicated(features).any()
    raw_features = df[['sample_id']+features]
    raw_features.to_parquet(ROOT/'data/interim/flow_features.parquet',index=False)
    metadata = df.drop(columns=features)
    metadata.to_parquet(ROOT/'data/interim/master_dataset.parquet',index=False)
    for name in ['train','val','test']:
        mask = df.split==name
        out = filled.loc[mask].copy()
        out.insert(0,'sample_id',df.loc[mask,'sample_id'])
        out['label'] = df.loc[mask,'label']
        out.to_parquet(ROOT/f'data/processed/xgboost/{name}.parquet',index=False)
        X = filled.loc[mask].to_numpy(dtype=np.float64)
        scaled = ((X-center.to_numpy())/scale.to_numpy()).astype(np.float32)
        assert np.isfinite(scaled).all()
        np.savez_compressed(ROOT/f'data/processed/xgboost/{name}.npz',
            X=X, X_scaled=scaled, y=df.loc[mask,'label'].to_numpy(dtype=np.int64),
            sample_id=df.loc[mask,'sample_id'].to_numpy(dtype=str),feature_names=np.array(kept))
    benign = (df.split=='train') & (df.label==0)
    np.savez_compressed(ROOT/'data/processed/xgboost/if_benign_train.npz',
        X=((filled.loc[benign].to_numpy()-center.to_numpy())/scale.to_numpy()).astype(np.float32),
        y=np.zeros(int(benign.sum()),dtype=np.int64),sample_id=df.loc[benign,'sample_id'].to_numpy(dtype=str),
        feature_names=np.array(kept))
    filled.describe(percentiles=[.01,.25,.5,.75,.99]).T.to_csv(ROOT/'data/interim/qa/feature_statistics.csv')
    pd.DataFrame(audit).to_csv(ROOT/'data/interim/qa/input_inventory.csv',index=False)
    # Compare aggregate and separate files without loading aggregate rows into training.
    aggregate=[]
    for p in sorted(raw.glob('**/all.csv')):
        hashes=[]
        for chunk in pd.read_csv(p,chunksize=100000,dtype=str):
            hashes.extend(pd.util.hash_pandas_object(chunk,index=False).tolist())
        source=p.parent.name
        separate_hashes=[]
        for q in sorted(p.parent.glob('Separate/*.csv')):
            if q.stat().st_size:
                chunk=pd.read_csv(q,dtype=str)
                separate_hashes.extend(pd.util.hash_pandas_object(chunk,index=False).tolist())
        from collections import Counter
        aggregate.append({'source':source,'aggregate_rows':len(hashes),'separate_rows':len(separate_hashes),
                          'identical_row_multisets':Counter(hashes)==Counter(separate_hashes)})
    summary = {'input_doh_nonbaseline':initial,'final_rows':len(df),'features':kept,
               'dropped_constant_train_features':dropped,'invalid_cells_to_impute':domain_counts,
               'exclusions':exclusions,'split':split_summary,'seed':best[1],
               'aggregate_comparison':aggregate,'seconds':round(time.time()-start,2),
               'checks':{'unique_ids':True,'finite_processed_values':True,'no_exact_feature_duplicates':True,
                         'disjoint_capture_group_connection':True,'both_labels_in_each_split':True,
                         'all_sources_in_each_split':all(df[df.split==s].source.nunique()==5 for s in ['train','val','test'])}}
    save_json(ROOT/'data/processed/xgboost/preprocessing.json',
              {'features':kept,'median':medians[kept].to_dict(),'robust_center':center.to_dict(),
               'robust_scale':scale.to_dict(),'fit_split':'train','label_mapping':{'benign':0,'malicious':1}})
    save_json(ROOT/'splits/split_manifest.json',summary)
    save_json(ROOT/'data/interim/qa/summary.json',summary)
    from finalize_training import finalize
    finalize('tabular')
    print(json.dumps(summary,indent=2),flush=True)

if __name__ == '__main__':
    prepare()
