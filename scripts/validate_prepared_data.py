"""Validate persisted training artifacts, independently of preparation."""
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.runtime-deps'))
import numpy as np
import pandas as pd

def check_fit(raw, columns, settings):
    """Recompute parameters from the allowed training rows, not saved arrays."""
    values = raw[columns]
    median = values.median().fillna(0)
    filled = values.fillna(median)
    expected = {
        'median': median,
        'robust_center': filled.median(),
        'robust_scale': (filled.quantile(.75) - filled.quantile(.25)).replace(0, 1),
    }
    for key, value in expected.items():
        actual = pd.Series(settings[key]).reindex(columns)
        assert np.isfinite(actual).all(), key
        assert np.allclose(actual, value), f'Train-only preprocessing mismatch: {key}'


def validate():
    (ROOT/'data/processed/validation.json').write_text(json.dumps({'status': 'VALIDATING'})+'\n')
    master=pd.read_parquet(ROOT/'data/interim/master_dataset.parquet').set_index('sample_id')
    features=pd.read_parquet(ROOT/'data/interim/flow_features.parquet').set_index('sample_id')
    assert master.index.is_unique and features.index.is_unique
    assert set(master.index)==set(features.index)
    assert not features.duplicated().any()
    assert set(master.label)=={0,1}
    for c in ['split_group','connection_key','capture_id']:
        assert master.groupby(c).split.nunique().max()==1,c
    config=json.loads((ROOT/'data/processed/xgboost/preprocessing.json').read_text())
    cols=config['features']
    med=pd.Series(config['median'])
    center=pd.Series(config['robust_center'])
    scale=pd.Series(config['robust_scale'])
    if_config=config['isolation_forest']
    if_med=pd.Series(if_config['median']);if_center=pd.Series(if_config['robust_center']);if_scale=pd.Series(if_config['robust_scale'])
    train_ids=master.index[master.split=='train']
    assert np.allclose(features.loc[train_ids,cols].median().fillna(0),med[cols])
    check_fit(features.loc[train_ids], cols, config)
    benign_train_ids = master.index[(master.split == 'train') & (master.label == 0)]
    check_fit(features.loc[benign_train_ids], cols, if_config)
    common_config = json.loads((ROOT/'data/interim/common/preprocessing.json').read_text())
    assert common_config['features'] == cols
    common_train_ids = pd.read_csv(ROOT/'data/interim/common/train_ids.csv').sample_id
    check_fit(features.loc[common_train_ids], cols, common_config)
    common_benign_ids = master.loc[common_train_ids].query('label == 0').index
    check_fit(features.loc[common_benign_ids], cols, common_config['isolation_forest'])
    common_center = pd.Series(common_config['robust_center'])[cols]
    common_scale = pd.Series(common_config['robust_scale'])[cols]
    common_if = common_config['isolation_forest']
    common_if_median = pd.Series(common_if['median'])[cols]
    common_if_center = pd.Series(common_if['robust_center'])[cols]
    common_if_scale = pd.Series(common_if['robust_scale'])[cols]
    assert json.loads((ROOT/'data/processed/xgboost/common/preprocessing.json').read_text()) == common_config
    for suffix, reference in [('', config), ('common', common_config)]:
        published = json.loads((ROOT/'data/processed/isolation_forest'/suffix/'preprocessing.json').read_text())
        assert published == dict(reference['isolation_forest'], features=cols, label_mapping=reference['label_mapping'])
    cnn_config=json.loads((ROOT/'data/processed/cnn1d/preprocessing.json').read_text())
    with np.load(ROOT/'data/processed/cnn1d/train.npz',allow_pickle=False) as z:
        raw=z['X_raw'].copy()
        raw[:,:,0]=np.log1p(raw[:,:,0]);raw[:,:,2]=np.log1p(raw[:,:,2])
        observed=raw[z['mask']]
        cnn_mean=observed.mean(axis=0);cnn_std=observed.std(axis=0)
        cnn_mean[1]=0;cnn_std[1]=1;cnn_std[cnn_std==0]=1
        assert np.allclose(cnn_mean,cnn_config['mean']) and np.allclose(cnn_std,cnn_config['std'])
    all_ids=set()
    for split in ['train','val','test']:
        table=pd.read_parquet(ROOT/f'data/processed/xgboost/{split}.parquet').set_index('sample_id')
        ids=pd.read_csv(ROOT/f'splits/{split}_ids.csv').sample_id
        assert set(table.index)==set(ids)==set(master.index[master.split==split])
        assert not all_ids.intersection(ids)
        all_ids.update(ids)
        assert np.array_equal(table.label,master.loc[table.index,'label'])
        assert set(table.label)=={0,1}
        assert list(table.columns)==cols+['label']
        assert np.isfinite(table.to_numpy()).all()
        assert not table[cols].duplicated().any(), 'Duplicate vectors introduced by imputation'
        expected=features.loc[table.index,cols].fillna(med)
        assert np.allclose(table[cols],expected)
        with np.load(ROOT/f'data/processed/xgboost/{split}.npz',allow_pickle=False) as z:
            assert np.array_equal(z['sample_id'],table.index.to_numpy())
            assert np.array_equal(z['y'],table.label)
            assert np.allclose(z['X'],table[cols])
            assert np.allclose(z['X_scaled'],(expected-center)/scale,rtol=2e-5,atol=1e-5)
            assert np.allclose(z['X_if'],(features.loc[table.index,cols].fillna(if_med)-if_center)/if_scale,rtol=2e-5,atol=1e-5)
        with np.load(ROOT/f'data/processed/cnn1d/{split}.npz',allow_pickle=False) as z:
            X,mask,y,cnn_ids=z['X'],z['mask'],z['y'],z['sample_id']
            assert X.shape==(len(y),32,3) and mask.shape==(len(y),32)
            assert set(y)=={0,1} and len(set(cnn_ids))==len(cnn_ids)
            assert np.isfinite(X).all() and np.isfinite(z['X_raw']).all()
            assert np.all(np.diff(mask.astype(int),axis=1)<=0)
            transformed=z['X_raw'].copy()
            transformed[:,:,0]=np.log1p(transformed[:,:,0]);transformed[:,:,2]=np.log1p(transformed[:,:,2])
            expected_cnn=(transformed-cnn_mean)/cnn_std;expected_cnn[~mask]=0
            assert np.allclose(X,expected_cnn)
            assert (z['X_raw'][:,:,2]>=0).all() and (X[~mask]==0).all()
            assert (mask.sum(axis=1)>=4).all()
            assert np.array_equal(y,master.loc[cnn_ids,'label'])
            assert (master.loc[cnn_ids,'split']==split).all()
            common=pd.read_parquet(ROOT/f'data/interim/common/{split}.parquet')
            assert np.array_equal(common.sample_id,cnn_ids)
            assert np.array_equal(common.label,y)
            common_config=json.loads((ROOT/'data/interim/common/preprocessing.json').read_text())
            common_med=pd.Series(common_config['median'])
            assert np.allclose(common[cols],features.loc[cnn_ids,cols].fillna(common_med))
            with np.load(ROOT/f'data/interim/common/{split}.npz',allow_pickle=False) as c:
                assert np.array_equal(c['sample_id'],cnn_ids) and np.array_equal(c['y'],y)
                assert np.allclose(c['X'],common[cols])
                assert np.isfinite(c['X_if']).all() and np.isfinite(c['X_scaled']).all()
                assert np.array_equal(c['feature_names'], cols)
                expected_if = (features.loc[cnn_ids,cols].fillna(common_if_median)-common_if_center)/common_if_scale
                assert np.allclose(c['X_if'], expected_if, rtol=2e-5, atol=1e-5)
                assert np.allclose(c['X_scaled'], (common[cols]-common_center)/common_scale, rtol=2e-5, atol=1e-5)

    assert all_ids==set(master.index)
    processed=pd.concat([pd.read_parquet(ROOT/f'data/processed/xgboost/{s}.parquet') for s in ['train','val','test']])
    assert not processed[cols].duplicated().any(), 'Duplicate processed vectors across splits'
    with np.load(ROOT/'data/processed/xgboost/if_benign_train.npz',allow_pickle=False) as z:
        assert (z['y']==0).all() and np.isfinite(z['X']).all()
        assert set(z['sample_id'])==set(master.index[(master.split=='train')&(master.label==0)])
        assert np.allclose(features.loc[z['sample_id'],cols].median().fillna(0),if_med[cols])
        assert np.allclose(z['X'],(features.loc[z['sample_id'],cols].fillna(if_med)-if_center)/if_scale,rtol=2e-5,atol=1e-5)
    common_train_ids=pd.read_csv(ROOT/'data/interim/common/train_ids.csv').sample_id
    assert np.allclose(features.loc[common_train_ids,cols].median().fillna(0),common_med[cols])
    with np.load(ROOT/'data/interim/common/if_benign_train.npz',allow_pickle=False) as z:
        assert (z['y']==0).all() and np.isfinite(z['X']).all()
        assert set(z['sample_id'])==set(master.loc[common_train_ids].query('label == 0').index)
    for profile,reference in [('','data/processed/xgboost'),('common','data/interim/common')]:
        target=ROOT/'data/processed/isolation_forest'/profile
        for split in ['train','val','test']:
            with np.load(ROOT/reference/f'{split}.npz',allow_pickle=False) as source:
                take=source['y']==0 if split=='train' else np.ones(len(source['y']),dtype=bool)
                with np.load(target/f'{split}.npz',allow_pickle=False) as result:
                    assert np.array_equal(result['sample_id'],source['sample_id'][take])
                    assert np.array_equal(result['y'],source['y'][take])
                    assert np.array_equal(result['X'],source['X_if'][take])
                    table=pd.read_parquet(target/f'{split}.parquet')
                    assert np.array_equal(table.sample_id,result['sample_id'])
                    assert np.array_equal(table[cols].to_numpy(),result['X'])
                    assert np.array_equal(pd.read_csv(target/f'{split}_ids.csv').sample_id,result['sample_id'])
                if profile=='common':
                    with np.load(ROOT/f'data/processed/xgboost/common/{split}.npz',allow_pickle=False) as result:
                        assert np.array_equal(result['sample_id'],source['sample_id'])
                        assert np.array_equal(result['X'],source['X']) and np.array_equal(result['y'],source['y'])
                        for key in ['X_scaled','X_if','feature_names']:
                            assert np.array_equal(result[key],source[key])
                        tab = pd.read_parquet(ROOT/f'data/processed/xgboost/common/{split}.parquet')
                        assert np.array_equal(tab.sample_id, result['sample_id'])
                        assert np.array_equal(tab.label, result['y'])
                        assert np.array_equal(tab[cols].to_numpy(), result['X'])
                        assert np.array_equal(pd.read_csv(ROOT/f'data/processed/xgboost/common/{split}_ids.csv').sample_id, result['sample_id'])
    # A small fit proves the arrays are accepted by conventional estimators;
    # no test scores or model choices are produced.
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import IsolationForest
    with np.load(ROOT/'data/processed/xgboost/train.npz',allow_pickle=False) as z:
        pick=np.concatenate([np.flatnonzero(z['y']==label)[:256] for label in [0,1]])
        X,y=z['X'][pick].astype(np.float32),z['y'][pick]
        assert np.isfinite(X).all()
        model=DecisionTreeClassifier(max_depth=3,random_state=42).fit(X,y)
        assert np.isfinite(model.predict_proba(X)).all()
    with np.load(ROOT/'data/processed/xgboost/if_benign_train.npz',allow_pickle=False) as z:
        X=z['X'][:512]
        model=IsolationForest(n_estimators=10,random_state=42,n_jobs=1).fit(X)
        assert np.isfinite(model.score_samples(X)).all()
    packets=pd.read_parquet(ROOT/'data/interim/packet_metadata.parquet')
    assert not packets.duplicated(['sample_id','packet_index']).any()
    assert packets.groupby('sample_id').packet_index.min().eq(0).all()
    assert packets.groupby('sample_id').timestamp.diff().dropna().ge(0).all()
    iats=packets.groupby('sample_id').timestamp.diff().fillna(0)
    assert np.allclose(iats,packets.interarrival_time)
    assert (packets.packet_length>0).all() and set(packets.direction)=={-1,1}
    mappings=pd.read_csv(ROOT/'data/interim/qa/packet_mapping.csv')
    assert mappings.sample_id.is_unique and set(mappings.sample_id)==set(packets.sample_id)
    snapshot=ROOT/'data/interim/versions/common_before_expansion/data/processed/cnn1d'
    if snapshot.exists():
        for split in ['train','val','test']:
            with np.load(snapshot/f'{split}.npz',allow_pickle=False) as old, np.load(ROOT/f'data/processed/cnn1d/{split}.npz',allow_pickle=False) as new:
                old_ids=set(old['sample_id']);new_ids=set(new['sample_id'])
                assert old_ids<=new_ids, 'Original sample lost or moved'
                old_dns={i for i in old_ids if master.at[i,'source']=='dns2tcp'}
                new_dns={i for i in new_ids if master.at[i,'source']=='dns2tcp'}
                assert old_dns<=new_dns,'Original dns2tcp samples lost or moved'
    # A new extraction may add samples but cannot move or drop the previous handover.
    recheck_snapshot=ROOT/'data/interim/versions/common_before_pcap_recheck/data/processed/cnn1d'
    if recheck_snapshot.exists():
        for split in ['train','val','test']:
            with np.load(recheck_snapshot/f'{split}.npz',allow_pickle=False) as old, np.load(ROOT/f'data/processed/cnn1d/{split}.npz',allow_pickle=False) as new:
                assert set(old['sample_id']) <= set(new['sample_id']), 'Previous common samples lost or moved'
    if 'tls_observation' in packets:
        assert packets.tls_observation.isin(['visible_handshake','outer_application_data','handshake_and_outer23',
                                            'other_tls','no_tcp_payload','unclassified_payload']).all()
        for split in ['train','val','test']:
            meta=pd.read_parquet(ROOT/f'data/processed/cnn1d/{split}_metadata.parquet')
            with np.load(ROOT/f'data/processed/cnn1d/{split}.npz',allow_pickle=False) as z:
                assert np.array_equal(meta.sample_id,z['sample_id'])
                assert np.array_equal(meta.label,z['y'])
                assert np.array_equal(meta.source,master.loc[z['sample_id'],'source'])
            tab_meta=pd.read_parquet(ROOT/f'data/processed/xgboost/common/{split}_metadata.parquet')
            assert meta.equals(tab_meta)
            if_meta=pd.read_parquet(ROOT/f'data/processed/isolation_forest/common/{split}_metadata.parquet')
            expected=meta.loc[meta.label==0] if split=='train' else meta
            assert if_meta.reset_index(drop=True).equals(expected.reset_index(drop=True))
    result={'status':'PASS','master_flows':len(master),'packet_flows':packets.sample_id.nunique(),
            'packet_rows':len(packets),'feature_count':len(cols),
            'checks':['persisted_schema','finite_values','binary_labels','unique_ids','exact_feature_duplicates',
                      'group_capture_connection_disjoint','train_only_imputation','train_only_scaling',
                      'benign_only_IF_train','CNN_shapes_masks_padding','packet_order_IAT',
                      'same_common_ids_labels_features','estimator_fit_smoke']}
    (ROOT/'data/processed/validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__': validate()
