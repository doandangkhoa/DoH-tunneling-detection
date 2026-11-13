"""Build ready-to-fit profiles; IF preprocessing is learned from benign train only."""
from pathlib import Path
import json
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'.runtime-deps'))
import numpy as np
import pandas as pd

def finalize(profile):
    folder=ROOT/'data/processed/xgboost' if profile=='tabular' else ROOT/'data/interim/common'
    folder.mkdir(parents=True,exist_ok=True)
    metadata=pd.read_parquet(ROOT/'data/interim/master_dataset.parquet').set_index('sample_id')
    raw=pd.read_parquet(ROOT/'data/interim/flow_features.parquet').set_index('sample_id')
    columns=json.loads((ROOT/'data/processed/xgboost/preprocessing.json').read_text())['features']
    prefix=ROOT/'splits' if profile=='tabular' else folder
    ids={s:pd.read_csv(prefix/f'{s}_ids.csv').sample_id.to_numpy() for s in ['train','val','test']}
    train=raw.loc[ids['train'],columns]
    benign_ids=metadata.loc[ids['train']].query('label == 0').index
    def fit(values):
        median=values.median().fillna(0)
        filled=values.fillna(median)
        center=filled.median()
        scale=(filled.quantile(.75)-filled.quantile(.25)).replace(0,1)
        return median,center,scale
    median,center,scale=fit(train)
    if_median,if_center,if_scale=fit(raw.loc[benign_ids,columns])
    for split in ids:
        sid=ids[split]; y=metadata.loc[sid,'label'].to_numpy(dtype=np.int64)
        X=raw.loc[sid,columns].fillna(median)
        X_if=((raw.loc[sid,columns].fillna(if_median)-if_center)/if_scale).to_numpy(dtype=np.float32)
        X_scaled=((X-center)/scale).to_numpy(dtype=np.float32)
        assert np.isfinite(X).all().all() and np.isfinite(X_if).all() and np.isfinite(X_scaled).all()
        table=X.reset_index();table['label']=y
        table.to_parquet(folder/f'{split}.parquet',index=False)
        np.savez_compressed(folder/f'{split}.npz',X=X.to_numpy(),X_scaled=X_scaled,X_if=X_if,
                            y=y,sample_id=sid.astype(str),feature_names=np.array(columns))
        if split=='train':
            take=y==0
            np.savez_compressed(folder/'if_benign_train.npz',X=X_if[take],y=y[take],
                                sample_id=sid[take].astype(str),feature_names=np.array(columns))
    info={'features':columns,'median':median.to_dict(),'robust_center':center.to_dict(),
          'robust_scale':scale.to_dict(),'fit_split':'train','label_mapping':{'benign':0,'malicious':1},
          'isolation_forest':{'fit_split':'benign_train_only','median':if_median.to_dict(),
                              'robust_center':if_center.to_dict(),'robust_scale':if_scale.to_dict()}}
    (folder/'preprocessing.json').write_text(json.dumps(info,indent=2,allow_nan=False)+'\n')
    publish(profile,folder,info)
    print(f'Finalized {profile}: train={len(ids["train"])} benign_train={len(benign_ids)}',flush=True)

def publish(profile,folder,info):
    """Expose algorithm-specific arrays in the project's original directories."""
    if_folder=ROOT/'data/processed/isolation_forest'
    if profile=='common':
        xgb_folder=ROOT/'data/processed/xgboost/common'
        xgb_folder.mkdir(parents=True,exist_ok=True)
        for name in ['train','val','test']:
            for suffix in ['.npz','.parquet','_ids.csv']:
                shutil.copy2(folder/f'{name}{suffix}',xgb_folder/f'{name}{suffix}')
        shutil.copy2(folder/'preprocessing.json',xgb_folder/'preprocessing.json')
        if_folder=if_folder/'common'
    if_folder.mkdir(parents=True,exist_ok=True)
    for split in ['train','val','test']:
        with np.load(folder/f'{split}.npz',allow_pickle=False) as z:
            take=z['y']==0 if split=='train' else np.ones(len(z['y']),dtype=bool)
            X=z['X_if'][take];y=z['y'][take];sid=z['sample_id'][take]
            np.savez_compressed(if_folder/f'{split}.npz',X=X,y=y,sample_id=sid,feature_names=z['feature_names'])
            table=pd.DataFrame(X,columns=z['feature_names'])
            table.insert(0,'sample_id',sid);table['label']=y
            table.to_parquet(if_folder/f'{split}.parquet',index=False)
            pd.DataFrame({'sample_id':sid}).to_csv(if_folder/f'{split}_ids.csv',index=False)
    config=dict(info['isolation_forest'],features=info['features'],label_mapping=info['label_mapping'])
    (if_folder/'preprocessing.json').write_text(json.dumps(config,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':
    finalize('tabular')
    finalize('common')
