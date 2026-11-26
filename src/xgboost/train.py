"""Fit XGBoost and select its threshold from grouped out-of-fold scores."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

config = yaml.safe_load(Path('configs/xgboost.yaml').read_text())
output = config['output']
model_dir = Path(output['model_directory'])
experiment_dir = Path(output['experiment_directory'])
model_dir.mkdir(parents=True, exist_ok=True)
experiment_dir.mkdir(parents=True, exist_ok=True)
recall_target = float(config['threshold']['minimum_recall'])
common = Path(config['data']['directory'])
with np.load(common / config['data']['development_file'], allow_pickle=False) as development:
    cols = development['feature_names']
    ids = pd.Index(development['sample_id'], name='sample_id')
    labels = development['y']
    raw = pd.DataFrame(development['X_raw'], index=ids, columns=cols)
with np.load(common / 'test.npz', allow_pickle=False) as test:
    test_ids = pd.Index(test['sample_id'])

manifest = pd.read_parquet(config['data']['fold_manifest']).set_index('sample_id')
if not ids.is_unique or not manifest.index.is_unique or len(ids) != len(manifest) or not ids.isin(manifest.index).all():
    raise ValueError('Fold manifest must cover each development sample exactly once')
data = manifest.loc[ids].copy()
if not np.array_equal(data.label, labels) or not ids.intersection(test_ids).empty:
    raise ValueError('Fold manifest labels must match development, with test IDs held out')
if data[['split_group', 'source', 'label', 'fold']].isna().any().any():
    raise ValueError('Fold manifest contains missing values')
if set(data.fold) != {0, 1, 2} or data.groupby('split_group').fold.nunique().ne(1).any():
    raise ValueError('Each collection group must belong to exactly one of the three folds')
params = config['model']
def fit_predict(training, scoring):
    features = raw.loc[training.index, cols]
    median = features.median().fillna(0)
    model = xgb.XGBClassifier(**params)
    model.fit(features.fillna(median).to_numpy(), training.label.to_numpy())
    scores = model.predict_proba(raw.loc[scoring, cols].fillna(median).to_numpy())[:,1]
    return model, median, scores

data['score'] = np.nan
for fold in range(3):
    held = data.fold.eq(fold)
    if set(data.loc[held, 'label']) != {0, 1}:
        raise ValueError(f'Fold {fold} must contain both labels')
    _, _, scores = fit_predict(data.loc[~held], data.index[held])
    data.loc[held,'score'] = scores
    print(f'Fold {fold+1} complete', flush=True)
assert np.isfinite(data.score).all()
thresholds = np.unique(data.score)
def rate_at(scores):
    sorted_scores = np.sort(scores)
    return (len(sorted_scores)-np.searchsorted(sorted_scores, thresholds))/len(sorted_scores)
recall = rate_at(data.loc[data.label.eq(1),'score'])
chrome = rate_at(data.loc[data.source.eq('Chrome'),'score'])
firefox = rate_at(data.loc[data.source.eq('Firefox'),'score'])
eligible = np.flatnonzero(recall >= recall_target)
i = min(eligible, key=lambda i: (max(chrome[i],firefox[i]), (chrome[i]+firefox[i])/2, -recall[i], -float(thresholds[i])))
threshold = float(thresholds[i])
selection = dict(threshold=threshold, recall_target=recall_target, oof_recall=float(recall[i]), oof_chrome_fpr=float(chrome[i]), oof_firefox_fpr=float(firefox[i]))
config['threshold']['value'] = threshold
(experiment_dir/'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
data.to_parquet(experiment_dir/'out_of_fold.parquet')
features = raw.loc[data.index, cols]
median = features.median().fillna(0)
model = xgb.XGBClassifier(**params)
model.fit(features.fillna(median).to_numpy(), data.label.to_numpy())
model.save_model(model_dir/'model.json')
(model_dir/'feature_names.json').write_text(json.dumps(cols.tolist(), indent=2))
(model_dir/'preprocessing.json').write_text(json.dumps({'features': cols.tolist(), 'median': median.to_dict()}, indent=2))
summary = dict(model='xgboost', data_profile='common_grouped', n_train=len(data),
               n_test=len(test_ids), threshold_method=config['threshold']['method'], **selection)
(experiment_dir/'run_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(selection)
