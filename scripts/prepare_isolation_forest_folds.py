"""Build the Isolation Forest fold inputs from the common manifests and raw flow features."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    profile = ROOT / 'data/processed/isolation_forest/common'
    manifest = pd.read_parquet(ROOT / 'splits/common_folds.parquet').set_index('sample_id')
    master = pd.read_parquet(ROOT / 'data/interim/master_dataset.parquet').set_index('sample_id')
    features = pd.read_parquet(ROOT / 'data/interim/flow_features.parquet').set_index('sample_id')
    test_path = profile / 'test.npz'
    with np.load(test_path, allow_pickle=False) as archive:
        test = {key: archive[key] for key in archive.files}
    columns = test['feature_names']
    ids = manifest.index.to_numpy(dtype=str)
    test_ids = test['sample_id']
    if not manifest.index.is_unique or not np.array_equal(master.loc[ids, 'label'], manifest.label):
        raise ValueError('Invalid development IDs or labels in fold manifest')
    if not manifest.split_group.equals(master.loc[ids, 'split_group']):
        raise ValueError('Fold groups do not match master metadata')
    if not set(manifest.split_group).isdisjoint(master.loc[test_ids, 'split_group']):
        raise ValueError('Development and test share a collection group')
    if not np.array_equal(master.loc[test_ids, 'label'], test['y']):
        raise ValueError('Test labels do not match master metadata')
    test['X_raw'] = features.loc[test_ids, columns].to_numpy()
    for path, arrays in (
        (profile / 'development.npz', dict(
            X_raw=features.loc[ids, columns].to_numpy(),
            y=manifest.label.to_numpy(dtype=np.int64),
            sample_id=ids,
            feature_names=columns,
        )),
        (test_path, test),
    ):
        temporary = path.with_suffix('.tmp.npz')
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)


if __name__ == '__main__':
    main()
