"""Package training arrays and one shared guide, without raw or QA reports."""
from pathlib import Path
import json
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    validation = json.loads((ROOT / 'data/processed/validation.json').read_text())
    assert validation['status'] == 'PASS'
    paths = [ROOT / 'docs/TRAINING_GUIDE.md']
    for folder in ['data/processed/cnn1d', 'data/processed/xgboost/common',
                   'data/processed/isolation_forest/common']:
        base = ROOT / folder
        paths.append(base / 'preprocessing.json')
        development = base / 'development.npz'
        if development.exists():
            paths.extend([development, ROOT / 'splits/common_folds.parquet'])
            if folder == 'data/processed/xgboost/common':
                paths.append(ROOT / 'data/interim/flow_features.parquet')
        for split in (['test'] if development.exists() else ['train', 'val', 'test']):
            paths.extend([base / f'{split}.npz', base / f'{split}_metadata.parquet'])
            if folder != 'data/processed/cnn1d':
                paths.extend([base / f'{split}.parquet', base / f'{split}_ids.csv'])
    # Legacy handovers still use separate train and validation files.
    # Their canonical copies are available outside the XGBoost profile.
    sources = {}
    for path in paths:
        source = path
        if not path.exists() and path.parent == ROOT / 'data/processed/xgboost/common':
            if path.name.startswith(('train', 'val')):
                folder = 'data/processed/cnn1d' if path.name.endswith('_metadata.parquet') else 'data/interim/common'
                source = ROOT / folder / path.name
        sources[path] = source
    missing = [str(p.relative_to(ROOT)) for p, source in sources.items() if not source.is_file()]
    if missing:
        raise FileNotFoundError('Missing handover files: ' + ', '.join(missing))
    paths = sorted(set(paths))
    out = ROOT / 'data/exports/common.tar.gz'
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + '.tmp')
    with tarfile.open(temporary, 'w:gz') as archive:
        for path in paths:
            archive.add(sources[path], arcname=str(path.relative_to(ROOT)), recursive=False)
    with tarfile.open(temporary, 'r:gz') as archive:
        assert set(archive.getnames()) == {str(p.relative_to(ROOT)) for p in paths}
        for member in archive.getmembers():
            with archive.extractfile(member) as stream, sources[ROOT / member.name].open('rb') as source:
                while True:
                    chunk = stream.read(1024 * 1024)
                    assert chunk == source.read(1024 * 1024), member.name
                    if not chunk:
                        break
    temporary.replace(out)
    print(f'{out}: {out.stat().st_size / 2**20:.1f} MiB, {len(paths)} files')


if __name__ == '__main__':
    main()
