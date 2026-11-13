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
        for split in ['train', 'val', 'test']:
            paths.extend([base / f'{split}.npz', base / f'{split}_metadata.parquet'])
            if folder != 'data/processed/cnn1d':
                paths.extend([base / f'{split}.parquet', base / f'{split}_ids.csv'])
    missing = [str(p.relative_to(ROOT)) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing handover files: ' + ', '.join(missing))
    paths = sorted(set(paths))
    out = ROOT / 'data/exports/common.tar.gz'
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + '.tmp')
    with tarfile.open(temporary, 'w:gz') as archive:
        for path in paths:
            archive.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
    with tarfile.open(temporary, 'r:gz') as archive:
        assert set(archive.getnames()) == {str(p.relative_to(ROOT)) for p in paths}
        for member in archive.getmembers():
            with archive.extractfile(member) as stream, (ROOT / member.name).open('rb') as source:
                while True:
                    chunk = stream.read(1024 * 1024)
                    assert chunk == source.read(1024 * 1024), member.name
                    if not chunk:
                        break
    temporary.replace(out)
    print(f'{out}: {out.stat().st_size / 2**20:.1f} MiB, {len(paths)} files')


if __name__ == '__main__':
    main()
