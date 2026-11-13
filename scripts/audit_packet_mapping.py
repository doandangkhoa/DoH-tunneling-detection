"""Check persisted packet-to-flow mapping backwards against the master CSV metadata.

Also reread a small deterministic set of raw captures to verify exported packet
lengths, directions and timestamps. This needs A's raw/interim data.
"""
from pathlib import Path
from collections import Counter
import json
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.runtime-deps'))
sys.path.insert(0, str(ROOT / 'src/data'))
import numpy as np
import pandas as pd
from prepare_packets import capture_packets, save_json


def audit():
    master = pd.read_parquet(ROOT / 'data/interim/master_dataset.parquet').set_index('sample_id')
    features = pd.read_parquet(ROOT / 'data/interim/flow_features.parquet').set_index('sample_id')
    mapping = pd.read_csv(ROOT / 'data/interim/qa/packet_mapping.csv').set_index('sample_id')
    packets = pd.read_parquet(ROOT / 'data/interim/packet_metadata.parquet')
    assert mapping.index.is_unique
    assert set(mapping.index) == set(packets.sample_id)
    info = mapping.join(master[['source', 'source_file', 'start_time', 'split', 'label']], validate='one_to_one')
    malicious = ~info.source.isin(['Chrome', 'Firefox'])
    provenance = info.loc[malicious].apply(
        lambda r: Path(r.member).stem == Path(r.source_file).stem, axis=1)
    assert provenance.all(), 'Mapped malicious packet capture differs from original CSV capture'
    first = packets.groupby('sample_id').timestamp.min()
    last = packets.groupby('sample_id').timestamp.max()
    start = pd.Series(info.start_time.to_numpy(dtype='datetime64[ns]').astype('int64') / 1e9, index=info.index) + info.offset_hours * 3600
    assert ((first.loc[info.index] - start >= -1e-6) &
            (first.loc[info.index] - start < 1.05)).all()
    assert (last.loc[info.index] <= start + features.loc[info.index, 'Duration'] + 1 + 1e-6).all()
    assert np.array_equal(packets.groupby('sample_id').size().loc[mapping.index], mapping.packets)
    # Every exported raw tensor position must correspond to the persisted packet row.
    tensor_ids = set()
    for split in ['train', 'val', 'test']:
        with np.load(ROOT / f'data/processed/cnn1d/{split}.npz', allow_pickle=False) as z:
            ids = z['sample_id']
            assert not tensor_ids.intersection(ids)
            tensor_ids.update(ids)
            positions = pd.Series(np.arange(len(ids)), index=ids)
            subset = packets[packets.sample_id.isin(ids)]
            row = positions.loc[subset.sample_id].to_numpy()
            column = subset.packet_index.to_numpy()
            expected = subset[['packet_length', 'direction', 'interarrival_time']].to_numpy()
            assert np.allclose(z['X_raw'][row, column], expected, rtol=1e-5, atol=1e-6)
            assert z['mask'][row, column].all()
            assert int(z['mask'].sum()) == len(subset)
    assert tensor_ids == set(mapping.index)
    inventory = pd.read_csv(ROOT / 'data/interim/qa/zip_inventory.csv')
    capture_stems = {Path(name).stem for name in inventory.member if name.endswith('.pcap')}
    missing_capture = {}
    for source, group in master.groupby('source'):
        if source in ['Chrome', 'Firefox']:
            continue  # Browser CSV names include exporter suffixes, unlike tunnel CSV names.
        missing_capture[source] = int((~group.source_file.map(lambda name: Path(name).stem).isin(capture_stems)).sum())
    coverage = []
    for (split, source), group in master.groupby(['split', 'source']):
        matched = int(group.index.isin(mapping.index).sum())
        coverage.append({'split': split, 'source': source, 'master_flows': len(group),
                         'mapped_flows': matched, 'coverage': matched / len(group)})
    pd.DataFrame(coverage).to_csv(ROOT / 'data/interim/qa/mapping_coverage.csv', index=False)
    # Pick the earliest mapped flow per source/split. No test model scores are read.
    selected = info.assign(first=first).sort_values('first').groupby(['source', 'split']).head(1)
    archives = {p.name: p for p in (ROOT / 'data/raw/pcap').rglob('*.zip')}
    raw_checks = []
    for sid, row in selected.iterrows():
        flow = master.loc[sid]
        wanted = packets.loc[packets.sample_id == sid].sort_values('packet_index')
        endpoints = tuple(sorted(((flow.SourceIP, int(flow.SourcePort)),
                                  (flow.DestinationIP, int(flow.DestinationPort)))))
        observed = []
        errors = []
        with zipfile.ZipFile(archives[row.archive]) as archive, archive.open(row.member) as stream:
            for timestamp, key, endpoint, length, tls, syn in capture_packets(stream, errors):
                if timestamp > wanted.timestamp.max() + 1e-6:
                    break
                if key == endpoints and wanted.timestamp.min() - 1e-6 <= timestamp <= wanted.timestamp.max() + 1e-6:
                    observed.append((timestamp, endpoint, length))
        # Compare multisets so tied timestamps/retransmissions are not collapsed.
        expected_lengths = Counter((float(p.timestamp), int(p.packet_length)) for p in wanted.itertuples())
        actual_lengths = Counter((t, length) for t, endpoint, length in observed)
        assert not (expected_lengths - actual_lengths), sid
        first_endpoint = observed[0][1]
        expected_directed = Counter((float(p.timestamp), int(p.packet_length), int(p.direction)) for p in wanted.itertuples())
        actual_directed = Counter((t, length, 1 if endpoint == first_endpoint else -1) for t, endpoint, length in observed)
        assert not (expected_directed - actual_directed), sid
        raw_checks.append({'sample_id': sid, 'source': row.source, 'split': row.split,
                           'archive': row.archive, 'member': row.member, 'packets': len(wanted)})
        print(f'Raw packet check PASS: {row.source}/{row.split}, {len(wanted)} packets', flush=True)
    capture_audit = json.loads((ROOT / 'data/interim/qa/packet_audit.json').read_text())
    failures = {}
    for source in sorted(master.source.unique()):
        rows = [r for r in capture_audit['captures'] if r.get('source') == source]
        rejections = Counter()
        for r in rows:
            rejections.update(r.get('rejected_started_flows', {}))
        failures[source] = {'capture_status': dict(Counter(r['status'] for r in rows)),
                            'candidate_rejections_not_unique_flows': dict(rejections)}
    result = {'status': 'PASS', 'master_flows': len(master), 'mapped_flows': len(mapping),
              'unmapped_flows': len(master) - len(mapping),
              'coverage': len(mapping) / len(master),
              'malicious_capture_provenance_checks': int(malicious.sum()),
              'all_mapped_time_windows_checked': True, 'all_raw_tensors_match_packet_metadata': True,
              'master_flows_without_same_named_tunnel_capture': missing_capture, 'raw_spot_checks': raw_checks,
              'coverage_by_source_split': coverage, 'capture_diagnostics': failures,
              'scope': 'all persisted mappings checked for time windows and malicious capture name; raw packets spot-checked, not an exhaustive independent raw decode'}
    save_json(ROOT / 'data/interim/qa/packet_mapping_recheck.json', result)
    print(json.dumps({k: result[k] for k in ['status', 'master_flows', 'mapped_flows', 'unmapped_flows', 'coverage']}, indent=2))


if __name__ == '__main__':
    audit()
