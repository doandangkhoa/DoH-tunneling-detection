"""Match real PCAP packets to cleaned CSV flows, exporting a bounded CNN subset.

No payload is exported. Supports classic PCAP Ethernet and Linux cooked v1/v2.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
import json
import socket
import struct
import sys
import zipfile
import hashlib
from bisect import bisect_left,bisect_right

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'.runtime-deps'))
sys.path.insert(0,str(Path(__file__).parent))
from prepare_dataset import save_json
import numpy as np
import pandas as pd

def packets(stream, include_tls_metadata=False):
    header=stream.read(24)
    endian={'d4c3b2a1':'<','a1b2c3d4':'>','4d3cb2a1':'<','a1b23c4d':'>'}.get(header[:4].hex())
    if endian is None: raise ValueError('Unsupported PCAP format')
    resolution=1e9 if header[:4].hex() in ['4d3cb2a1','a1b23c4d'] else 1e6
    link=struct.unpack(endian+'I',header[20:24])[0]
    if link not in [1,113,276]: raise ValueError(f'Unsupported link type {link}')
    unpack=struct.Struct(endian+'IIII').unpack
    while True:
        record=stream.read(16)
        if not record: return
        if len(record)!=16: raise ValueError('Truncated packet header')
        sec,usec,caplen,wirelen=unpack(record)
        if caplen>16*1024*1024: raise ValueError('Invalid packet size')
        b=stream.read(caplen)
        if len(b)!=caplen: raise ValueError('Truncated packet body')
        off={1:14,113:16,276:20}[link]
        proto=b[12:14] if link==1 else b[14:16] if link==113 else b[:2]
        if link==1:
            while proto in [b'\x81\x00',b'\x88\xa8'] and len(b)>=off+4:
                proto=b[off+2:off+4];off+=4
        if proto!=b'\x08\x00' or len(b)<off+20: continue
        if b[off]>>4!=4 or b[off+9]!=6: continue
        if struct.unpack('!H',b[off+6:off+8])[0]&0x3fff: continue
        tcp=off+(b[off]&15)*4
        if tcp<off+20 or len(b)<tcp+20: continue
        sport,dport=struct.unpack('!HH',b[tcp:tcp+4])
        if sport!=443 and dport!=443: continue
        a=(socket.inet_ntoa(b[off+12:off+16]),sport)
        c=(socket.inet_ntoa(b[off+16:off+20]),dport)
        key=tuple(sorted((a,c)))
        payload=tcp+(b[tcp+12]>>4)*4
        tls=(len(b)>=payload+3 and b[payload] in [20,21,22,23] and b[payload+1]==3
             and b[payload+2] in [0,1,2,3,4])
        # IP total length: independent of Ethernet vs Linux cooked capture headers.
        length=struct.unpack('!H',b[off+2:off+4])[0]
        row=(sec+usec/resolution,key,a,length,tls,bool(b[tcp+13]&2 and not b[tcp+13]&16))
        if include_tls_metadata:
            # Observable record-header types only, no TCP reassembly or decryption.
            types=set();cursor=payload
            stop=min(len(b),off+length)
            while cursor+5<=stop:
                kind=b[cursor];version=b[cursor+1:cursor+3]
                record_length=int.from_bytes(b[cursor+3:cursor+5],'big')
                if kind not in [20,21,22,23] or version[0]!=3 or version[1]>4 or record_length>18432: break
                types.add(kind);cursor+=5+record_length
            row+=('handshake_and_outer23' if 22 in types and 23 in types else
                   'visible_handshake' if 22 in types else 'outer_application_data' if 23 in types else
                   'other_tls' if types else 'no_tcp_payload' if stop<=payload else 'unclassified_payload',)
        yield row

def matching_windows(values,index,timestamp):
    """Return every overlapping CSV window, retaining ambiguous matches for QA."""
    starts,prefix_ends=index
    lo=bisect_left(prefix_ends,timestamp)
    hi=bisect_right(starts,timestamp)
    return [v for v in values[lo:hi] if timestamp<=v[1]+1]

def capture_packets(stream,errors,include_tls_metadata=False):
    """Retain complete records before a truncated tail; never suppress other errors."""
    try:
        yield from packets(stream,include_tls_metadata=include_tls_metadata)
    except ValueError as exc:
        if str(exc) not in ['Truncated packet body','Truncated packet header']: raise
        errors.append(str(exc))

def calibration_bucket(key,timestamp):
    # CSV start times have integer-second resolution. Within this bucket the
    # admissible start seconds are identical, including the 1.05s boundary.
    second=int(timestamp)
    return key,second,timestamp-(second-1)<1.05

def prepare():
    (ROOT/'data/processed/cnn1d').mkdir(parents=True,exist_ok=True)
    (ROOT/'data/interim/common').mkdir(parents=True,exist_ok=True)
    extraction=json.loads((ROOT/'configs/packet_extraction.json').read_text())
    full_captures={(r['archive'],r['member']) for r in extraction['validation_benign_full_captures']}
    previous={}
    for split in ['train','val','test']:
        path=ROOT/f'data/processed/cnn1d/{split}.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as z: previous[split]=set(z['sample_id'])
    df=pd.read_parquet(ROOT/'data/interim/master_dataset.parquet')
    flows=pd.read_parquet(ROOT/'data/interim/flow_features.parquet')[['sample_id','Duration']]
    df=df.merge(flows,on='sample_id',validate='one_to_one')
    row_split=df.split.to_numpy(); row_label=df.label.to_numpy()
    row_ids=df.sample_id.to_numpy()
    epochs=df.start_time.to_numpy(dtype='datetime64[ns]').astype('int64')/1e9
    lookups={}
    indexes={}
    for source,sub in df.groupby('source'):
        lookup=defaultdict(list)
        for i,r in sub.iterrows():
            key=tuple(sorted(((r.SourceIP,int(r.SourcePort)),(r.DestinationIP,int(r.DestinationPort)))))
            lookup[key].append((epochs[i],epochs[i]+r.Duration,i))
        for values in lookup.values(): values.sort()
        lookups[source]=lookup
        indexes[source]={k:([v[0] for v in vs],np.maximum.accumulate([v[1]+1 for v in vs]).tolist())
                         for k,vs in lookup.items()}
    lookup_hash=hashlib.sha256((ROOT/'data/interim/master_dataset.parquet').read_bytes()).hexdigest()
    cache_dir=ROOT/'data/interim/packet_cache'
    cache_dir.mkdir(exist_ok=True)
    jobs=[]; zip_entries=[]
    for archive in sorted((ROOT/'data/raw/pcap').glob('**/*.zip')):
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                if not info.is_dir():
                    zip_entries.append({'archive':str(archive.relative_to(ROOT)),'member':info.filename,
                                        'uncompressed_bytes':info.file_size,'compressed_bytes':info.compress_size,
                                        'crc32':f'{info.CRC:08x}'})
            members=[i for i in z.infolist() if i.filename.endswith('.pcap')]
            if not members: continue
            source=next((s for s in lookups if s.lower() in archive.name.lower()),None)
            if source is None: continue
            if source in ['Chrome','Firefox']:
                # Browser window is controlled by configuration; null scans the full capture.
                selected=members
            else:
                # Deterministic sample across resolver/scenario configurations.
                selected=[i for i in sorted(members,key=lambda i:i.filename)
                          if 'tunnel' in i.filename]
                limit=extraction.get('tunnel_captures_per_zip',{}).get(source,8)
                if limit is not None: selected=selected[::max(1,len(selected)//limit)][:limit]
            jobs.extend((archive,source,m.filename) for m in selected)
    pd.DataFrame(zip_entries).to_csv(ROOT/'data/interim/qa/zip_inventory.csv',index=False)
    records=[];audit=[]
    for job,(archive,source,member) in enumerate(jobs):
        lookup=lookups[source]
        expanded_window=extraction['browser_window_seconds']
        with zipfile.ZipFile(archive) as z:
            info=z.getinfo(member)
            cache_key=hashlib.sha256(f'{lookup_hash}|parser-v3-1|{archive.name}|{member}|{info.CRC}|{expanded_window}'.encode()).hexdigest()
        cache_path=cache_dir/f'{cache_key}.json'
        if cache_path.exists():
            try: cached=json.loads(cache_path.read_text())
            except json.JSONDecodeError: cache_path.rename(cache_path.with_suffix('.incomplete'))
            else:
                records.extend(cached['records']);audit.append(cached['audit'])
                if job%100==0: print(f'[{job+1}/{len(jobs)}] cache {source}',flush=True)
                continue
        if info.file_size<24:
            item={'archive':archive.name,'member':member,'source':source,'status':'empty_or_short_capture',
                  'bytes':info.file_size,'matched_flows':0}
            audit.append(item);save_json(cache_path,{'records':[],'audit':item})
            continue
        # Calibrate only integer-hour clock differences using tuple/start matches.
        votes=Counter(); seen=set(); checked_seconds=set();first_time=None;calibration_errors=[]
        with zipfile.ZipFile(archive) as z, z.open(member) as stream:
            for n,(t,key,a,length,tls,syn) in enumerate(capture_packets(stream,calibration_errors)):
                if first_time is None: first_time=t
                bucket=calibration_bucket(key,t)
                if key in lookup and bucket not in checked_seconds:
                    checked_seconds.add(bucket)
                    for start,end,i in lookup[key]:
                        hours=round((t-start)/3600)
                        if (i,hours) not in seen and -12<=hours<=14 and 0 <= t-start-hours*3600 < 1.05:
                            seen.add((i,hours))
                            votes[hours]+=1
                clear=(bool(votes) and votes.most_common(1)[0][1]>=3 and
                       (len(votes)==1 or votes.most_common(1)[0][1]>2*votes.most_common(2)[1][1]))
                if source in ['Chrome','Firefox']:
                    if n>=1000000 or t-first_time>7200 or (clear and (n>=50000 or t-first_time>300)): break
                elif n>=50000 or t-first_time>300: break
        if not votes or votes.most_common(1)[0][1]<3:
            audit.append({'archive':archive.name,'member':member,'status':'insufficient_clock_matches',
                          'source':source,'votes':dict(votes),'calibration_read_errors':calibration_errors})
            save_json(cache_path,{'records':[],'audit':audit[-1]})
            if job%100==0: print(f'[{job+1}/{len(jobs)}] skip {source}: clock votes {dict(votes)}',flush=True)
            continue
        offset,count=votes.most_common(1)[0]
        if len(votes)>1 and count<=votes.most_common(2)[1][1]*2:
            audit.append({'archive':archive.name,'member':member,'source':source,'status':'ambiguous_clock','votes':dict(votes)})
            save_json(cache_path,{'records':[],'audit':audit[-1]});continue
        groups={}; ambiguous=0; tainted=set(); cut=None; scanned=0; capture_start=None
        read_errors=[];last_timestamp=None
        reasons=Counter(); seen_candidates=set()
        expand=(archive.name,member) in full_captures
        with zipfile.ZipFile(archive) as z, z.open(member) as stream:
            for t,key,a,length,tls,syn,tls_kind in capture_packets(stream,read_errors,include_tls_metadata=True):
                scanned+=1
                last_timestamp=t
                if cut is None:
                    capture_start=t
                    cut=t+expanded_window if source in ['Chrome','Firefox'] and not expand and expanded_window is not None else float('inf')
                if t>cut: break
                local=t-offset*3600
                values=lookup.get(key)
                if not values: continue
                candidates=matching_windows(values,indexes[source][key],local)
                if len(candidates)!=1:
                    if len(candidates)>1:
                        ambiguous+=1
                        for _,_,idx in candidates:
                            seen_candidates.add(idx)
                            if idx not in groups or len(groups[idx]['packets'])<32:
                                tainted.add(idx)
                    continue
                start,end,i=candidates[0]
                seen_candidates.add(i)
                if expand and expanded_window is not None and (row_split[i]!='val' or row_label[i]!=0) and t>capture_start+expanded_window:
                    continue
                if i not in groups:
                    # Do not accept flows whose beginning is absent in this capture.
                    if not 0<=local-start<1.05:
                        reasons['packet_observations_missing_flow_start']+=1
                        continue
                    groups[i]={'first_endpoint':a,'syn':syn,'packets':[],'tls':False,'end':end+offset*3600}
                g=groups[i]
                if len(g['packets'])<32:
                    g['packets'].append((t,length,1 if a==g['first_endpoint'] else -1,tls_kind))
                    g['tls']|=tls
        if read_errors and last_timestamp is not None: cut=min(cut,last_timestamp)
        accepted=0; capture_records=[]
        for i,g in groups.items():
            ps=sorted(g['packets'])
            if i in tainted: reasons['ambiguous_first_sequence']+=1;continue
            if len(ps)<4: reasons['fewer_than_4_packets']+=1;continue
            if not g['tls']: reasons['no_observed_tls_header']+=1;continue
            if len(ps)<32 and g['end']+1>cut: reasons['truncated_at_read_window']+=1;continue
            accepted+=1
            capture_records.append((row_ids[i],ps,archive.name,member,offset,g['syn']))
        audit.append({'archive':archive.name,'member':member,'source':source,'status':'processed',
                      'offset_hours':offset,'clock_votes':count,'scanned_tls_port_packets':scanned,
                      'matched_flows':accepted,'ambiguous_packet_assignments_skipped':ambiguous,
                      'candidate_unique_flows':len(seen_candidates),'started_flows':len(groups),
                      'candidate_flows_without_start':len(seen_candidates-set(groups)),
                      'capture_read_errors':read_errors,
                      'rejected_started_flows':{k:v for k,v in reasons.items() if k!='packet_observations_missing_flow_start'},
                      'packet_observations_missing_flow_start':reasons['packet_observations_missing_flow_start']})
        records.extend(capture_records)
        save_json(cache_path,{'records':capture_records,'audit':audit[-1]})
        if job<40 or job%50==0: print(f'[{job+1}/{len(jobs)}] {source} {member}: matched={accepted} offset={offset}',flush=True)
    # A flow captured in multiple files is exported only once.
    unique={}
    for record in records:
        sid=record[0]
        if sid not in unique or len(record[1])>len(unique[sid][1]): unique[sid]=record
    ids=sorted(unique)
    split_by_id=df.set_index('sample_id')['split'].to_dict()
    expanded_sources=set(df.source.unique())
    new_ids_by_split={}
    for split in ['train','val','test']:
        current={sid for sid in ids if split_by_id[sid]==split}
        if split in previous:
            assert previous[split]<=current,f'{split}: old samples lost during expansion'
            new_ids_by_split[split]=current-previous[split]
            assert df.set_index('sample_id').loc[list(new_ids_by_split[split]),'source'].isin(expanded_sources).all()
    current_val={sid for sid in ids if split_by_id[sid]=='val'}
    if 'val' in previous and extraction.get('dataset_version')!='common':
        assert previous['val']<=current_val
        added=current_val-previous['val']
        assert df.set_index('sample_id').loc[list(added),'label'].eq(0).all()
        expansion_path=ROOT/'data/interim/qa/validation_expansion.json'
        expansion={
            'previous_val_rows':len(previous['val']),'new_val_rows':len(current_val),
            'added_benign_rows':len(added),'train_test_membership_unchanged':True,
            'split_policy':'unchanged parent manifests; full scan of two existing validation captures'}
        if added or not expansion_path.exists(): save_json(expansion_path,expansion)
    save_json(ROOT/'data/interim/qa/common_expansion.json',{
        'dataset_version':extraction.get('dataset_version'), 'parent_split_unchanged':True,
        'old_samples_preserved':True,'sources_expanded':sorted(expanded_sources),
        'previous_rows':{s:len(v) for s,v in previous.items()},
        'added_rows':{s:len(v) for s,v in new_ids_by_split.items()},
        'policy':'all available browser and tunnel captures; parent groups unchanged'})
    save_json(ROOT/'data/interim/qa/packet_audit_partial.json',{'jobs':len(jobs),'captures':audit})
    if not ids: raise ValueError('No verified PCAP to CSV mappings')
    metadata=df.set_index('sample_id').loc[ids]
    X=np.zeros((len(ids),32,3),dtype=np.float32)
    mask=np.zeros((len(ids),32),dtype=bool)
    packet_rows=[];mapping=[]
    for n,sid in enumerate(ids):
        _,ps,archive,member,offset,syn=unique[sid]
        previous=ps[0][0]
        for j,(timestamp,length,direction,tls_kind) in enumerate(ps):
            iat=timestamp-previous
            X[n,j]=[length,direction,iat];mask[n,j]=True;previous=timestamp
            packet_rows.append((sid,j,timestamp,length,direction,iat,tls_kind))
        mapping.append({'sample_id':sid,'archive':archive,'member':member,'offset_hours':offset,
                        'direction_basis':'syn_initiator' if syn else 'first_observed_endpoint',
                        'packets':len(ps)})
    train=metadata.split.to_numpy()=='train'
    assert train.any()
    # log1p reduces packet length and IAT range; fit moments on valid train packets only.
    transformed=X.copy()
    transformed[:,:,0]=np.log1p(transformed[:,:,0]);transformed[:,:,2]=np.log1p(transformed[:,:,2])
    center=transformed[train][mask[train]].mean(axis=0)
    scale=transformed[train][mask[train]].std(axis=0)
    center[1]=0;scale[1]=1;scale[scale==0]=1
    normalized=(transformed-center)/scale;normalized[~mask]=0
    assert np.isfinite(normalized).all() and (X[:,:,2]>=0).all()
    summary={}
    for split in ['train','val','test']:
        take=metadata.split.to_numpy()==split
        y=metadata.label.to_numpy(dtype=np.int64)[take]
        if set(y)!={0,1}: raise ValueError(f'CNN {split} missing a class')
        np.savez_compressed(ROOT/f'data/processed/cnn1d/{split}.npz',
            X=normalized[take],X_raw=X[take],mask=mask[take],y=y,sample_id=np.array(ids)[take],
            feature_names=np.array(['packet_length','direction','interarrival_time']))
        chosen=metadata.loc[take].reset_index()
        chosen[['sample_id']].to_csv(ROOT/f'data/interim/common/{split}_ids.csv',index=False)
        tab=pd.read_parquet(ROOT/f'data/processed/xgboost/{split}.parquet').set_index('sample_id').loc[chosen.sample_id].reset_index()
        assert np.array_equal(tab.label,y)
        tab.to_parquet(ROOT/f'data/interim/common/{split}.parquet',index=False)
        summary[split]={'rows':int(take.sum()),'benign':int((y==0).sum()),'malicious':int((y==1).sum()),
                        'sources':{str(k):int(v) for k,v in chosen.source.value_counts().items()}}
    pd.DataFrame(packet_rows,columns=['sample_id','packet_index','timestamp','packet_length','direction','interarrival_time','tls_observation']).to_parquet(ROOT/'data/interim/packet_metadata.parquet',index=False)
    pd.DataFrame(mapping).to_csv(ROOT/'data/interim/qa/packet_mapping.csv',index=False)
    save_json(ROOT/'data/processed/cnn1d/preprocessing.json',{'mean':center.tolist(),'std':scale.tolist(),
              'fit_split':'train','layout':'N,L,C','sequence_length':32,'minimum_packets':4,
              'log1p_channels':[0,2],'padding':'zero_after_normalization','mask_required':True,
              'length_unit':'IPv4_total_length_bytes','iat_unit':'seconds',
              'direction':'SYN initiator when first packet is SYN; otherwise first observed endpoint',
              'subset_policy':'all available browser and tunnel captures, subject to configured per-source limits',
              'dataset_version':extraction.get('dataset_version'),
              'extraction_config':'configs/packet_extraction.json'})
    save_json(ROOT/'data/interim/qa/packet_audit.json',{'splits':summary,'captures':audit})
    from finalize_training import finalize
    finalize('common')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': prepare()
