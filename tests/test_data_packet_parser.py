"""Offline binary fixtures: capture-link types, timestamps and malformed input."""
import io
from pathlib import Path
import struct
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/data'))
from prepare_packets import packets,matching_windows,capture_packets,calibration_bucket

def fixture(link, endian='<', nanos=False, payload=b'\x16\x03\x03'):
    magic=0xa1b23c4d if nanos else 0xa1b2c3d4
    header=struct.pack(endian+'IHHIIII',magic,2,4,0,0,65535,link)
    ip=bytearray(20);ip[0]=0x45;ip[2:4]=struct.pack('!H',40+len(payload));ip[9]=6
    ip[12:16]=bytes([192,168,20,1]);ip[16:20]=bytes([1,1,1,1])
    tcp=bytearray(20);tcp[:4]=struct.pack('!HH',12345,443);tcp[12]=0x50;tcp[13]=2
    prefix={1:bytes(12)+b'\x08\x00',113:bytes(14)+b'\x08\x00',276:b'\x08\x00'+bytes(18)}[link]
    body=prefix+ip+tcp+payload
    record=struct.pack(endian+'IIII',1600000000,500000000 if nanos else 500000,len(body),len(body))
    return header+record+body

@pytest.mark.parametrize('link',[1,113,276])
@pytest.mark.parametrize('endian',['<','>'])
@pytest.mark.parametrize('nanos',[False,True])
def test_packet_metadata(link,endian,nanos):
    rows=list(packets(io.BytesIO(fixture(link,endian,nanos))))
    assert len(rows)==1
    t,key,source,length,tls,syn=rows[0]
    assert t==1600000000.5 and length==43 and tls and syn
    assert source==('192.168.20.1',12345)
    assert key==tuple(sorted((source,('1.1.1.1',443))))

def test_truncated_packet_rejected():
    with pytest.raises(ValueError,match='Truncated packet body'):
        list(packets(io.BytesIO(fixture(1)[:-1])))

def test_unsupported_format_rejected():
    with pytest.raises(ValueError,match='Unsupported PCAP format'):
        list(packets(io.BytesIO(b'bad')))

@pytest.mark.parametrize('payload,expected',[
    (b'\x16\x03\x03\x00\x01\x01','visible_handshake'),
    (b'\x17\x03\x03\x00\x01\xff','outer_application_data'),
    (b'\x16\x03\x03\x00\x01\x01\x17\x03\x03\x00\x01\xff','handshake_and_outer23'),
])
def test_observable_record_types(payload,expected):
    row=next(packets(io.BytesIO(fixture(1,payload=payload)),include_tls_metadata=True))
    assert row[-1]==expected

@pytest.mark.parametrize('t,expected',[(0,[]),(10,[0]),(20,[0,1]),(22,[0,1]),(30,[0]),(102,[])])
def test_overlapping_flow_windows(t,expected):
    windows=[(10,100,0),(20,21,1)]
    index=([10,20],[101,101])
    assert [v[2] for v in matching_windows(windows,index,t)]==expected

def test_complete_records_preserved_before_truncated_tail():
    data=fixture(1)
    errors=[]
    rows=list(capture_packets(io.BytesIO(data+data[24:-1]),errors))
    assert len(rows)==1 and errors==['Truncated packet body']

def test_invalid_format_not_suppressed():
    with pytest.raises(ValueError,match='Unsupported PCAP format'):
        list(capture_packets(io.BytesIO(b'bad'),[]))

def test_clock_bucket_preserves_previous_second_boundary():
    assert calibration_bucket('flow',100.01)==calibration_bucket('flow',100.04)
    assert calibration_bucket('flow',100.06)==calibration_bucket('flow',100.9)
    assert calibration_bucket('flow',100.04)!=calibration_bucket('flow',100.06)

def test_deduplicating_clock_buckets_preserves_all_start_matches():
    times=[100.9,100.06,100.01,100.04,101.7,101.01,102.8]
    starts=[99,100,101,102]
    expected={s for t in times for s in starts if 0<=t-s<1.05}
    seen=set();found=set()
    for t in times:
        b=calibration_bucket('flow',t)
        if b in seen: continue
        seen.add(b)
        found.update(s for s in starts if 0<=t-s<1.05)
    assert found==expected
