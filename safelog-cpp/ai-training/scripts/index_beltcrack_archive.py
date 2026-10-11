"""Read public RAR5 quick-open metadata; sparse files are NOT complete archives."""
import ctypes,hashlib,json,msvcrt,struct,zlib
from pathlib import Path
import requests

root=Path(__file__).resolve().parents[1]
base=root/'data/equipment-beltcrack/range-index'
base.mkdir(exist_ok=True)
assert not(base/"members.json").exists(), "Preserve an existing archive index"
entries=json.loads((root/'data/equipment-beltcrack/source-files.json').read_text(encoding='utf-8'))['data']

def vint(data,pos):
    value=0
    for bit in range(0,70,7):
        byte=data[pos];pos+=1;value|=(byte&127)<<bit
        if byte<128:return value,pos
    raise ValueError('Invalid vint')

def block(data,pos=0):
    start=pos;expected=struct.unpack_from('<I',data,pos)[0];pos+=4
    size,pos=vint(data,pos);end=pos+size
    assert zlib.crc32(data[start+4:end])==expected,'Header CRC differs'
    kind,pos=vint(data,pos);flags,pos=vint(data,pos)
    extra,pos=vint(data,pos)if flags&1 else(0,pos)
    packed,pos=vint(data,pos)if flags&2 else(0,pos)
    out=dict(kind=kind,flags=flags,extra=extra,packed=packed,end=end,start=start)
    if kind in(2,3):
        ff,pos=vint(data,pos);unpacked,pos=vint(data,pos);attrs,pos=vint(data,pos)
        if ff&2:pos+=4
        crc=struct.unpack_from('<I',data,pos)[0]if ff&4 else None
        if ff&4:pos+=4
        comp,pos=vint(data,pos);host,pos=vint(data,pos);n,pos=vint(data,pos)
        out.update(name=data[pos:pos+n].decode('utf-8'),unpacked=unpacked,file_flags=ff,compression=comp,host=host,crc32=crc)
    return out

def acquire(entry,start,end):
    key=f'{entry["fileName"]}-{start}-{end}.bin';dest=base/key
    if not dest.exists():
        r=requests.get('https://china.scidb.cn/download?fileId='+entry['id'],headers={'Range':f'bytes={start}-{end}'},timeout=90)
        assert r.status_code==206 and r.headers['Content-Range']==f'bytes {start}-{end}/{entry["size"]}'
        assert len(r.content)==end-start+1
        dest.write_bytes(r.content)
    data=dest.read_bytes();assert len(data)==end-start+1
    return data

def sparse(path,size):
    if path.exists():return
    with path.open('wb')as f:
        returned=ctypes.c_ulong()
        handle=ctypes.c_void_p(msvcrt.get_osfhandle(f.fileno()))
        assert ctypes.windll.kernel32.DeviceIoControl(handle,0x900c4,None,0,None,0,ctypes.byref(returned),None)
        f.truncate(size)

members=[]
for vi,entry in enumerate(entries):
    first=acquire(entry,0,65535);main=block(first,8)
    assert main['kind']==1
    pos=main['end']-main['extra'];q=None
    while pos<main['end']:
        length,pos=vint(first,pos);end=pos+length;kind,pos=vint(first,pos)
        if kind==1:
            flags,pos=vint(first,pos)
            if flags&1:q,pos=vint(first,pos)
        pos=end
    assert q and q>0
    q+=8
    tail=acquire(entry,q,entry['size']-1);head=block(tail)
    assert head['kind']==3 and head['name']=='QO'
    assert (head['compression']&0x380)==0
    data=tail[head['end']:head['end']+head['packed']]
    path=base/entry['fileName'];sparse(path,entry['size'])
    offsets={};pos=0
    with path.open('r+b')as file:
        file.seek(0);file.write(first);file.seek(q);file.write(tail)
        while pos<len(data):
            start=pos;crc=struct.unpack_from('<I',data,pos)[0];pos+=4
            size,pos=vint(data,pos);end=pos+size
            assert zlib.crc32(data[start+4:end])==crc,'Quick-open record CRC differs'
            flags,pos=vint(data,pos);back,pos=vint(data,pos);length,pos=vint(data,pos)
            cached=data[pos:pos+length];assert len(cached)==length
            target=q-back;assert 8<=target<q
            file.seek(target);file.write(cached);offsets[target]=cached
            pos=end
        # Traverse actual block boundaries, validating every header. Small uncached
        # headers are fetched from their original public byte positions.
        cursor=main['end'];seen=0
        while cursor<q:
            cached=offsets.get(cursor)
            if cached is None:
                if cursor<65000:cached=first[cursor:]
                else:cached=acquire(entry,cursor,min(cursor+2047,entry['size']-1));file.seek(cursor);file.write(cached)
            meta=block(cached);meta.update(volume=vi,offset=cursor,header_size=meta['end'])
            if meta['kind']==2:
                members.append(meta)
                seen+=1
            cursor+=meta['end']+meta['packed']
        assert cursor==q,(cursor,q)
    print(json.dumps({'volume':vi+1,'member_parts':seen,'cache_headers':len(offsets),'metadata_bytes':len(first)+len(tail)}),flush=True)
output=base/'members.json'
output.write_text(json.dumps({'members':members,'complete_archives':False,'whole_volume_md5_verified':False,
    'official_file_listing_sha256':hashlib.sha256((root/'data/equipment-beltcrack/source-files.json').read_bytes()).hexdigest()},indent=2),encoding='utf-8')
print('Quick-open member index ready',len(members),flush=True)
