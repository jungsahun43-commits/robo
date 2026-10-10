"""Pinned, selective IDEA acquisition for local noncommercial research only.

No full-archive MD5 verification is claimed: fetched ZIP members are CRC checked
by zipfile, and local originals/ranges are SHA256 recorded. Source photos and
their adaptations must not be redistributed under CC BY-NC-ND 4.0.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sys
import time
import zipfile
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor
from xml.etree import ElementTree as ET
import requests

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.fetch_rc2119 import sha,write_new

BASE=ROOT/'data/idea-research'
RECORD='https://zenodo.org/api/records/15120522'
PINNED={'damage.zip':(9554121593,'md5:f8d8e6dabbb92e9c73141f416a472df7'),
        'no_damage.zip':(10634137686,'md5:3db0d572ff8915c02766b8f5dadef4bc')}
LICENSE='cc-by-nc-nd-4.0'

def request(url,**kwargs):
    for attempt in range(8):
        try:response=requests.get(url,timeout=(20,60),**kwargs)
        except requests.RequestException:
            if attempt==7:raise
            print('Transient source connection failure; retrying',flush=True);time.sleep(5);continue
        if response.status_code in (429,500,502,503,504):
            delay=min(60,max(2,int(response.headers.get('retry-after','5'))))
            response.close();print(f'Server retry after {delay}s',flush=True);time.sleep(delay);continue
        response.raise_for_status();return response
    raise RuntimeError('Public source retry budget exhausted')

def metadata():
    path=BASE/'record.json'
    if not path.exists():
        document=request(RECORD).json()
        if document['metadata']['license']['id']!=LICENSE:raise ValueError('Publisher licence changed')
        files={f['key']:(f['size'],f['checksum'])for f in document['files']}
        if files!=PINNED:raise ValueError('Publisher archive size/checksum changed')
        write_new(path,document)
    document=json.loads(path.read_text(encoding='utf-8'))
    if document['metadata']['license']['id']!=LICENSE or {f['key']:(f['size'],f['checksum'])for f in document['files']}!=PINNED:
        raise ValueError('Pinned metadata differs')
    return document

class RangeFile(io.RawIOBase):
    """Read-only ZIP file view with bounded, persistent 1 MiB range blocks."""
    def __init__(self,key):
        self.size=PINNED[key][0];self.position=0;self.block_size=1<<20
        self.url=RECORD+'/files/'+key+'/content';self.cache=BASE/'ranges'/key
        self.cache.mkdir(parents=True,exist_ok=True);self.memory={}
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        target=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        if target<0:raise ValueError('Negative seek')
        self.position=target;return target
    def read(self,size=-1):
        end=self.size if size<0 else min(self.size,self.position+size)
        chunks=[]
        while self.position<end:
            block=self.position//self.block_size;start=block*self.block_size;stop=min(start+self.block_size,self.size)
            path=self.cache/f'{start:012d}-{stop:012d}.bin'
            if block not in self.memory:
                if not path.exists():
                    response=request(self.url,headers={'Range':f'bytes={start}-{stop-1}','Accept-Encoding':'identity'})
                    if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{stop-1}/{self.size}':
                        raise ValueError('Server did not return requested exact byte range')
                    content=response.content
                    if len(content)!=stop-start:raise ValueError('Truncated range')
                    with path.open('xb')as handle:handle.write(content)
                content=path.read_bytes()
                if len(content)!=stop-start:raise ValueError('Cached range size differs')
                if len(self.memory)>=8:self.memory.pop(next(iter(self.memory)))
                self.memory[block]=content
            amount=min(end-self.position,stop-self.position)
            chunks.append(self.memory[block][self.position-start:self.position-start+amount]);self.position+=amount
        return b''.join(chunks)

def member_name(name):
    p=PurePosixPath(name)
    if p.is_absolute()or '..'in p.parts or '\\'in name or ':'in name:raise ValueError('Unsafe ZIP member')
    return p.name

def annotations():
    metadata();destination=BASE/'annotation-index.json'
    if destination.exists():raise ValueError('Preserve completed annotation index')
    rows=[];inventory={}
    for key in PINNED:
        print('Reading publisher ZIP directory: '+key,flush=True)
        with zipfile.ZipFile(RangeFile(key))as archive:
            members=archive.infolist();inventory[key]=[{'name':m.filename,'bytes':m.file_size,'crc32':m.CRC,'compressed_bytes':m.compress_size}for m in members]
            image_members={Path(m.filename).stem:m.filename for m in members if m.filename.lower().endswith(('.jpg','.jpeg','.png'))and '__MACOSX'not in m.filename}
            xmls=[m for m in members if m.filename.lower().endswith('.xml')and '__MACOSX'not in m.filename]
            for i,entry in enumerate(xmls,1):
                name=member_name(entry.filename);path=BASE/'annotations'/key/name;path.parent.mkdir(parents=True,exist_ok=True)
                content=archive.read(entry) # ZIP CRC checked, no archive extraction/path traversal.
                if path.exists():
                    if path.read_bytes()!=content:raise ValueError('Do not overwrite publisher annotation')
                else:
                    with path.open('xb')as handle:handle.write(content)
                root=ET.fromstring(content)
                rows.append({'archive':key,'annotation_member':entry.filename,'annotation':path.relative_to(ROOT).as_posix(),
                    'annotation_sha256':hashlib.sha256(content).hexdigest(),'image_member':image_members.get(Path(name).stem),
                    'document':xml_value(root)})
                if i%500==0:print(f'{key}: {i}/{len(xmls)} XML members verified',flush=True)
    write_new(BASE/'zip-directory.json',inventory)
    write_new(destination,{'schema':'idea_publisher_annotation_index_v1','record_sha256':sha(BASE/'record.json'),
        'zip_directory_sha256':sha(BASE/'zip-directory.json'),'license':LICENSE,'full_archive_md5_verified':False,
        'zip_member_crc32_verified':True,'items':rows})
    print(json.dumps({'annotations':len(rows),'first_documents':rows[:3]}),flush=True)

def xml_value(element):
    if not list(element):return (element.text or '').strip()
    result={}
    for child in element:
        value=xml_value(child)
        if child.tag in result:
            if not isinstance(result[child.tag],list):result[child.tag]=[result[child.tag]]
            result[child.tag].append(value)
        else:result[child.tag]=value
    return result

def acquire_images(rows):
    """Download only publisher image members selected by an audited policy."""
    result=[]
    for key in PINNED:
        selected=[r for r in rows if r['archive']==key]
        with zipfile.ZipFile(RangeFile(key))as archive:
            entries={row['image_member']:archive.getinfo(row['image_member'])for row in selected}
            def one(row):
                member=row['image_member']
                if member is None:raise ValueError('Missing publisher paired image')
                path=BASE/'originals'/key/member_name(member);path.parent.mkdir(parents=True,exist_ok=True)
                entry=entries[member]
                if not path.exists():
                    # One contiguous range per image avoids hundreds of tiny
                    # seek requests while retaining exact member CRC checks.
                    start=entry.header_offset;stop=min(PINNED[key][0],start+entry.compress_size+4096)
                    response=request(RECORD+'/files/'+key+'/content',headers={'Range':f'bytes={start}-{stop-1}','Accept-Encoding':'identity'})
                    if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{stop-1}/{PINNED[key][0]}':
                        raise ValueError('Image member range differs')
                    content=decode_member(response.content,entry)
                    with path.open('xb')as handle:handle.write(content)
                else:
                    if path.stat().st_size!=entry.file_size or zlib.crc32(path.read_bytes())&0xffffffff!=entry.CRC:
                        raise ValueError('Publisher cached image CRC/size changed')
                return {**row,'image':path.relative_to(ROOT).as_posix(),'source_image_sha256':sha(path),
                    'source_bytes':entry.file_size,'source_crc32':entry.CRC}
            with ThreadPoolExecutor(max_workers=4)as pool:
              for i,item in enumerate(pool.map(one,selected),1):
                result.append(item)
                if i%25==0:print(f'{key}: {i}/{len(selected)} selected originals CRC checked',flush=True)
    return result

def decode_member(payload,entry):
    if len(payload)<30 or payload[:4]!=b'PK\x03\x04':raise ValueError('Invalid local ZIP header')
    fields=struct.unpack('<4s5H3I2H',payload[:30]);flags,method=fields[2],fields[3]
    if flags&1 or method not in(0,8)or method!=entry.compress_type:raise ValueError('Unsupported ZIP member compression/encryption')
    offset=30+fields[-2]+fields[-1]
    raw=payload[offset:offset+entry.compress_size]
    if len(raw)!=entry.compress_size:raise ValueError('Truncated ZIP member range')
    content=raw if method==0 else zlib.decompress(raw,-15)
    if len(content)!=entry.file_size or zlib.crc32(content)&0xffffffff!=entry.CRC:raise ValueError('Publisher member CRC/size differs')
    return content

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['annotations']);args=parser.parse_args();annotations()
