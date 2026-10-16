"""Replay the already completed v57/v61/v66/v68 graph, with no RGB or distance audit."""
import json
from collections import defaultdict
from .insplad_pixel_groups import Groups


def replay(root):
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    olddir=root/'data/industrial-insplad-grouped-v44';v57=root/'data/industrial-insplad-lower-shackle-v57';ostdir=root/'data/industrial-ostrava-masks-v61';ccsdir=root/'data/industrial-corrosion-condition-v66';embdir=root/'data/industrial-corrosion-json-pairs-v68'
    unsup=[]
    for path in sorted(v57.glob('fingerprints-*.json')):
        if not path.name.endswith('-sha.json'):unsup.extend(read(path)['rows'])
    old=read(olddir/'grouped-all-source.json')['rows'];held=read(olddir/'retained-TEST-fingerprints.json')['rows'];ost=read(ostdir/'decoded-pairs.json')['rows'];ccs=read(ccsdir/'manifest.json')['rows'];emb=read(embdir/'manifest.json')['rows'];views=[v for r in ccs for v in r['views']]
    assert tuple(map(len,(unsup,old,held,ost,views,emb)))==(27620,11525,8995,105,882,440)
    a,b=27620,39145;c,d,e=48140,48245,49127
    rows=unsup+old+held+ost+views+emb
    keys=['unsupervised:'+r['member'] for r in unsup]+['supervised:'+r['member'] for r in old]+['held:'+r['image'] for r in held]+['ostrava:'+r['member'] for r in ost]+['ccs66:'+r['id'] for r in views]+['ccs68:'+r['id']+'/embedded' for r in emb]
    g=Groups(len(rows));oldgroups={};meta={};capture={};roles=[]
    for j,r in enumerate(old):
        if r['group_id'] in oldgroups:g.union(a+j,oldgroups[r['group_id']])
        else:oldgroups[r['group_id']]=a+j
    for i,r in enumerate(unsup):
        if r['group_proxy'] in meta:g.union(i,meta[r['group_proxy']])
        else:meta[r['group_proxy']]=i
        for key in r['old_source_group_links']:g.union(i,oldgroups[key])
    edges=[json.loads(v) for v in (v57/'pixel-joining-edges.jsonl').read_text(encoding='utf-8').splitlines()];assert len(edges)==94
    for i,j,*_ in edges:g.union(i,j)
    def canon():
        out={}
        for i,key in enumerate(keys):r=g.find(i);out[r]=min(out.get(r,key),key)
        return out
    um={r['member']:i for i,r in enumerate(unsup)};om={r['member']:c+i for i,r in enumerate(ost)}
    for i,j,*_ in read(ostdir/'new-pixel-links.json')['rows']:g.union(i,j)
    for i,r in enumerate(ost,c):
        stamp=r['capture_second_proxy']
        if stamp:
            if stamp in capture:g.union(i,capture[stamp])
            else:capture[stamp]=i
    entry={}
    for i,r in enumerate(views,d):
        if r['entry_id'] in entry:g.union(i,entry[r['entry_id']])
        else:entry[r['entry_id']]=i
    for i,j,*_ in read(ccsdir/'new-pixel-links.json')['rows']:g.union(i,j)
    for i,r in enumerate(views,d):
        stamp=r['capture_second_proxy'] if r['variant']=='original/images' else None
        if stamp:
            if stamp in capture:g.union(i,capture[stamp])
            else:capture[stamp]=i
    for i,r in enumerate(emb,e):g.union(i,entry[r['id']])
    for i,j,*_ in read(embdir/'new-pixel-links.json')['rows']:g.union(i,j)
    for i,r in enumerate(emb,e):
        stamp=r['capture_second_proxy']
        if stamp:
            if stamp in capture:g.union(i,capture[stamp])
            else:capture[stamp]=i
    canonical=canon()
    for directory,mapping,key,groupkey in ((v57,um,'member','pixel_group_id'),(ostdir,om,'member','group_id')):
        for r in read(directory/'manifest.json')['rows']:assert canonical[g.find(mapping[r[key]])]==r[groupkey]
    for r in ccs:assert canonical[g.find(entry[r['id']])]==r['group_id']
    for i,r in enumerate(emb,e):assert canonical[g.find(i)]==r['group_id']
    held_indices=[i for i,r in enumerate(unsup) if r['publisher_role']=='test']+[a+j for j,r in enumerate(old) if r['publisher_role']=='val']+list(range(b,c))
    held_indices += [om[r['member']] for r in read(ostdir/'manifest.json')['rows'] if r['phase']=='test']+[entry[r['id']] for r in ccs if r['publisher_role']=='Test']+[e+i for i,r in enumerate(emb) if r['phase']=='test']
    prior=('data/industrial-insplad-grouped-v44/manifest.json','data/industrial-insplad-glass-v47/manifest.json','data/industrial-insplad-lightning-v50/manifest.json','data/industrial-insplad-varigrip-v53/manifest.json')
    for name in prior:
        for r in read(root/name)['rows']:
            if r['phase'] in ('fit','validation','cal'):roles.append((oldgroups[r['group_id']],r['phase']))
    for directory,mapping,key in ((v57,um,'member'),(ostdir,om,'member')):
        for r in read(directory/'manifest.json')['rows']:
            if r['phase'] in ('fit','validation','cal'):roles.append((mapping[r[key]],r['phase']))
    for r in ccs:
        if r['phase'] in ('fit','validation','cal'):roles.append((entry[r['id']],r['phase']))
    for i,r in enumerate(emb,e):
        if r['phase'] in ('fit','validation','cal'):roles.append((i,r['phase']))
    return rows,keys,g,held_indices,roles,capture
