"""Audit publisher masks and keep only its TRAIN patches clear of detected overlap."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter,defaultdict
from pathlib import Path
import hashlib
import sys
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.train_facility_target import read,save,sha
from scripts.prepare_damsegment import dhash

COLOR_LABELS={7:'concrete_crack',1:'concrete_spalling',3:'rust_stain',6:'efflorescence'}


def mask_targets(path,classes):
    with Image.open(path) as handle:a=np.asarray(handle.convert('RGB'))
    bits=(a>=200).astype(np.uint8)
    codes=bits[:,:,0]+2*bits[:,:,1]+4*bits[:,:,2]
    if (codes==5).any():raise ValueError('Unexpected publisher mask color')
    targets=[-1]*len(classes);masks=np.zeros((len(classes),80,80),dtype=np.uint8);known=np.zeros(len(classes),dtype=np.uint8)
    for color,label in COLOR_LABELS.items():
        k=classes.index(label);binary=codes==color;targets[k]=int(binary.any())
        reduced=np.asarray(Image.fromarray(np.uint8(binary)*255).resize((640,640),Image.Resampling.BOX))>0
        masks[k]=reduced.reshape(80,8,80,8).max((1,3));known[k]=1
        if bool(masks[k].any())!=bool(targets[k]):known[k]=0  # Keep original photo label.
    return targets,masks,known


def fingerprint(path):
    with Image.open(path) as image:return dhash(image)


def source_rows(existing):
    items=[];unknown=0;overlap=[]
    for split,expected in (('train',563),('val',87),('test',93)):
        paths=[p for p in sorted((ROOT/'data/s2ds/source'/split).glob('*.png')) if not p.stem.endswith('_lab')]
        if len(paths)!=expected:raise ValueError('Publisher S2DS photo count changed')
        for path in paths:
            label=path.with_name(path.stem+'_lab.png')
            with Image.open(label) as handle:mask=np.asarray(handle)
            if mask.shape[:2]!=(1024,1024) or (mask.shape[-1]==4 and not (mask[:,:,3]==255).all()):
                raise ValueError('Unexpected source mask size/transparency')
            bits=(mask[:,:,:3]>=200).astype(np.uint8);codes=bits[:,:,0]+2*bits[:,:,1]+4*bits[:,:,2]
            unknown+=int((codes==5).sum())
            with Image.open(path) as handle:image=handle.convert('RGB')
            if image.size!=(1024,1024):raise ValueError('Publisher photo size changed')
            digest=hashlib.sha256(b'1024x1024:'+image.tobytes()).hexdigest()
            if digest in existing:overlap.append(path.relative_to(ROOT).as_posix())
            items.append({'image':path.relative_to(ROOT).as_posix(),'split':split,'pixel_sha256':digest,
                          'image_sha256':sha(path),'label_sha256':sha(label)})
    result={'items':items,'unknown_color_pixels':unknown,'exact_overlap':overlap}
    save(ROOT/'data/s2ds/pixel-audit.json',result)
    return result


def main():
    archive=ROOT/'data/facility-archives/s2ds.zip';download=read(ROOT/'reports/facility-target-s2ds-download.json')
    if sha(archive)!=download['sha256']:raise ValueError('Downloaded S2DS archive changed')
    classes=read(ROOT/'data/damsegment-training/train.json')['classes']
    records=read(ROOT/'data/dacl10k-yolo/records.json')
    existing={r['pixel_sha256'] for r in records}
    for s in ('train','test'):existing|={i['pixel_sha256'] for i in read(ROOT/f'data/damsegment-training/{s}.json')['items']}
    for s in ('train','val','test'):existing|={i['pixel_sha256'] for i in read(ROOT/f'data/codebrim-training/{s}.json')['items']}
    rows=source_rows(existing)
    if rows['unknown_color_pixels']:raise ValueError('Initial S2DS color audit failed')
    old=[ROOT/'data/dacl10k-yolo/images'/r['split']/(r['stem']+'.jpg') for r in records]
    for s in ('train','test'):old += [ROOT/i['image'] for i in read(ROOT/f'data/damsegment-training/{s}.json')['items']]
    for s in ('train','val','test'):old += [ROOT/i['image'] for i in read(ROOT/f'data/codebrim-training/{s}.json')['items']]
    cache=ROOT/'data/s2ds/existing-fingerprints.json'
    signatures={str(p.relative_to(ROOT)):p.stat().st_size for p in old}
    if cache.exists() and read(cache)['image_sizes']==signatures:
        old_hashes=np.array(read(cache)['dhash'],dtype=np.uint64)
    else:
        hashes=[]
        with ThreadPoolExecutor(max_workers=4) as pool:
            for index,value in enumerate(pool.map(fingerprint,old),1):
                hashes.append(value)
                if index%3000==0:print(f'S2DS existing-photo fingerprint audit {index}/{len(old)}',flush=True)
        save(cache,{'image_sizes':signatures,'dhash':hashes})
        old_hashes=np.array(hashes,dtype=np.uint64)
    items=rows['items'];paths=[ROOT/i['image'] for i in items]
    with ThreadPoolExecutor(max_workers=4) as pool:hashes=np.array(list(pool.map(fingerprint,paths)),dtype=np.uint64)
    parent=list(range(len(items)))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,h in enumerate(hashes):
        for j in np.flatnonzero(np.bitwise_count(hashes[:i]^h)<=6):parent[root(i)]=root(int(j))
    groups=defaultdict(list)
    for i in range(len(items)):groups[root(i)].append(i)
    excluded=[];kept=[];overlap_count=0;cross_count=0
    for indices in groups.values():
        overlap=any(items[i]['image'] in rows['exact_overlap'] or (np.bitwise_count(old_hashes^hashes[i])<=6).any() for i in indices)
        cross=len({items[i]['split'] for i in indices})>1
        overlap_count+=int(overlap);cross_count+=int(cross)
        if overlap or cross:
            excluded.extend({'image':items[i]['image'],'split':items[i]['split'],'existing_near_overlap':overlap,'cross_split_near_group':cross} for i in indices)
        else:kept.extend(i for i in indices if items[i]['split']=='train')
    output=ROOT/'data/s2ds-spatial-training';(output/'masks').mkdir(parents=True,exist_ok=True)
    counts=np.zeros((len(classes),2),dtype=np.int64);prepared=[]
    for index in sorted(kept):
        item=items[index];source=ROOT/item['image'];label=source.with_name(source.stem+'_lab.png')
        target,mask,known=mask_targets(label,classes);dest=output/'masks'/f'{source.stem}.npz'
        np.savez_compressed(dest,mask=mask,known=known)
        positive=(mask*known[:,None,None]).sum((1,2),dtype=np.int64)
        counts += np.stack((positive,known.astype(np.int64)*6400-positive),axis=1)
        prepared.append({'image':source.relative_to(ROOT).as_posix(),'pixel_target':dest.relative_to(ROOT).as_posix(),
                         'targets':target,'domain':'s2ds','source_split':'train','pixel_sha256':item['pixel_sha256'],
                         'image_sha256':item['image_sha256'],'label_sha256':item['label_sha256'],'pixel_target_sha256':sha(dest),
                         'group_id':'s2ds-near-group:'+str(root(index))})
    audit={'status':'prepared','source':'https://github.com/ben-z-original/s2ds','archive_sha256':download['sha256'],
           'publisher_counts':{'train':563,'val':87,'test':93},'usable_train_patches':len(prepared),
           'known_labels':list(COLOR_LABELS.values()),'unknown_labels':[c for c in classes if c not in COLOR_LABELS.values()],
           'exact_existing_overlap':len(rows['exact_overlap']),'existing_near_groups_excluded':overlap_count,'cross_split_near_groups_excluded':cross_count,
           'excluded_members':excluded,'existing_fingerprint_photos':len(old),'fingerprint_cache_sha256':sha(cache),
           'per_label_pixel_cells':{label:{'positive':int(counts[k,0]),'negative':int(counts[k,1])} for k,label in enumerate(classes)},
           'per_label_train_counts':{c:{'positive':sum(i['targets'][k]==1 for i in prepared),'negative':sum(i['targets'][k]==0 for i in prepared),'unknown':sum(i['targets'][k]<0 for i in prepared)} for k,c in enumerate(classes)},
           'palette_policy':'Publisher RGB channel threshold200: white crack, red spalling, yellow corrosion, cyan efflorescence; green vegetation, blue control point and black background have no positive mapped defect; remaining three categories unknown',
           'license':'Academic use only; do not distribute raw/derived dataset; record publisher terms and citations with models',
           'limitation':'Publisher patches lack original scene/parent IDs. Exact and whole-image dHash<=6 checks do not prove crop or site independence. Only author TRAIN is supplemental; its val/test remain outside optimizer and model selection.'}
    save(output/'train.json',{'split':'train','classes':classes,'items':prepared,'audit':audit})
    save(ROOT/'reports/facility-target-s2ds-data-audit.json',audit)
    print(__import__('json').dumps({k:v for k,v in audit.items() if k!='excluded_members'},indent=2),flush=True)


if __name__=='__main__':main()
