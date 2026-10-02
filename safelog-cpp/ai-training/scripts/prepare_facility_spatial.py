"""Build TRAIN-only coarse pixel targets from source polygons/masks and absent labels."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_facility_target import read, save, sha, split_supplemental, dacl_items
from scripts.prepare_facility_data import DACL_MAPPING


def masks_for(item, parent, record, classes):
    target = item['targets']; known = np.array([int(t==0) for t in target],dtype=np.uint8)
    result = np.zeros((len(classes),80,80),dtype=np.uint8)
    if parent['domain']=='codebrim' or ('Damage Classification' in parent.get('source','')):
        return result,known,0  # No invented positive regions.
    with Image.open(ROOT/parent['image']) as handle: width,height=handle.size
    if parent['domain']=='dacl':
        source=ROOT/'data'/record['source']
        doc=read(source.parent.parent.parent/'annotations/train'/f'{source.stem}.json')
        masks=[Image.new('L',(width,height)) for _ in classes]
        for shape in doc['shapes']:
            label=DACL_MAPPING.get(shape['label'])
            if label is None or len(shape['points'])<3:continue
            points=[(x*width/doc['imageWidth'],y*height/doc['imageHeight']) for x,y in shape['points']]
            ImageDraw.Draw(masks[classes.index(label)]).polygon(points,fill=255)
    else:
        source=Path(parent['source']); level={'E':'Easy','M':'Medium','H':'Hard'}[source.stem[0]]
        path=ROOT/'data/damsegment/source/Damage Segmentaion'/level/'Labels/Mask'/f'{source.stem}_mask.png'
        with Image.open(path) as handle: rgb=np.asarray(handle.convert('RGB'))
        masks=[Image.fromarray(np.uint8(np.all(rgb==color,axis=-1))*255) for color in ((255,0,0),(0,0,255))]+[None]*5
    conflicts=0
    for k,mask in enumerate(masks):
        if mask is None or target[k]<0:continue
        if item.get('box_in_parent'):mask=mask.crop(tuple(item['box_in_parent']))
        # BOX retains a narrow marked pixel when downsampling; max-pooling asks
        # whether any annotated damage exists within each 8x8 input cell.
        array=np.asarray(mask.resize((640,640),Image.Resampling.BOX))>0
        coarse=array.reshape(80,8,80,8).max((1,3)).astype(np.uint8)
        if bool(coarse.any()) != bool(target[k]):
            known[k]=0; conflicts+=1;continue  # Preserve the photo label; pixel truth is unresolved.
        result[k]=coarse;known[k]=1
    return result,known,conflicts


def main():
    split=split_supplemental(); original,classes=dacl_items('train',640)
    base=original+[{**i,'domain':'damsegment'} for i in split['train']]
    code=read(ROOT/'data/codebrim-training/train.json')
    if code['split']!='train' or code['classes']!=classes:raise ValueError('Code split mismatch')
    base+=code['items']; parents={i['image']:i for i in base}
    details=read(ROOT/'data/facility-detail-training/manifest.json')
    if any(i['parent_split']!='train' or i['parent_image'] not in parents for i in details['items']):raise ValueError('Detail split leakage')
    records={r['stem']:r for r in read(ROOT/'data/dacl10k-yolo/records.json') if r['split']=='train'}
    output=ROOT/'data/facility-spatial-training';(output/'masks').mkdir(parents=True,exist_ok=True)
    items=base+details['items'];counts=np.zeros((7,2),dtype=np.int64); conflicts=0
    def task(pair):
        index,item=pair;parent=parents[item.get('parent_image',item['image'])]
        record=records.get(Path(parent['image']).stem)
        masks,known,conflict=masks_for(item,parent,record,classes)
        dest=output/'masks'/f'{index:05d}.npz';np.savez_compressed(dest,mask=masks,known=known)
        pos=(masks*known[:,None,None]).sum((1,2),dtype=np.int64)
        neg=known.astype(np.int64)*80*80-pos
        return {**item,'pixel_target':dest.relative_to(ROOT).as_posix()},np.stack((pos,neg),axis=1),conflict
    prepared=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,(item,count,conflict) in enumerate(pool.map(task,enumerate(items)),1):
            prepared.append(item);counts+=count;conflicts+=conflict
            if i%3000==0:print(f'Spatial training targets {i}/{len(items)}',flush=True)
    audit={'split':'train','image_size':640,'grid_size':80,'full_photo_or_patch_rows':len(base),'augmentation_rows':len(details['items']),
           'domains':dict(Counter(i['domain'] for i in prepared)),'per_label_pixel_cells':{label:{'positive':int(counts[k,0]),'negative':int(counts[k,1])} for k,label in enumerate(classes)},
           'photo_pixel_presence_conflicts_masked_unknown':conflicts,'split_sha256_source':sha(ROOT/'data/damsegment-training/train.json'),
           'policy':'Only DACL train/new Dam train/CODEBRIM train and their TRAIN detail children; source polygons/masks, negatives from asserted absent classes; CODEBRIM positives have no invented pixel regions; 8x8 any-positive cells; photo labels unchanged',
           'scope':'Spatial loss supervision, not full-resolution segmentation evaluation or field localization accuracy'}
    save(output/'train.json',{'split':'train','classes':classes,'items':prepared,'full_count':len(base),'audit':audit})
    save(ROOT/'reports/facility-target-spatial-data-audit.json',audit)
    print(__import__('json').dumps(audit,indent=2))


if __name__=='__main__':main()
