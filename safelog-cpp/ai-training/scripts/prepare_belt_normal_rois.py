"""Read two exports; prepare v2 normal cores and conservative provisional splits."""
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path,PurePosixPath
import shutil
import sys
import zipfile
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from safelog_ai.belt_normal_roi import LABELS,parent,provisional_split,normal_core,fingerprints,distance,grouped
from scripts.audit_belt_normal_coco import checked_bbox
from scripts.unified_main_data import read,sha,DATA
DATA_OUT=ROOT/'data/belt-normal-rois-v1'; RUN=ROOT/'runs/belt-normal-rois-v1'


def new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f: json.dump(value,f,ensure_ascii=False,indent=2)


def load(archive):
    rows=[]
    with zipfile.ZipFile(archive) as packed:
        paths=[i.filename for i in packed.infolist()]
        assert len(paths)==len(set(paths)) and sum(i.file_size for i in packed.infolist())<2_000_000_000
        assert all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts and ':' not in n and '\\' not in n for n in paths)
        assert packed.testzip() is None
        license_text=packed.read('README.dataset.txt').decode('utf-8'); assert 'License: CC BY 4.0' in license_text
        for path in sorted(n for n in paths if n.endswith('_annotations.coco.json')):
            split=path.split('/')[0]; assert split in ('train','valid','test')
            annotation=json.loads(packed.read(path)); categories={c['id']:c['name'].strip().lower() for c in annotation['categories']}
            assert set(categories.values())<=LABELS and 'normal belt' in categories.values()
            by=defaultdict(list)
            for a in annotation['annotations']: by[a['image_id']].append(a)
            ids={im['id'] for im in annotation['images']}; assert len(ids)==len(annotation['images']) and set(by)<=ids
            for im in annotation['images']:
                name=split+'/'+im['file_name']; raw=packed.read(name)
                with Image.open(io.BytesIO(raw)) as src:
                    assert src.getexif().get(274) in (None,1)
                    image=src.convert('RGB'); assert image.size==(im['width'],im['height'])
                    pixels=np.asarray(image); rgbsha=hashlib.sha256(str(pixels.shape).encode()+pixels.tobytes()).hexdigest()
                    phash=fingerprints(image)
                original=parent(im.get('extra',{}).get('name',im['file_name']))
                boxes=[(categories[a['category_id']],checked_bbox(a['bbox'],im['width'],im['height'])) for a in by[im['id']]]
                cores=[normal_core(box,boxes,im['width'],im['height']) for label,box in boxes if label=='normal belt']
                cores=[c for c in cores if c is not None]
                rows.append({'id':name,'parent':original,'publisher_split':split,'split':provisional_split(original,split),
                    'bytes_sha256':hashlib.sha256(raw).hexdigest(),'rgb_sha256':rgbsha,'phash':phash,'boxes':boxes,
                    'core':max(cores,key=lambda c:(c[2]-c[0],-c[0],-c[1])) if cores else None,
                    'whole_photo_normal_ground_truth':'unknown','mechanical_health_ground_truth':'unknown'})
    return rows


def main():
    assert not (DATA_OUT/'manifest.json').exists(), 'Preserve completed preparation'
    DATA_OUT.mkdir(exist_ok=True); RUN.mkdir(exist_ok=True)
    v1=ROOT/'data/belt-normal-roboflow-v1/source.coco.zip'
    v2source=Path('C:/Users/c/Downloads/belt defects.v2i.coco.zip'); v2=DATA_OUT/'source.coco.zip'
    if v2.exists(): assert sha(v2)==sha(v2source), 'Preserve earlier downloaded archive'
    else: shutil.copy2(v2source,v2)
    rows=load(v2); prior=load(v1); assert len(rows)==1548 and len(prior)==497
    print(json.dumps({'decoded_images':len(rows),'v2_label_count':6,'v1_label_count':7}),flush=True)
    by_parent=defaultdict(set)
    for r in rows: by_parent[r['parent']].add(r['publisher_split'])
    groups=grouped(rows); selections=defaultdict(list)
    print(json.dumps({'similarity_groups':len(set(groups))}),flush=True)
    for group,row in zip(groups,rows): row['similarity_group']=group; selections[group].append(row)
    # A similarity group that spans proposed split ranges is not kept in any split.
    eligible=[]; exclusions=Counter()
    worn_parents={r['parent'] for r in prior if any(label=='worn surface' for label,box in r['boxes'])}
    for group,items in selections.items():
        if any(r['parent'] in worn_parents for r in items): exclusions['v1_worn_label_parent_excluded']+=1; continue
        ranges={r['split'] for r in items if r['split'] in ('fit','cal','reserved')}
        if len(ranges)>1: exclusions['cross_provisional_split_similarity_groups']+=1; continue
        candidates=[r for r in items if r['split'] in ('fit','cal','reserved') and r['core'] is not None]
        if not candidates: continue
        chosen=min(candidates,key=lambda r:(-(r['core'][2]-r['core'][0]),r['id']))
        eligible.append(chosen)
    # Compare all full-frame hashes to earlier belt TRAIN/knownVAL before deciding fit.
    old=read(DATA/'manifest.json'); reference=old['fit']['belt_damage']+old['belt_known_validation']
    def hash_old(row):
        with Image.open(ROOT/row['image']) as image:
            w,h=row['width'],row['height']; rw,rh=round(640*w/max(w,h)),round(640*h/max(w,h)); l,t=(640-rw)//2,(640-rh)//2
            hashes=fingerprints(image.crop((l,t,l+rw,t+rh)))
        return row['id'],row['split'],hashes
    with ThreadPoolExecutor(max_workers=4) as pool: old_hashes=list(pool.map(hash_old,reference))
    prepared=[]; conflicts=[]
    with zipfile.ZipFile(v2) as packed:
        for row in sorted(eligible,key=lambda r:(r['split'],r['parent'])):
            matches=[{'id':name,'split':split} for name,split,h in old_hashes if distance(row['phash'],h)<=6]
            # Old TRAIN matches can supplement FIT labels, but never CAL/reserved.
            if any(m['split']=='val' for m in matches) or (matches and row['split']!='fit'):
                conflicts.append({'id':row['id'],'matches':matches}); continue
            with Image.open(io.BytesIO(packed.read(row['id']))) as image:
                core=image.convert('RGB').crop(tuple(row['core'])); original=core.copy(); crop=core.resize((640,640),Image.Resampling.BILINEAR)
            index=len(prepared); path=DATA_OUT/'prepared'/f'{index:04d}.png'; path.parent.mkdir(exist_ok=True); crop.save(path)
            source_roi=DATA_OUT/'native-cores'/f'{index:04d}.png'; source_roi.parent.mkdir(exist_ok=True); original.save(source_roi)
            item={**row,'image':path.relative_to(ROOT).as_posix(),'image_sha256':sha(path),
                'source_roi':source_roi.relative_to(ROOT).as_posix(),'source_roi_sha256':sha(source_roi),
                'label':0,'label_scope':'publisher_normal_bbox_interior_ROI_only','pixel_mask_ground_truth':'unknown',
                'source_collection_history_verified':False,'earlier_TRAIN_similarity_matches':matches}
            prepared.append(item)
    counts=Counter(r['split'] for r in prepared)
    manifest={'status':'prepared_provisional_normal_ROIs_not_whole_photos','source':'https://universe.roboflow.com/gofaones-workspace/belt-defects/dataset/2',
        'publisher':'Gofaones Workspace','license':'CC BY 4.0','archive_sha256':sha(v2),'v1_archive_sha256':sha(v1),
        'exported_v2_images':len(rows),'source_parent_hints':len(by_parent),'publisher_cross_split_parent_hints':sum(len(v)>1 for v in by_parent.values()),
        'dedup_similarity_groups':len(selections),'eligible_explicit_normal_groups_before_prior_source_check':len(eligible),
        'excluded_prior_source_overlap_groups':len(conflicts),'exclusions':dict(exclusions),'split_counts':dict(counts),'rows':prepared,
        'sampling':'One explicit normal core per conservative parent/exactRGB/rotation-mirror pHash<=6 component; core60% of shortest bbox side, min64, excludes ANY overlap with nonnormal annotations',
        'provisional_split':'FIT publisherTRAIN frame0..149; CAL frame180..239; reserved frame270+; remaining gaps/unknown excluded. Frame hints are NOT verified original recording IDs.',
        'normal_whole_photo_labels_created':0,'normal_pixel_masks_created':0,'normal_roi_labels_from_publisher':len(prepared),
        'new_independent_factory_photos_verified':0,'original_sequence_groups_verified':False,'reserved_model_predictions_performed':False,
        'old_BeltCrack_reserved_TEST_used':False,'new_source_annotations_not_ai_generated':True}
    manifest['preparation_source_sha256']={n:sha(ROOT/n) for n in ('safelog_ai/belt_normal_roi.py','scripts/prepare_belt_normal_rois.py','tests/test_belt_normal_roi.py')}
    new(DATA_OUT/'manifest.json',manifest); new(RUN/'source-audit.json',{'v2_rows':rows,'prior_source_overlap':conflicts})
    print(json.dumps({k:v for k,v in manifest.items() if k!='rows'}),flush=True)


if __name__=='__main__': main()
