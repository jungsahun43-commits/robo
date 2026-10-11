"""Build a separate foreign-object task with whole-recording-name group splits."""
import hashlib
import io
import json
import re
import zipfile
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np
from PIL import Image,ImageOps

from fetch_conveyor_foreign import DATA as SOURCE, ROOT, EXPECTED_SHA, sha

DATA = SOURCE/'grouped-photo-v1'
SIZE = 384
PREVIEW_GROUP = 'boardcamera_iron_ore_jaspilito'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as handle:
        json.dump(value,handle,indent=2)


def canonical_group(name):
    # Explicit filename alias, not proof of a different physical machine.
    return name[:-len('_fastandfurious')] if name.endswith('_fastandfurious') else name


def group_split(rows):
    classes = defaultdict(set)
    for row in rows:
        classes[canonical_group(row['recording_name_hint'])].add(row['foreign_object_photo_label'])
    key = lambda name:hashlib.sha256(('conveyor-foreign-groups-v1|'+name).encode()).hexdigest()
    positive = sorted((g for g,v in classes.items() if 1 in v and g!=PREVIEW_GROUP),key=key)
    negative = sorted((g for g,v in classes.items() if 1 not in v),key=key)
    assert len(positive)==12 and len(negative)==6 and PREVIEW_GROUP in classes
    val,test = positive[:2]+negative[:1],positive[2:5]+negative[1:2]
    train = positive[5:]+negative[2:]+[PREVIEW_GROUP]
    return {g:split for split,groups in [('train',train),('val',val),('test',test)] for g in groups}


def choose(rows):
    splits = group_split(rows)
    frames = defaultdict(list)
    for row in rows:
        frames[canonical_group(row['recording_name_hint']),row['frame_number_hint']].append(row)
    repeated_keys = {key for key,items in frames.items() if len(items)>1}
    candidates = defaultdict(list)
    for row in rows:
        group = canonical_group(row['recording_name_hint'])
        if (group,row['frame_number_hint']) in repeated_keys:
            continue
        split = splits[group]
        wanted_source = 'anomalous' if row['foreign_object_photo_label'] else ('test' if split=='test' else 'train')
        if row['source_split']!=wanted_source:
            continue
        candidates[split,group,row['foreign_object_photo_label']].append(row)
    chosen = []
    for (split,group,label),items in sorted(candidates.items()):
        ordered = sorted(items,key=lambda row:row['frame_number_hint'])
        indices = np.unique(np.linspace(0,len(ordered)-1,min(64,len(ordered)),dtype=int))
        chosen.extend({**ordered[i],'split':split,'recording_group':group} for i in indices)
    return chosen,splits,{'repeated_group_frame_keys_excluded':len(repeated_keys),
                         'rows_under_repeated_keys_excluded':sum(len(frames[k]) for k in repeated_keys)}


def prepare():
    assert not DATA.exists(),'Preserve completed experiment inputs'
    inventory = read(SOURCE/'curated-label-inventory.json')['rows']
    source = read(SOURCE/'archive-index.json')
    assert source['publisher_sha256_verified'] and source['archive_sha256']==EXPECTED_SHA
    selected,splits,key_audit = choose(inventory)
    raw_videos = {Path(r['name']).stem for r in source['members'] if r['name'].startswith('original-raw-videos/')}
    assert set(splits)<=raw_videos
    new(DATA/'selection.json',{'selected':selected,'group_splits':splits,'key_audit':key_audit,
        'selection':'64 numerically evenly spaced frames per group/class; recording-stratified SHA order; before scores',
        'preview_group_forced_to_training':PREVIEW_GROUP,'source_test_never_selected_for_train_or_val':True,
        'source_archive_sha256':EXPECTED_SHA,'source_inventory_sha256':sha(SOURCE/'curated-label-inventory.json'),
        'group_alias':'Only suffix_fastandfurious removed; actual physical independence not verified'})
    original,prepared = DATA/'originals',DATA/'prepared'
    original.mkdir();prepared.mkdir()
    rows = []
    with zipfile.ZipFile(SOURCE/'image-data.zip') as archive:
        for i,entry in enumerate(selected):
            payload = archive.read(entry['name'])  # zipfile verifies selected memberCRC.
            with Image.open(io.BytesIO(payload)) as image:
                rgb = ImageOps.exif_transpose(image).convert('RGB')
                width,height = rgb.size
                pixels = np.asarray(rgb)
                pixel_sha = hashlib.sha256(str(pixels.shape).encode()+pixels.tobytes()).hexdigest()
                scale = SIZE/max(width,height)
                w,h = max(1,round(width*scale)),max(1,round(height*scale))
                left,top = (SIZE-w)//2,(SIZE-h)//2
                padded = Image.new('RGB',(SIZE,SIZE),(123,116,103))
                padded.paste(rgb.resize((w,h),Image.Resampling.BILINEAR),(left,top))
            native = original/f'{i}.jpg'
            native.write_bytes(payload)
            image_path = prepared/f'{i}.png'
            padded.save(image_path)
            rows.append({**entry,'id':entry['name'],'width':width,'height':height,
                'decoded_rgb_sha256':pixel_sha,'native':native.relative_to(DATA).as_posix(),
                'native_sha256':hashlib.sha256(payload).hexdigest(),'image':image_path.relative_to(DATA).as_posix(),
                'image_sha256':sha(image_path)})
            if i%100==0:
                print(json.dumps({'decoded_and_crc_checked':i,'selected':len(selected)}),flush=True)
    buckets = defaultdict(list)
    for row in rows:
        buckets[row['decoded_rgb_sha256']].append(row)
    exclusions,kept = [],[]
    for items in buckets.values():
        if len({r['split'] for r in items})>1 or len({r['foreign_object_photo_label'] for r in items})>1:
            exclusions.extend({'id':r['id'],'reason':'Cross-split or conflicting-label identicalRGB'} for r in items)
        else:
            ordered = sorted(items,key=lambda r:r['id'])
            kept.append(ordered[0])
            exclusions.extend({'id':r['id'],'reason':'Same-label same-split identicalRGB duplicate'} for r in ordered[1:])
    kept.sort(key=lambda r:(r['split'],r['recording_group'],r['foreign_object_photo_label'],r['frame_number_hint']))
    counts = Counter((r['split'],r['foreign_object_photo_label']) for r in kept)
    for split in ('train','val','test'):
        assert counts[split,0]>0 and counts[split,1]>0
    assert not any(r['source_split']=='test' for r in kept if r['split']!='test')
    new(DATA/'manifest.json',{'status':'frozen','class':'conveyor_foreign_object','input_size':SIZE,'rows':kept,
        'counts':{s+'/'+str(label):n for (s,label),n in counts.items()},'group_splits':splits,
        'selected_before_rgb_audit':len(selected),'rgb_exclusions':exclusions,'key_audit':key_audit,
        'remaining_exact_rgb_duplicates':0,'normal_definition':'No foreign objects in iron-ore flow, not healthy belt',
        'label_source':'Publisher curated normal-op/anomaly photo folders; no new AI or human labels',
        'label_geometry':'Whole-photo presence only, no defect bbox or segmentation truth',
        'source_test_decoded_for_data_quality_audit':True,'source_test_used_for_fit_or_model_selection':False,
        'source_test_photo_count_selected':sum(r['source_split']=='test' for r in kept),
        'source_video_name_groups_disjoint':True,'independent_factory_or_physical_objects_verified':False,
        'near_frame_similarity_not_removed_within_group':True,'selection_sha256':sha(DATA/'selection.json'),
        'source_archive_sha256':EXPECTED_SHA,'preparation_script_sha256':sha(Path(__file__)),
        'native_dimensions_by_label':{str(label):dict(Counter(str((r['width'],r['height'])) for r in kept
            if r['foreign_object_photo_label']==label)) for label in (0,1)}})
    print(json.dumps({'prepared':len(kept),'counts':{s+'/'+str(label):n for (s,label),n in counts.items()},
        'rgb_exclusions':len(exclusions),'groups':dict(Counter(splits.values()))}),flush=True)


if __name__=='__main__':
    prepare()
