"""Previously unscored frames from known TEST recordings, selected before calibration."""
import hashlib
import io
import zipfile
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np
from PIL import Image,ImageOps

from prepare_conveyor_expansion import SOURCE, DATA as PRIOR, OLD, read, new, sha, evenly
from prepare_conveyor_foreign import canonical_group

DATA = SOURCE/'calibration-check-v3'


def choose(inventory,prior):
    used = {r['id'] for r in prior['rows']}
    keys = defaultdict(list)
    for row in inventory:
        keys[canonical_group(row['recording_name_hint']),row['frame_number_hint']].append(row)
    buckets = defaultdict(list)
    for row in inventory:
        group = canonical_group(row['recording_name_hint']); label = row['foreign_object_photo_label']
        if prior['group_splits'][group]!='test' or row['name'] in used or len(keys[group,row['frame_number_hint']])>1:
            continue
        if row['source_split']!=('anomalous' if label else 'test'):
            continue
        buckets[group,label].append(row)
    return [{**r,'id':r['name'],'recording_group':g,'split':'fresh_same_recordings'}
            for (g,y),items in sorted(buckets.items()) for r in evenly(items,128)]


def main():
    assert not DATA.exists()
    prior = read(PRIOR/'manifest.json')
    selected = choose(read(SOURCE/'curated-label-inventory.json')['rows'],prior)
    new(DATA/'selection.json',{'rows':selected,'selection_before_calibration_fit_or_new_model_scores':True,
        'rule':'128 evenly spaced unused frames per original TEST recording/class',
        'prior_manifest_sha256':sha(PRIOR/'manifest.json'),'same_known_test_recordings':True,
        'new_independent_factory_or_recording':False})
    (DATA/'originals').mkdir();(DATA/'prepared').mkdir()
    known = {r['decoded_rgb_sha256'] for r in prior['rows']}
    rows,excluded = [],[]
    with zipfile.ZipFile(SOURCE/'image-data.zip') as archive:
        for i,row in enumerate(selected):
            payload = archive.read(row['name'])
            with Image.open(io.BytesIO(payload)) as image:
                rgb = ImageOps.exif_transpose(image).convert('RGB');array = np.array(rgb)
                digest = hashlib.sha256(str(array.shape).encode()+array.tobytes()).hexdigest()
                if digest in known:
                    excluded.append(row['id']);continue
                known.add(digest)
                scale = 384/max(rgb.size);w,h = [max(1,round(v*scale)) for v in rgb.size]
                prepared = Image.new('RGB',(384,384),(123,116,103))
                prepared.paste(rgb.resize((w,h),Image.Resampling.BILINEAR),((384-w)//2,(384-h)//2))
            native,png = DATA/f'originals/{i}.jpg',DATA/f'prepared/{i}.png'
            native.write_bytes(payload);prepared.save(png)
            rows.append({**row,'decoded_rgb_sha256':digest,'native':native.relative_to(DATA).as_posix(),
                'image':png.relative_to(DATA).as_posix(),'native_sha256':sha(native),'image_sha256':sha(png)})
            if i%200==0:
                print({'prepared':i,'selected':len(selected)},flush=True)
    new(DATA/'manifest.json',{'status':'frozen','rows':rows,'counts':dict(Counter(str(r['foreign_object_photo_label']) for r in rows)),
        'exact_rgb_exclusions':excluded,'prior_manifest_sha256':sha(PRIOR/'manifest.json'),
        'group_splits':prior['group_splits'],'new_independent_factory_or_recording':False,
        'previously_scored_selected_photo_ids':0,'new_ai_or_human_labels':0,'source_sha256':sha(Path(__file__))})
    print({'prepared':len(rows),'counts':read(DATA/'manifest.json')['counts'],'exclusions':len(excluded)},flush=True)


if __name__=='__main__':
    main()
