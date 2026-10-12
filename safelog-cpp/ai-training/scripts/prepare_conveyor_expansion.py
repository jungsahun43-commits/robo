"""Expand only frozen TRAIN recordings; keep both original benchmarks byte identical."""
import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from prepare_conveyor_foreign import SOURCE, DATA as OLD, SIZE, canonical_group, read, new, sha

DATA = SOURCE/'train-expansion-v2'


def evenly(items, limit):
    ordered = sorted(items, key=lambda r:r['frame_number_hint'])
    return [ordered[i] for i in np.unique(np.linspace(0, len(ordered)-1, min(limit, len(ordered)), dtype=int))] if ordered else []


def choose(inventory, prior):
    groups = prior['group_splits']
    frames = defaultdict(list)
    for row in inventory:
        frames[canonical_group(row['recording_name_hint']), row['frame_number_hint']].append(row)
    repeated = {key for key, values in frames.items() if len(values)>1}
    original = {r['id']:r for r in prior['rows']}
    selected = {r['id']:{**r, 'from_completed_v1':True} for r in prior['rows']}
    buckets = defaultdict(list)
    for row in inventory:
        group = canonical_group(row['recording_name_hint'])
        if (group, row['frame_number_hint']) in repeated or row['source_split']=='test':
            continue
        label = row['foreign_object_photo_label']
        if row['source_split']!=('anomalous' if label else 'train'):
            continue
        if groups[group]=='train':
            buckets['train', group, label].append(row)
        elif groups[group]=='val' and row['name'] not in original:
            buckets['audit_val', group, label].append(row)
    for (split, group, label), items in sorted(buckets.items()):
        for row in evenly(items, 256 if split=='train' else 64):
            if row['name'] not in selected:
                selected[row['name']] = {**row, 'id':row['name'], 'split':split,
                    'recording_group':group, 'from_completed_v1':False}
    return sorted(selected.values(), key=lambda r:(r['split'], r['recording_group'],
                  r['foreign_object_photo_label'], r['frame_number_hint']))


def prepare():
    assert not DATA.exists(), 'Preserve finished experiment inputs'
    prior = read(OLD/'manifest.json')
    selected = choose(read(SOURCE/'curated-label-inventory.json')['rows'], prior)
    new(DATA/'selection.json', {'selected':selected, 'source_manifest_sha256':sha(OLD/'manifest.json'),
        'group_splits':prior['group_splits'], 'selection_before_new_scores':True,
        'train_rule':'Union original TRAIN and256 numerically even frames per original TRAIN group/class',
        'additional_validation_rule':'64 unsampled frames per original VAL group/class; same recordings, not new factory',
        'original_val_test_unchanged':True, 'original_val_and_test_are_reused_known_benchmarks':True,
        'source_test_used_for_fit_or_selection':False})
    (DATA/'originals').mkdir(); (DATA/'prepared').mkdir()
    rows = []
    with zipfile.ZipFile(SOURCE/'image-data.zip') as archive:
        for i, entry in enumerate(selected):
            native, image_path = DATA/f'originals/{i}.jpg', DATA/f'prepared/{i}.png'
            if entry['from_completed_v1']:
                payload = (OLD/entry['native']).read_bytes()
                prepared = (OLD/entry['image']).read_bytes()
                assert hashlib.sha256(payload).hexdigest()==entry['native_sha256']
                assert hashlib.sha256(prepared).hexdigest()==entry['image_sha256']
                native.write_bytes(payload); image_path.write_bytes(prepared)
                row = dict(entry)
            else:
                payload = archive.read(entry['name'])  # Selected member CRC verified by zipfile.
                with Image.open(io.BytesIO(payload)) as photo:
                    rgb = ImageOps.exif_transpose(photo).convert('RGB')
                    width, height = rgb.size
                    array = np.array(rgb)
                    rgb_sha = hashlib.sha256(str(array.shape).encode()+array.tobytes()).hexdigest()
                    scale = SIZE/max(rgb.size)
                    w,h = [max(1,round(v*scale)) for v in rgb.size]
                    prepared = Image.new('RGB',(SIZE,SIZE),(123,116,103))
                    prepared.paste(rgb.resize((w,h),Image.Resampling.BILINEAR),((SIZE-w)//2,(SIZE-h)//2))
                native.write_bytes(payload); prepared.save(image_path)
                row = {**entry, 'width':width, 'height':height, 'decoded_rgb_sha256':rgb_sha,
                       'native_sha256':sha(native), 'image_sha256':sha(image_path)}
            row.update(native=native.relative_to(DATA).as_posix(), image=image_path.relative_to(DATA).as_posix())
            rows.append(row)
            if i%250==0:
                print(json.dumps({'prepared':i,'selected':len(selected)}),flush=True)
    buckets = defaultdict(list)
    for row in rows:
        buckets[row['decoded_rgb_sha256']].append(row)
    kept, excluded = [], []
    for entries in buckets.values():
        original = [r for r in entries if r['from_completed_v1']]
        assert len(original)<=1, 'Completed v1 exact duplicates unexpectedly appeared'
        if len(entries)>1 and len({r['split'] for r in entries})>1:
            retained = original
        elif len({r['foreign_object_photo_label'] for r in entries})>1:
            retained = original
        else:
            retained = original or sorted(entries,key=lambda r:r['id'])[:1]
        kept.extend(retained)
        retained_ids = {r['id'] for r in retained}
        excluded.extend({'id':r['id'],'split':r['split'],'reason':'New exactRGB duplicate or conflicting-label exclusion'}
                        for r in entries if r['id'] not in retained_ids)
    kept.sort(key=lambda r:(r['split'],r['recording_group'],r['foreign_object_photo_label'],r['frame_number_hint']))
    assert {r['id'] for r in kept if r['from_completed_v1']}=={r['id'] for r in prior['rows']}
    assert all(r['source_split']!='test' for r in kept if r['split'] in ('train','val','audit_val'))
    for phase in ('val','test'):
        assert {r['id'] for r in kept if r['split']==phase}=={r['id'] for r in prior['rows'] if r['split']==phase}
    new(DATA/'manifest.json', {'status':'frozen','rows':kept,'input_size':SIZE,
        'counts':{s+'/'+str(y):n for (s,y),n in Counter((r['split'],r['foreign_object_photo_label']) for r in kept).items()},
        'group_splits':prior['group_splits'],'original_v1_input_sha256':sha(OLD/'manifest.json'),
        'selected_before_rgb_audit':len(selected),'rgb_exclusions':excluded,'remaining_exact_rgb_duplicates':0,
        'normal_definition':prior['normal_definition'],'label_source':prior['label_source'],
        'source_archive_sha256':prior['source_archive_sha256'],'source_test_used_for_fit_or_selection':False,
        'source_test_new_payloads_decoded':0,'original_val_test_byte_identical':True,
        'additional_validation_is_new_recording_or_factory':False,
        'old_val_and_test_results_already_known':True,'training_change_motivated_by_known_val_misses':True,
        'new_ai_or_human_labels':0,'preparation_source_sha256':sha(Path(__file__))})
    print(json.dumps({'prepared':len(kept),'counts':read(DATA/'manifest.json')['counts'],
                      'new_exact_rgb_exclusions':len(excluded)}),flush=True)


if __name__=='__main__':
    prepare()
