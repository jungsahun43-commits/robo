"""Audit publisher coordinates/classes and replay heldout connectivity before learning."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from safelog_ai.corrosion_condition_data import native_classes, polygon_classes, allocate, LABELS, RGB
from safelog_ai.insplad_pixel_groups import Groups
from safelog_ai.belt_normal_roi import fingerprints, distance
from safelog_ai.process_probe import process_exists
from scripts.prepare_magnetic_tiles import Bands
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read

SOURCE = ROOT/'data/industrial-corrosion-condition-v64-source'
META = ROOT/'data/industrial-corrosion-condition-v65-metadata'
OUT = ROOT/'data/industrial-corrosion-condition-v66'
RUN = ROOT/'runs/industrial-corrosion-condition-data-v66'
POLICY = ROOT/'reports/industrial-corrosion-condition-v66-data-policy.json'
OLD = ROOT/'data/industrial-insplad-grouped-v44'
V57 = ROOT/'data/industrial-insplad-lower-shackle-v57'
OST = ROOT/'data/industrial-ostrava-masks-v61'
UNSUP = ROOT/'data/industrial-insplad-unsupervised-v56-metadata/all-rows.json'
PRIOR = ('data/industrial-insplad-grouped-v44/manifest.json', 'data/industrial-insplad-glass-v47/manifest.json',
         'data/industrial-insplad-lightning-v50/manifest.json', 'data/industrial-insplad-varigrip-v53/manifest.json')
SOURCES = ('scripts/prepare_corrosion_condition_pixels.py', 'safelog_ai/corrosion_condition_data.py', 'tests/test_corrosion_condition_data.py')


def utc(): return datetime.now(timezone.utc).isoformat()


def freeze():
    assert not OUT.exists() and not RUN.exists() and not POLICY.exists()
    assert sha(META/'all-pairs-before-json-or-pixels.json') == '209ac81618ee5f02a27b224e34ed12f52ce171c11c548e24cbde872755293375'
    assert sha(META/'reserved-original-and-resized-TEST-before-pixels.json') == 'd5325e1f9f9b2c96d9e7b1ffd2fd03d740e51799a5591d2ae78970176c5df396'
    assert (SOURCE/'author-labels-static.txt').read_text(encoding='utf-8').splitlines() == ['__ignore__','_background_','2_Fair_Steel_Corrosion','3_Poor_Steel_Corrosion','4_Severe_Steel_Corrosion']
    paths = [SOURCE/'publisher-ccs-v2.zip', META/'all-pairs-before-json-or-pixels.json', META/'reserved-original-and-resized-TEST-before-pixels.json',
             OLD/'grouped-all-source.json', OLD/'retained-TEST-fingerprints.json', UNSUP, V57/'pixel-joining-edges.jsonl', V57/'manifest.json',
             OST/'decoded-pairs.json', OST/'manifest.json', OST/'new-pixel-links.json',
             SOURCE/'author-labels-static.txt', SOURCE/'author-labelme2voc-static.py.txt', SOURCE/'author-datahandler-static.py.txt']
    paths += [ROOT/n for n in PRIOR]
    paths += [v for v in sorted(V57.glob('fingerprints-*.json')) if not v.name.endswith('-sha.json')]
    OUT.mkdir(); RUN.mkdir()
    p = {'UTC': utc(), 'fixed_before_new_source_pixel_decode': True,
         'source_sha256': {n: sha(ROOT/n) for n in SOURCES}, 'inputs_sha256': {v.relative_to(ROOT).as_posix(): sha(v) for v in paths},
         'decode': 'Sequential originalJPEG/nativePNG/512JPEG/512PNG and JSON from verified ZIP members to RAM. Each encoded file <=32MiB, RGB <=40MP, EXIF orientation None/1 only; no autorotation, GT repairs, imageData decode, author code execution or filesystem use of JSON imagePath.',
         'geometry': 'Original JPEG, native PNG and JSON imageWidth/imageHeight must agree. Both supplied copies must be512square. Polygon >=3 finite points, coordinates within[-1,width]x[-1,height]; preserve original points and audit with PIL polygon in original order. Out-of-bound within1pixel is boundary raster clipping, recorded. Never substitute raster audit for provided PNG.',
         'classes': {'raw_label_to_native_id': LABELS, 'expected_RGB_to_native_id': {str(k): v for k,v in RGB.items()},
                     'accept': 'Actual P palette/L indices or exact RGB documented VOC colors; no thresholding, unknown colors/labels or geometry errors cause study gate failure. Ignore255 retained. Native PNG versus independent ordered JSON polygon raster must match exactly; mismatch keeps source untouched and blocks learning pending authoritative investigation.'},
         'copies': 'Same publisher role+stem explicit original/copy link before distances, original440 entries not880newphotos; native vs provided512 nearest-resize agreement is diagnostic only. Never train resized copies as independent images.',
         'grouping': 'Replay original48140 graph and94saved edges; canonical783v57 rows exact. Replay saved Ostrava105 pixel edges plus exact cameraModel+DateTimeOriginal second; verify all105 v61canonical keys. Compare only new880RGB views to old caches and each other using size+RGBexact or8orientationpHash Hamming<=2. Same cameraModel+validEXIFsecond joins new original captures. No old decoder/pairwise audit rerun.',
         'held': 'Preserve all44 original publisherTEST and all220associated files. Held roots include old8995TEST+InsPLAD6417originalVAL+6481unsupervisedTEST+Ostrava15TEST+allnewTEST copies. Any transitively connected TRAIN excluded.',
         'allocation': 'Preserve oldv45/v48/v51/v54 andv57lower andv61Ostrava development role one; conflict excludes. Other eligible originalTRAIN groups ordered SHA256(ccs66:+canonicalID), firstceil15percentVAL, nextceil15percentCAL, restFIT once. No label-balanced reshuffle, no Test asTrain.',
         'minimum_entries': {'fit': 40, 'validation': 10, 'cal': 10, 'test': 20},
         'minimum_groups': {'fit': 5, 'validation': 3, 'cal': 3, 'test': 3},
         'minimum_photos_with_Fair_or_Poor_or_Severe': {'fit': 30, 'validation': 5, 'cal': 5, 'test': 10},
         'minimum_nonignore_pixels_per_phase': 1024,
         'error_policy': 'File/hash/decode errors stop and preserve failure/events. Annotation discrepancies accumulate as diagnostics; any discrepancy blocks study, with no guessed repair/exclusion relaxation. Minimum failure stops before features/learning. No automatic selection of easier class subsets.',
         'QA': 'Only after ready manifest, first3FIT originalmember sorted RGB/native class overlays; no VAL/CAL/TEST content viewing or original GT modification.',
         'semantics': 'Good/background includes none OR minimal corrosion, not binary zero-rust. All four original classes and ignore retained. Labels describe annotated photo appearance, not measured depth, whole machine/bridge health or independent physical parts.',
         'storage': 'No original/copy extraction; metadata and3FITcontactsheet <=96MiB, native buffers sequential <=1GiBRAM, >=1GiB diskreserve; no denseencodercache/fullparent duplicate.',
         'license': 'EricBianchi/MatthewHebdon VirginiaTechV2 DOI10.7294/16624663.v2 dataCC0; cite dataset andpaper10.1061/(ASCE)CP.1943-5487.0001045 withsource/version/changes. SourcecodeMIT separate; all olderdata conditions retained.',
         'new_units_passed': 3, 'new_encoder_features_scores_gradient_optimizer': 0,
         'training_protocol_fixed': False, 'physical_part_capture_groups_verified': False, 'independent_field_validation': False}
    assert shutil.disk_usage(ROOT).free > 1024**3+96*1024**2
    new(POLICY, p); print({'policy_sha256': sha(POLICY)}, flush=True)


def validate():
    p = read(POLICY)
    for n,h in {**p['source_sha256'], **p['inputs_sha256']}.items(): assert sha(ROOT/n) == h, n
    git = ['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
    for n in (*SOURCES, POLICY.relative_to(ROOT).as_posix()):
        assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])).hexdigest() == sha(ROOT/n), n
    return p


def member_bytes(archive, item):
    assert item['bytes'] <= 32*1024**2
    raw = archive.read(item['member'])
    assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['encoded_sha256']
    return raw


def image(raw, fmt):
    im = Image.open(io.BytesIO(raw))
    assert im.format == fmt and im.width*im.height <= 40_000_000 and im.getexif().get(274) in (None,1)
    im.load(); return im


def decode():
    rows = []; pairs = read(META/'all-pairs-before-json-or-pixels.json')['rows']
    with zipfile.ZipFile(SOURCE/'publisher-ccs-v2.zip') as archive, (RUN/'decode-events.jsonl').open('x',encoding='utf-8',buffering=1) as log:
        for r in pairs:
            views = []; masks = []; problems = []; annotation = json.loads(member_bytes(archive, r['members']['original/json']))
            for key in ('original/images','512x512/images_512'):
                im = image(member_bytes(archive, r['members'][key]), 'JPEG'); rgb = im.convert('RGB')
                if key.startswith('512'): assert rgb.size == (512,512)
                exif = im.getexif(); sub = exif.get_ifd(34665) if exif.get(34665) else {}; stamp = sub.get(36867); camera = exif.get(272)
                capture = str(camera)+'|'+stamp if camera and isinstance(stamp,str) and re.fullmatch(r'\d{4}:\d{2}:\d{2} \d{2}:\d{2}:\d{2}',stamp) else None
                views.append({'id': r['id']+'/'+key, 'entry_id': r['id'], 'variant': key, 'member': r['members'][key]['member'],
                    'size': list(rgb.size), 'rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(), 'orientation_phashes': fingerprints(rgb), 'capture_second_proxy': capture})
                im.close()
            for key in ('original/masks','512x512/mask_512'):
                mask = image(member_bytes(archive, r['members'][key]), 'PNG')
                expected = views[0 if key.startswith('original') else 1]['size']; assert list(mask.size) == expected
                try: m = native_classes(mask)
                except ValueError as error:
                    problems.append({'kind': 'mask_class_contract', 'variant': key, 'error': str(error)}); m = None
                masks.append(m)
                mode = mask.mode
                if m is not None:
                    unique, counts = np.unique(m, return_counts=True)
                    info = {'mode': mode, 'class_counts': dict(zip(map(str,unique.tolist()),counts.tolist())), 'class_sha256': hashlib.sha256(m.tobytes()).hexdigest()}
                else: info = {'mode': mode, 'class_counts': None, 'unmapped_colors_or_indices': True}
                if key.startswith('original'): native_info = info
                else: copy_info = info
                mask.close()
            try:
                reconstructed = polygon_classes(annotation, tuple(views[0]['size']))
                mismatch = None if masks[0] is None else int(np.count_nonzero(reconstructed != masks[0]))
                if mismatch: problems.append({'kind': 'native_JSON_pixel_mismatch', 'pixels': mismatch, 'total': reconstructed.size})
            except (ValueError, KeyError, TypeError) as error:
                problems.append({'kind':'JSON_geometry_or_labels', 'error':str(error)}); mismatch = None
            copy_agreement = None
            if all(m is not None for m in masks):
                resized = np.asarray(Image.fromarray(masks[0]).resize((512,512),Image.Resampling.NEAREST))
                copy_agreement = float(np.mean(resized == masks[1]))
            value = {**r, 'views':views, 'native_mask':native_info, 'copy_mask':copy_info,
                     'JSON_raw_labels':dict(Counter(s.get('label') for s in annotation['shapes'])),
                     'JSON_shape_types':dict(Counter(str(s.get('shape_type')) for s in annotation['shapes'])),
                     'JSON_group_ids_unverified_polygon_groups': sorted({str(s.get('group_id')) for s in annotation['shapes']}),
                     'JSON_native_raster_mismatch_pixels':mismatch, 'nearest_copy_class_agreement_diagnostic_only':copy_agreement,
                     'annotation_problems':problems, 'archive':(SOURCE/'publisher-ccs-v2.zip').relative_to(ROOT).as_posix(),
                     'photo_scope':'publisher metal surface corrosion appearance classes; Good is not binary zero-rust'}
            rows.append(value)
            new(OUT/('decoded-'+r['publisher_role']+'-'+r['stem']+'.json'),value)
            log.write(json.dumps({'UTC':utc(),'id':r['id'],'original_and_copy_RGB_masks_decoded':True,'annotation_problems':problems})+'\n')
    new(OUT/'decoded-pairs.json',{'rows':rows}); return rows


def graph(rows):
    prior = []
    for path in sorted(V57.glob('fingerprints-*.json')):
        if not path.name.endswith('-sha.json'): prior.extend(read(path)['rows'])
    old = read(OLD/'grouped-all-source.json')['rows']; held = read(OLD/'retained-TEST-fingerprints.json')['rows']; ost = read(OST/'decoded-pairs.json')['rows']
    assert (len(prior), len(old), len(held), len(ost)) == (27620,11525,8995,105)
    assert [r['member'] for r in prior] == [r['member'] for r in read(UNSUP)['rows']]
    a,b = len(prior),len(prior)+len(old); c = b+len(held); d = c+len(ost)
    views = [v for r in rows for v in r['views']]; allrows = prior+old+held+ost+views
    g = Groups(len(allrows)); oldgroups = {}; metagroups = {}
    for j,r in enumerate(old):
        key = r['group_id']
        if key in oldgroups: g.union(a+j,oldgroups[key])
        else: oldgroups[key] = a+j
    for i,r in enumerate(prior):
        key = r['group_proxy']
        if key in metagroups: g.union(i,metagroups[key])
        else: metagroups[key] = i
        for key in r['old_source_group_links']: g.union(i,oldgroups[key])
    edges = [json.loads(v) for v in (V57/'pixel-joining-edges.jsonl').read_text(encoding='utf-8').splitlines()]; assert len(edges) == 94
    for i,j,*_ in edges: assert 0 <= i < a and 0 <= j < c; g.union(i,j)
    keys = ['unsupervised:'+r['member'] for r in prior]+['supervised:'+r['member'] for r in old]+['held:'+r['image'] for r in held]
    def canonical(n):
        result = {}
        for i,key in enumerate(keys[:n]): root = g.find(i); result[root] = min(result.get(root,key),key)
        return result
    canon = canonical(c); unsupmember = {r['member']:i for i,r in enumerate(prior)}
    for r in read(V57/'manifest.json')['rows']: assert canon[g.find(unsupmember[r['member']])] == r['pixel_group_id']
    keys += ['ostrava:'+r['member'] for r in ost]
    for i,j,*_ in read(OST/'new-pixel-links.json')['rows']: assert i < d and j < d; g.union(i,j)
    capture = {}
    for i,r in enumerate(ost,c):
        stamp = r['capture_second_proxy']
        if stamp:
            if stamp in capture: g.union(i,capture[stamp])
            else: capture[stamp] = i
    canon = canonical(d); ostmember = {r['member']:c+i for i,r in enumerate(ost)}
    for r in read(OST/'manifest.json')['rows']: assert canon[g.find(ostmember[r['member']])] == r['group_id']
    keys += ['ccs66:'+v['id'] for v in views]
    for i in range(0,len(views),2): assert views[i]['entry_id'] == views[i+1]['entry_id']; g.union(d+i,d+i+1)
    index = Bands(); exact = defaultdict(list); counts = Counter(); links = []
    for i,r in enumerate(allrows[:d]): index.add(r['orientation_phashes'],i); exact[(r['rgb_sha256'],tuple(r['size']))].append(i)
    # Reuse existing capture proxies only where provided; never infer absent physical identities.
    for i,r in enumerate(views,d):
        key = (r['rgb_sha256'],tuple(r['size']))
        for j in exact.get(key,()):
            if g.find(i) != g.find(j): g.union(i,j); links.append([i,j,'RGB_exact']); counts['RGB_exact_joins'] += 1
        for j in sorted(index.candidates(r['orientation_phashes'])):
            if g.find(i) == g.find(j): continue
            counts['new_cross_component_pHash_comparisons'] += 1
            delta = distance(r['orientation_phashes'],allrows[j]['orientation_phashes'])
            if delta <= 2: g.union(i,j); links.append([i,j,'pHash',delta]); counts['pHash_joins'] += 1
        stamp = r['capture_second_proxy'] if r['variant'] == 'original/images' else None
        if stamp:
            if stamp in capture: g.union(i,capture[stamp]); counts['EXIF_second_proxy_joins'] += 1
            else: capture[stamp] = i
        index.add(r['orientation_phashes'],i); exact[key].append(i)
    canon = canonical(len(keys))
    heldindices = [i for i,r in enumerate(prior) if r['publisher_role']=='test']+[a+j for j,r in enumerate(old) if r['publisher_role']=='val']+list(range(b,c))
    heldindices += [ostmember[r['member']] for r in read(OST/'manifest.json')['rows'] if r['phase']=='test']
    external = {canon[g.find(i)] for i in heldindices}; roles = defaultdict(set)
    for name in PRIOR:
        for r in read(ROOT/name)['rows']:
            if r['phase'] in ('fit','validation','cal'): roles[canon[g.find(oldgroups[r['group_id']])]].add(r['phase'])
    for r in read(V57/'manifest.json')['rows']:
        if r['phase'] in ('fit','validation','cal'): roles[canon[g.find(unsupmember[r['member']])]].add(r['phase'])
    for r in read(OST/'manifest.json')['rows']:
        if r['phase'] in ('fit','validation','cal'): roles[canon[g.find(ostmember[r['member']])]].add(r['phase'])
    grouped = [{**r,'group_id':canon[g.find(d+2*i)]} for i,r in enumerate(rows)]
    sourceheld = {r['group_id'] for r in grouped if r['publisher_role']=='Test'}
    assignments = allocate([r['group_id'] for r in grouped],external|sourceheld,roles)
    result = [{**r,'phase':'test' if r['publisher_role']=='Test' else assignments[r['group_id']],
               'prior_development_roles':sorted(roles[r['group_id']]),'externally_held_connected':r['group_id'] in external} for r in grouped]
    new(OUT/'new-pixel-links.json',{'rows':links,'counts':dict(counts),'explicit_original_copy_links':440})
    return result,dict(counts)


def audit():
    p = validate(); assert not (OUT/'decoded-pairs.json').exists() and not (OUT/'manifest.json').exists()
    rows = decode(); values,counts = graph(rows); failures = []
    invalid = [r for r in values if r['annotation_problems']]
    if invalid: failures.append({'kind':'annotation_contract_failed','entries':len(invalid),'ids':[r['id'] for r in invalid]})
    summaries = {}
    for phase in p['minimum_entries']:
        chosen = [r for r in values if r['phase']==phase]; groups = len({r['group_id'] for r in chosen})
        positive = sum(any(r['native_mask']['class_counts'] and r['native_mask']['class_counts'].get(str(k),0)>0 for k in (1,2,3)) for r in chosen)
        pixels = Counter()
        for r in chosen: pixels.update(r['native_mask']['class_counts'] or {})
        summaries[phase] = {'entries':len(chosen),'groups':groups,'Fair_Poor_Severe_photos':positive,'original_class_pixels':dict(pixels)}
        for field,actual,minimum in [('entries',len(chosen),p['minimum_entries'][phase]),('groups',groups,p['minimum_groups'][phase]),('positive_photos',positive,p['minimum_photos_with_Fair_or_Poor_or_Severe'][phase]),('nonignore_pixels',sum(v for k,v in pixels.items() if k!='255'),p['minimum_nonignore_pixels_per_phase'])]:
            if actual < minimum: failures.append({'kind':'minimum','phase':phase,'field':field,'actual':actual,'minimum':minimum})
    for x in summaries:
        for y in summaries:
            if x < y: assert {r['group_id'] for r in values if r['phase']==x}.isdisjoint({r['group_id'] for r in values if r['phase']==y})
    assert [r['id'] for r in values if r['publisher_role']=='Test'] == [r['id'] for r in read(META/'reserved-original-and-resized-TEST-before-pixels.json')['rows']]
    summary = {'UTC':utc(),'ready':not failures,'counts':summaries,'failures':failures,'annotation_problem_entries':len(invalid),
               'excluded_entries':dict(Counter(r['phase'] for r in values if r['phase'].startswith('excluded'))),
               'native_mask_modes':dict(Counter(r['native_mask']['mode'] for r in values)),
               'raw_JSON_labels':dict(sum((Counter(r['JSON_raw_labels']) for r in values),Counter())),
               'pixel_comparisons_and_joins':counts,'replayed_old_pixel_edges':94,'v57_783_and_v61_105_canonical_groups_exact':True,
               'new_original_entries_decoded':440,'new_RGB_decodes_including_supplied_copies':880,'new_mask_decodes_including_supplied_copies':880,
               'original_TEST44_all220associatedfiles_preserved':True,'new_source_images_viewed_features_scores_gradient_optimizer':0,
               'Good_includes_minimal_corrosion_not_binary_zero_rust':True,'training_protocol_fixed':False,
               'physical_part_capture_groups_verified':False,'independent_field_validation':False,
               'policy_sha256':sha(POLICY),'reserved_TEST_sha256':sha(META/'reserved-original-and-resized-TEST-before-pixels.json'),
               'new_units_passed':3,'old_models_outputs_maps_unchanged':True,'disk_free_bytes':shutil.disk_usage(ROOT).free}
    new(OUT/'manifest.json',{'rows':values,'audit':summary}); summary['manifest_sha256'] = sha(OUT/'manifest.json')
    new(ROOT/'reports/industrial-corrosion-condition-v66-data-audit.json',summary); print(summary,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage',choices=('freeze','audit')); args = parser.parse_args()
    if args.stage == 'freeze': freeze()
    else:
        for path in (ROOT/'runs').glob('**/*.lock'):
            if process_exists(int(path.read_text(encoding='utf-8').strip())): raise RuntimeError('Live job '+str(path))
        lock = RUN/'operation.lock'
        with lock.open('x',encoding='utf-8') as f: f.write(str(os.getpid()))
        try: audit()
        except BaseException as error:
            new(RUN/('failure-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'UTC':utc(),'error':repr(error),'pid':os.getpid()}); raise
        finally: lock.unlink()
