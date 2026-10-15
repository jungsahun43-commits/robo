"""Fixed grouping/splits for author images and masks before model fitting."""
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from safelog_ai.belt_normal_roi import fingerprints,distance
from scripts.fetch_equipment_expansion import sha
from scripts.study_equipment_expansion import read
from scripts.prepare_equipment_expansion import new
DATA=ROOT/'data/industrial-magnetic-tiles-v16';RUN=ROOT/'runs/industrial-magnetic-tiles-v16'
REPORT=ROOT/'reports/industrial-magnetic-tiles-v16-data-audit.json'


class Bands:
    """Three disjoint bands: Hamming<=2 must share at least one band."""
    def __init__(self):self.index=defaultdict(set)
    def keys(self,hashes):
        for value in hashes:
            n=int(value,16)
            for start,bits in ((0,22),(22,21),(43,21)):yield start,(n>>start)&((1<<bits)-1)
    def add(self,hashes,key):
        for band in self.keys(hashes):self.index[band].add(key)
    def candidates(self,hashes):
        result=set()
        for band in self.keys(hashes):result.update(self.index.get(band,()))
        return result


def source(path,receipt):
    rel=path.relative_to(DATA/'originals').as_posix();mask=path.with_suffix('.png')
    assert sha(path)==receipt['files'][rel] and sha(mask)==receipt['files'][mask.relative_to(DATA/'originals').as_posix()]
    with Image.open(path) as im:rgb=im.convert('RGB')
    with Image.open(mask) as im:
        assert im.size==rgb.size
        a=np.asarray(im.convert('L'))
    label=int(path.parts[-3]!='MT_Free');positive=int((a>=128).sum())
    scale=min(512/rgb.width,512/rgb.height)
    binary=Image.fromarray((a>=128).astype('uint8')*255)
    resized_positive=int((np.asarray(binary.resize((max(1,round(rgb.width*scale)),max(1,round(rgb.height*scale))),Image.Resampling.NEAREST))>0).sum())
    annotation_consistent=label==int(positive>0) and label==int(resized_positive>0)
    if not label:assert positive==0,rel
    found=re.fullmatch(r'(exp\d+)_num_(\d+)',path.stem);assert found,rel
    return {'id':'magnetic_tiles/'+rel,'image':path.relative_to(ROOT).as_posix(),'image_sha256':sha(path),
        'mask':mask.relative_to(ROOT).as_posix(),'mask_sha256':sha(mask),'rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),
        'orientation_phashes':fingerprints(rgb),'size':list(rgb.size),'label':label,'defect':path.parts[-3],
        'publisher_split':'unsplit','filename_block_proxy':found[1]+':'+str(int(found[2])//1000),
        'publisher_positive_pixels_at128':positive,'mask_intermediate_pixels':int(((a>0)&(a<255)).sum()),
        'annotation_consistent':annotation_consistent,
        'positive_pixels_after_view512':resized_positive,
        'mask_semantics':'author grayscale annotation binarized >=128, fixed before fitting; antialiased values are not all positive',
        'mechanical_health':'not_assessed','whole_photo_scope':'publisher_cropped_component_surface'}


def main():
    assert not (DATA/'manifest.json').exists() and not RUN.exists()
    receipt=read(DATA/'download.json');paths=sorted((DATA/'originals').rglob('*.jpg'));assert len(paths)==1344
    with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(lambda p:source(p,receipt),paths))
    # Existing TEST photos are used solely for exclusion fingerprints, never targets.
    manifests=['data/equipment-expansion-v1/manifest.json','data/industrial-ksdd-supervised-v10/manifest.json','data/industrial-bsclass-v1/manifest.json']
    old={}
    for name in manifests:
        for r in read(ROOT/name)['rows']:
            if r['phase'] in ('test','reserved_test'):old[r['image']]={**r,'audit_source':name}
    def old_fingerprint(r):
        assert sha(ROOT/r['image'])==r['image_sha256']
        with Image.open(ROOT/r['image']) as im:rgb=im.convert('RGB')
        return {**r,'rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),'orientation_phashes':fingerprints(rgb)}
    with ThreadPoolExecutor(max_workers=4) as pool:old_rows=list(pool.map(old_fingerprint,old.values()))
    old_index=Bands();old_RGB=defaultdict(set)
    for i,r in enumerate(old_rows):old_index.add(r['orientation_phashes'],i);old_RGB[r['rgb_sha256']].add(i)
    roots=list(range(len(rows)));links=[]
    def find(i):
        while roots[i]!=i:roots[i]=roots[roots[i]];i=roots[i]
        return i
    def union(a,b):
        x,y=find(a),find(b);roots[max(x,y)]=min(x,y)
    seen=Bands();seen_rgb={};blocks={};old_matches={}
    for i,r in enumerate(rows):
        if r['filename_block_proxy'] in blocks:union(i,blocks[r['filename_block_proxy']])
        else:blocks[r['filename_block_proxy']]=i
        if r['rgb_sha256'] in seen_rgb:union(i,seen_rgb[r['rgb_sha256']])
        else:seen_rgb[r['rgb_sha256']]=i
        for j in sorted(seen.candidates(r['orientation_phashes'])):
            if distance(r['orientation_phashes'],rows[j]['orientation_phashes'])<=2:
                union(i,j);links.append({'a':r['id'],'b':rows[j]['id'],'Hamming_max':2})
        seen.add(r['orientation_phashes'],i)
        matches=set(old_RGB.get(r['rgb_sha256'],()))
        for j in old_index.candidates(r['orientation_phashes']):
            if distance(r['orientation_phashes'],old_rows[j]['orientation_phashes'])<=2:matches.add(j)
        if matches:old_matches[i]=sorted(old_rows[j]['image'] for j in matches)
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[find(i)].append(i)
    # Technical source inspection already viewed the first4 masks per class and one example image.
    # Keep those source groups in FIT; no such example can become a new held-out TEST.
    inspected={p.relative_to(ROOT).as_posix() for folder in sorted((DATA/'originals').iterdir()) for p in sorted(folder.rglob('*.jpg'))[:4]}
    forced=[]
    for members in groups.values():
        key=min(rows[i]['id'] for i in members)
        if any(not rows[i]['annotation_consistent'] for i in members):phase='excluded_missing_defect_mask_group'
        elif any(i in old_matches for i in members):phase='excluded_near_existing_TEST'
        elif any(rows[i]['image'] in inspected for i in members):phase='fit';forced.append(key)
        else:
            rank=int(hashlib.sha256(('magnetic16:'+key).encode()).hexdigest(),16)%1000
            phase='fit' if rank<500 else 'validation' if rank<650 else 'cal' if rank<800 else 'test'
        for i in members:rows[i].update(group_id=key,phase=phase)
    phases={p:{r['group_id'] for r in rows if r['phase']==p} for p in ('fit','validation','cal','test')}
    assert all(not phases[a]&phases[b] for i,a in enumerate(phases) for b in list(phases)[i+1:])
    counts={p:dict(Counter('defect' if r['label'] else 'normal' for r in rows if r['phase']==p)) for p in ('fit','validation','cal','test','excluded_near_existing_TEST','excluded_missing_defect_mask_group')}
    assert min(counts['fit'].values())>=50 and min(counts['validation'].values())>=20 and counts['cal']['normal']>=20 and min(counts['test'].values())>=20
    audit={'utc':datetime.now(timezone.utc).isoformat(),'source_revision':receipt['revision'],'source_download_receipt_sha256':sha(DATA/'download.json'),
        'research_permission':receipt['research_permission'],'source_pictures':len(rows),'source_categories':dict(Counter(r['defect'] for r in rows)),
        'counts':counts,'group_count':len(groups),'group_rule':'connected exact RGB, any8orientation pHash<=2, OR same experiment filename numeric block //1000 across labels',
        'normalization_threshold_for_author_grayscale_masks':128,'source_masks_with_intermediate_values':sum(r['mask_intermediate_pixels']>0 for r in rows),
        'missing_defect_mask_source_ids':[r['id'] for r in rows if not r['annotation_consistent']],
        'empty_defect_masks_converted_to_normal':False,'initial_prepare_failure_preserved':'data/industrial-magnetic-tiles-v16/prepare.log',
        'near_links':links,'existing_TEST_photos_fingerprinted':len(old_rows),'existing_TEST_manifests_sha256':{n:sha(ROOT/n) for n in manifests},
        'old_TEST_matches':{rows[i]['id']:m for i,m in old_matches.items()},'source_inspected_examples_forced_FIT':sorted(inspected),'forced_FIT_groups':forced,
        'publisher_provided_official_split':False,'split_fixed_before_model_scores':True,'physical_acquisition_groups_verified':False,
        'filename_block_is_verified_physical_identity':False,'pHash_is_proof_of_same_physical_part':False,
        'new_TRAIN_near_old_or_new_TEST_groups':False,'new_model_features_training_CAL_TEST_scores_computed':False,
        'source_images_masks_Git_blobs_checked':True,'publisher_defect_masks_available':True,'unreviewed_v14_annotations_used':False}
    RUN.mkdir();new(DATA/'manifest.json',{'schema':'safelog-magnetic-tiles-grouped-v16','rows':rows,'audit':audit});new(REPORT,audit)
    print({'counts':counts,'groups':len(groups),'old_TEST_matches':len(old_matches),'old_TEST_fingerprinted':len(old_rows)},flush=True)


if __name__=='__main__':main()
