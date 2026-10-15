"""Publisher object-fold holdout plus source/TEST near exclusion, without inference."""
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import io
from pathlib import Path
import pickle
import pickletools
import sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.fetch_ksdd1_masks import DATA
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.prepare_magnetic_tiles import Bands
from scripts.study_equipment_expansion import read
from safelog_ai.belt_normal_roi import fingerprints,distance

REPORT=ROOT/'reports/industrial-ksdd1-v18-data-audit.json'


def container_split(raw):
    allowed={'MARK','LIST','PUT','STRING','APPEND','GET','TUPLE','STOP'}
    assert all(op.name in allowed for op,arg,pos in pickletools.genops(raw)),'Unexpected publisher split opcode'
    class ContainersOnly(pickle.Unpickler):
        def find_class(self,module,name):raise ValueError('No class loading from dataset')
        def persistent_load(self,pid):raise ValueError('No external references from dataset')
    value=ContainersOnly(io.BytesIO(raw)).load()
    assert isinstance(value,tuple) and len(value)==3
    train,test,all_names=value
    assert len(train)==len(test)==3
    assert all(isinstance(g,list) and all(isinstance(n,str) and n.startswith('kos') and n[3:].isdigit() for n in g) for g in [*train,*test,all_names])
    assert len(set(all_names))==len(all_names)==50
    assert all(set(a).isdisjoint(b) and set(a)|set(b)==set(all_names) for a,b in zip(train,test))
    assert sorted(n for fold in test for n in fold)==sorted(all_names)
    return train,test,all_names


def prepare_row(path,receipt):
    rel=path.relative_to(DATA).as_posix();mask=path.with_name(path.stem+'_label.bmp');mrel=mask.relative_to(DATA).as_posix()
    assert sha(path)==receipt['files'][rel] and sha(mask)==receipt['files'][mrel]
    with Image.open(path) as im:rgb=im.convert('RGB')
    with Image.open(mask) as im:
        assert im.size==rgb.size
        pixels=int((np.array(im.convert('L'))>0).sum())
        scale=min(512/rgb.width,1408/rgb.height);view_pixels=int((np.array(im.convert('L').resize((round(rgb.width*scale),round(rgb.height*scale)),Image.Resampling.NEAREST))>0).sum())
    return {'id':'kolektor_sdd1/'+path.parent.name+'/'+path.name,'source':'kolektor_sdd1','category':'ksdd1_commutator_surface',
            'image':path.relative_to(ROOT).as_posix(),'image_sha256':receipt['files'][rel],'mask':mask.relative_to(ROOT).as_posix(),'mask_sha256':receipt['files'][mrel],
            'size':list(rgb.size),'rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),'orientation_phashes':fingerprints(rgb),
            'publisher_object_id':path.parent.name,'publisher_original_archive_unsplit':True,'label':int(pixels>0),'defect':'surface_crack' if pixels else 'no_visible_surface_defect',
            'mask_pixels':pixels,'positive_pixels_at_native512x1408':view_pixels,'annotation_consistent':bool(pixels)==bool(view_pixels),
            'whole_photo_scope':'single surface view of publisher electrical commutator; not whole-equipment health',
            'mechanical_or_electrical_functional_health':'not_assessed','normal_label_means':'no visible annotated defect in this view; source objects themselves may be defective'}


def main():
    assert not REPORT.exists() and not (DATA/'manifest.json').exists()
    receipt=read(DATA/'download.json');train_folds,test_folds,objects=container_split((DATA/'publisher-splits/split.pyb').read_bytes())
    paths=sorted((DATA/'originals').rglob('*.jpg'));assert len(paths)==399
    with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(lambda p:prepare_row(p,receipt),paths))
    assert sum(r['label'] for r in rows)==52 and set(r['publisher_object_id'] for r in rows)==set(objects)
    names=['data/equipment-expansion-v1/manifest.json','data/industrial-ksdd-supervised-v10/manifest.json','data/industrial-bsclass-v1/manifest.json','data/industrial-magnetic-tiles-v16/manifest.json']
    heldout={}
    for name in names:
        for r in read(ROOT/name)['rows']:
            if r['phase'] in ('test','reserved_test'):heldout[r['image']]=dict(r)
    def heldout_fingerprint(r):
        assert sha(ROOT/r['image'])==r['image_sha256']
        if not r.get('orientation_phashes') or not r.get('rgb_sha256'):
            with Image.open(ROOT/r['image']) as im:
                rgb=im.convert('RGB');r['orientation_phashes']=fingerprints(rgb);r['rgb_sha256']=hashlib.sha256(rgb.tobytes()).hexdigest()
        return r
    with ThreadPoolExecutor(max_workers=4) as pool:old=list(pool.map(heldout_fingerprint,heldout.values()))
    old_index=Bands();old_rgb=defaultdict(set)
    for i,r in enumerate(old):old_index.add(r['orientation_phashes'],i);old_rgb[r['rgb_sha256']].add(i)
    parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):
        a,b=find(i),find(j);parent[max(a,b)]=min(a,b)
    index=Bands();seen_rgb={};seen_object={};near_links=[];old_matches={}
    for i,r in enumerate(rows):
        obj=r['publisher_object_id']
        if obj in seen_object:union(i,seen_object[obj])
        else:seen_object[obj]=i
        if r['rgb_sha256'] in seen_rgb:union(i,seen_rgb[r['rgb_sha256']])
        else:seen_rgb[r['rgb_sha256']]=i
        for j in sorted(index.candidates(r['orientation_phashes'])):
            if distance(r['orientation_phashes'],rows[j]['orientation_phashes'])<=2:
                union(i,j);near_links.append({'a':r['id'],'b':rows[j]['id']})
        index.add(r['orientation_phashes'],i)
        match=set(old_rgb.get(r['rgb_sha256'],()))
        for j in old_index.candidates(r['orientation_phashes']):
            if distance(r['orientation_phashes'],old[j]['orientation_phashes'])<=2:match.add(j)
        if match:old_matches[i]=[old[j]['image'] for j in sorted(match)]
    groups=defaultdict(list)
    for i in range(len(rows)):groups[find(i)].append(i)
    author_test=set(test_folds[0]);excluded=[]
    for members in groups.values():
        key=min(rows[i]['id'] for i in members)
        has_test=any(rows[i]['publisher_object_id'] in author_test for i in members)
        bad=any(i in old_matches or not rows[i]['annotation_consistent'] for i in members)
        rank=int(hashlib.sha256(('ksdd1-v18:'+key).encode()).hexdigest()[:16],16)/2**64
        phase='fit' if rank<.70 else 'validation' if rank<.85 else 'cal'
        for i in members:
            r=rows[i];is_test=r['publisher_object_id'] in author_test
            r['appearance_object_group']=key
            r['phase']='excluded_existing_TEST_or_mask_group' if bad else 'test' if is_test else 'excluded_author_TEST_near_group' if has_test else phase
            r['publisher_fold0_role']='TEST' if is_test else 'TRAIN'
            if r['phase'].startswith('excluded'):excluded.append(r['id'])
    phase_sets={p:{r['appearance_object_group'] for r in rows if r['phase']==p} for p in ('fit','validation','cal','test')}
    assert all(phase_sets[a].isdisjoint(phase_sets[b]) for i,a in enumerate(phase_sets) for b in list(phase_sets)[i+1:])
    counts={p:dict(Counter('defect' if r['label'] else 'normal' for r in rows if r['phase']==p)) for p in sorted({r['phase'] for r in rows})}
    minimums={'fit':5,'validation':2,'cal':2,'test':5}
    ready=all(min(counts.get(p,{}).get('normal',0),counts.get(p,{}).get('defect',0))>=n for p,n in minimums.items())
    audit={'UTC':datetime.now(timezone.utc).isoformat(),'status':'ready_for_research_protocol' if ready else 'insufficient_grouped_source_do_not_train',
           'source_download_receipt_sha256':sha(DATA/'download.json'),'source_images':399,'annotated_surface_defect_views':52,'no_visible_defect_views':347,
           'publisher_object_folders':50,'object_identity_basis':'author paper: eight nonoverlapping surface views per electrical commutator; one missing view in archive',
           'cross_source_physical_identity_verified':False,'official_fold0_TEST_objects':sorted(author_test),'original_author_3fold_TEST_sizes':[len(f) for f in test_folds],
           'FIT_VAL_CAL_rule':'remaining publisher fold0 TRAIN, connected object/exactRGB/any8orientationpHash2 groups; SHA fixed70/15/15, no score-based rearrangement',
           'source_groups':len(groups),'counts':counts,'within_source_near_links':near_links,'existing_TEST_photos_checked':len(old),
           'existing_TEST_matches':{rows[i]['id']:m for i,m in old_matches.items()},'excluded_views':excluded,
           'existing_TEST_manifest_sha256':{n:sha(ROOT/n) for n in names},'normal_views_do_not_mean_whole_object_is_healthy':True,
           'native_view_preparation_check':'width512 height1408 aspect preserving, grey127 letterbox, nearest real mask>0; no defective mask vanishes',
           'license':'CC BY-NC-SA 4.0 per author page, noncommercial academic research, attribution/share-alike',
           'citation':'Tabernik et al., DOI10.1007/s10845-019-01476-x',
           'split_decoded':'strict container-only opcode allowlist with all globals/persistent references rejected; no archive code executed',
           'source_image_content_viewed_by_agent':False,'new_model_scores_or_optimizer_updates':0,'public_benchmark_not_independent_field_validation':True,
           'photo20percent_PASS_requires_each_TEST_label_at_least20':True,'current_eligible_for_app_category':False}
    new(DATA/'manifest.json',{'schema':'safelog-ksdd1-object-groups-v18','rows':rows,'audit':audit});new(REPORT,audit)
    print({'status':audit['status'],'counts':counts,'object_groups':len(groups),'old_TEST_matches':len(old_matches),'old_TEST_photos_checked':len(old),'publisher_fold0_TEST_objects':len(author_test)},flush=True)


if __name__=='__main__':main()
