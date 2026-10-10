"""Conservative building-only IDEA photo supplement, without generated labels."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from xml.etree import ElementTree as ET
import cv2
import numpy as np
from PIL import Image,ImageOps

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.fetch_idea_research import BASE,LICENSE,acquire_images,metadata
from scripts.fetch_rc2119 import sha,write_new
from scripts.audit_rc2119 import SAFETY_CLASSES,read,fingerprint,existing_inventory,overlap
from scripts.screen_rc2119_crops import features,geometric_candidate,POLICY

STRUCTURES={'AE_BUILDING','GL-AEDES'}
RC_ELEMENTS={'RC column','RC beam','RC wall','RC Deck/slab','RC joint'}
POSITIVE_LIMIT=200;NEGATIVE_LIMIT=200;ACQUIRE_NEGATIVES=250
MANIFEST='data/idea-research/train.json'

def author_targets(root):
    """Negatives require the publisher's explicit whole-photo no-damage tag."""
    if root.findtext('structureType')not in STRUCTURES:raise ValueError('Outside building-only scope')
    targets=[-1]*7
    presence=root.findtext('damagePresence');objects=root.findall('object')
    if presence=='no damage':
        if objects:raise ValueError('Contradictory normal-photo annotation')
        return [0,0,-1,-1,-1,-1,-1],'author_no_damage'
    if presence!='damage':raise ValueError('Unknown publisher damage presence')
    for obj in objects:
        if obj.get('element')not in RC_ELEMENTS:continue
        name=obj.find('name')
        if name is None:raise ValueError('Missing damage name')
        if name.text=='Spalling'and name.get('damage')=='Concrete spalling':targets[1]=1
        if name.text=='Crack':targets[0]=1
    if targets[1]!=1:raise ValueError('No explicit concrete-spalling object')
    return targets,'rc_spalling'

def validate_geometry(root,size):
    if tuple(int(root.findtext('size/'+k))for k in('width','height'))!=tuple(size):raise ValueError('XML/image size differs')
    w,h=size
    for obj in root.findall('object'):
        box=obj.find('bndbox')
        if box is None:raise ValueError('Missing bounding geometry')
        values=[float(box.findtext(k))for k in('xmin','ymin','xmax','ymax')]
        if not all(math.isfinite(x)for x in values)or not(0<=values[0]<values[2]<=w and 0<=values[1]<values[3]<=h):
            raise ValueError('Invalid publisher bounding geometry')
        polygon=obj.find('polygon')
        if polygon is not None:
            points=list(polygon)
            if len(points)<6 or len(points)%2:raise ValueError('Invalid publisher polygon')
            for i,p in enumerate(points):
                x=float(p.text)
                if not math.isfinite(x)or not 0<=x<=(w if i%2==0 else h):raise ValueError('Out-of-frame publisher polygon')

def acquisition_selection(rows):
    positive=[];negative=[]
    for row in rows:
        root=ET.parse(ROOT/row['annotation']).getroot()
        try:targets,stratum=author_targets(root)
        except ValueError:continue
        expected='no_damage.zip'if stratum=='author_no_damage'else'damage.zip'
        if row['archive']!=expected or row['image_member']is None:continue
        (positive if stratum=='rc_spalling'else negative).append({**row,'targets':targets,'stratum':stratum})
    negative=sorted(negative,key=lambda r:r['image_member']);random.Random(61).shuffle(negative)
    return positive+negative[:ACQUIRE_NEGATIVES],{'eligible_rc_spalling':len(positive),'eligible_building_no_damage':len(negative)}

def acquire():
    metadata();path=BASE/'acquisition.json'
    if path.exists():raise ValueError('Preserve existing acquisition ledger')
    selected,counts=acquisition_selection(read(BASE/'annotation-index.json')['items'])
    print(json.dumps({'acquisition_selected':len(selected),**counts}),flush=True)
    rows=acquire_images(selected)
    write_new(path,{'schema':'idea_selected_acquisition_v1','selection_seed':61,'selection_counts':counts,
        'annotation_index_sha256':sha(BASE/'annotation-index.json'),'items':rows,'new_training_epochs':0})

def inspect(row):
    if sha(ROOT/row['image'])!=row['source_image_sha256']or sha(ROOT/row['annotation'])!=row['annotation_sha256']:
        raise ValueError('Acquired source bytes differ')
    with Image.open(ROOT/row['image'])as source:source.verify()
    with Image.open(ROOT/row['image'])as source:
        orientation=source.getexif().get(274,1);raw_size=source.size;image=ImageOps.exif_transpose(source).convert('RGB')
        pixels=hashlib.sha256(f'{image.width}x{image.height}:'.encode()+image.tobytes()).hexdigest();dh=fingerprint(image)
    root=ET.parse(ROOT/row['annotation']).getroot();targets,stratum=author_targets(root)
    issue=None
    try:
        if root.findtext('filename')!=row['image_member']:raise ValueError('XML filename differs from paired publisher photo')
        validate_geometry(root,raw_size)
        if orientation!=1:raise ValueError('Nontrivial EXIF needs separate registration review')
    except (ValueError,TypeError)as error:issue=str(error)
    return {**row,'targets':targets,'stratum':stratum,'source_size':list(image.size),'orientation':orientation,
        'pixel_sha256':pixels,'dhash':dh,'geometry_issue':issue,'annotation_geometry_issue':issue}

def audit():
    path=BASE/'source-audit.json'
    if path.exists():raise ValueError('Preserve existing source audit')
    rows=read(BASE/'acquisition.json')['items']
    with ThreadPoolExecutor(max_workers=4)as pool:rows=list(pool.map(inspect,rows))
    old_pixels,old_hashes,proof=existing_inventory()
    extra=read(ROOT/'data/rc2119-training/source-audit.json')['items']
    old_pixels.update(r['pixel_sha256']for r in extra)
    old_hashes=np.concatenate((old_hashes,np.asarray([r['dhash']for r in extra],np.uint64)))
    by_image={r['image']:r for r in rows}
    screened=[{**by_image[r['image']],**r,'excluded_reasons':list(r['excluded_reasons'])}for r in overlap(rows,old_pixels,old_hashes)]
    # Conservative grouping closes source overlap and label contradictions.
    conflicting={r['group']for r in screened if any(q['group']==r['group']and q['stratum']!=r['stratum']for q in screened)}
    for row in screened:
        row['group']=row['group'].replace('rc2119','idea')
        if row['geometry_issue']:row['excluded_reasons'].append('publisher_geometry_or_exif_issue')
        if row['group'].replace('idea','rc2119')in conflicting:row['excluded_reasons'].append('mixed_positive_normal_near_group')
    summary={'schema':'idea_source_audit_v1','acquisition_sha256':sha(BASE/'acquisition.json'),'license':LICENSE,
        'downloaded_original_photos':len(screened),'selection_counts':read(BASE/'acquisition.json')['selection_counts'],
        'photo_groups':len({r['group']for r in screened}),'whole_photo_excluded':sum(bool(r['excluded_reasons'])for r in screened),
        'eligible_counts':dict(Counter(r['stratum']for r in screened if not r['excluded_reasons'])),
        'full_archive_md5_verified':False,'zip_members_crc_verified':True,'old_inventory':proof,
        'additional_rc2119_photos_screened':len(extra),'heldout_labels_or_predictions_used':False,
        'scope':'Selected building RC spalling positives; explicit whole-photo no-damage building negatives. Normal material unspecified, other5 targets unknown.',
        'scene_independence_verified':False,'new_training_epochs':0}
    write_new(path,{'summary':summary,'items':screened});write_new(ROOT/'reports/facility-idea-source-audit.json',summary)
    print(json.dumps(summary),flush=True)

def old_paths():
    cache=read(ROOT/'data/convid-training/existing-fingerprints.json')
    old=list(cache['image_sizes'])+[f"data/convid-training/originals/{r['id']}.jpg"for r in read(ROOT/'data/convid-training/source-audit.json')['items']]
    old +=[r['image']for r in read(ROOT/'data/rc2119-training/source-audit.json')['items']]
    return old

def cache():
    """Source-image-only feature preparation; never read heldout target labels."""
    cv2.setNumThreads(1);cv2.setRNGSeed(61);old=old_paths()
    path=BASE/'existing-sift.npz';record=BASE/'existing-sift.json'
    if path.exists()or record.exists():raise ValueError('Preserve existing feature cache')
    points=[];descriptions=[];lengths=[]
    with ThreadPoolExecutor(max_workers=4)as pool:
        for i,(xy,desc)in enumerate(pool.map(features,(ROOT/p for p in old)),1):
            points.append(xy);descriptions.append(desc);lengths.append(len(desc))
            if i%2000==0:print(f'Existing source SIFT {i}/{len(old)}',flush=True)
    matrix=np.concatenate(descriptions);offsets=np.r_[0,np.cumsum(lengths)];del descriptions
    np.savez(path,descriptors=matrix,points=np.concatenate(points),offsets=offsets)
    write_new(record,{'policy':POLICY,'opencv_version':cv2.__version__,'images':old,'source_byte_sizes':[(ROOT/p).stat().st_size for p in old],
        'cache_sha256':sha(path),'descriptors':len(matrix),'heldout_labels_or_predictions_used':False})
    print('Existing-image SIFT cache prepared',flush=True)

def screen():
    final=BASE/'crop-review.json'
    if final.exists():raise ValueError('Preserve crop screen')
    cv2.setNumThreads(1);cv2.setRNGSeed(61)
    old=old_paths();record=read(BASE/'existing-sift.json')
    if record['policy']!=POLICY or record['images']!=old or record['source_byte_sizes']!=[(ROOT/p).stat().st_size for p in old]or record['cache_sha256']!=sha(BASE/'existing-sift.npz'):
        raise ValueError('Existing-image-only SIFT cache differs')
    with np.load(BASE/'existing-sift.npz',allow_pickle=False)as fields:
        matrix=fields['descriptors'];all_points=fields['points'];offsets=fields['offsets']
    points=[all_points[offsets[i]:offsets[i+1]]for i in range(len(old))]
    rows=read(BASE/'source-audit.json')['items'];new=[r['image']for r in rows]
    with ThreadPoolExecutor(max_workers=4)as pool:new_features=list(pool.map(features,(ROOT/p for p in new)))
    excluded=set();reviews=[]
    # Cross-source local matching excludes the query itself from the index.
    # Within-source grouping uses dHash only, without a scene-independence claim.
    old_matrix=matrix
    matcher_old=cv2.FlannBasedMatcher(dict(algorithm=1,trees=4),dict(checks=64));matcher_old.add([old_matrix]);matcher_old.train()
    for i,(xy,desc)in enumerate(new_features):
        votes=defaultdict(list)
        if len(desc):
            for pair in matcher_old.knnMatch(desc,k=2):
                if len(pair)!=2 or pair[0].distance>=POLICY['descriptor_ratio_max']*pair[1].distance:continue
                m=pair[0];owner=int(np.searchsorted(offsets,m.trainIdx,side='right')-1)
                votes[owner].append((m.queryIdx,int(m.trainIdx-offsets[owner])))
        candidates=[]
        for owner,pairs in votes.items():
            proof=geometric_candidate(xy,points[owner],pairs)
            if proof:candidates.append({'existing_image':old[owner],**proof})
        if candidates:excluded.add(rows[i]['group'])
        reviews.append({'image':new[i],'existing_shared_view_candidates':candidates})
        if(i+1)%50==0:print(f'IDEA local-view screen {i+1}/{len(rows)}',flush=True)
    blocked={r['image']for r in rows if r['group']in excluded}
    public={'schema':'idea_crop_screen_v1','policy':POLICY,'old_images':len(old),'old_descriptors':len(old_matrix),
        'new_photos':len(rows),'excluded_images':len(blocked),'source_audit_sha256':sha(BASE/'source-audit.json'),
        'within_source_local_crop_pairs_screened':False,'within_source_dhash_groups_screened':True,
        'scene_independence_verified':False,'heldout_labels_or_predictions_used':False,'new_training_epochs':0}
    write_new(final,{'summary':public,'excluded_images':sorted(blocked),'items':reviews})
    write_new(ROOT/'reports/facility-idea-crop-audit.json',public)

def prepare():
    path=ROOT/MANIFEST
    if path.exists():raise ValueError('Preserve train preparation')
    audit=read(BASE/'source-audit.json');crop=read(BASE/'crop-review.json');excluded=set(crop['excluded_images'])
    if crop['summary']['source_audit_sha256']!=sha(BASE/'source-audit.json'):raise ValueError('Audit binding differs')
    groups=defaultdict(list)
    for row in audit['items']:
        if not row['excluded_reasons']and row['image']not in excluded:groups[row['group']].append(row)
    selected=[]
    for stratum,limit in(('rc_spalling',POSITIVE_LIMIT),('author_no_damage',NEGATIVE_LIMIT)):
        candidates=[min(values,key=lambda r:r['image'])for values in groups.values()if values[0]['stratum']==stratum]
        candidates.sort(key=lambda r:r['image']);random.Random(61).shuffle(candidates);selected+=candidates[:limit]
    if not any(r['stratum']=='rc_spalling'for r in selected)or not any(r['stratum']=='author_no_damage'for r in selected):
        raise ValueError('Both screened positive and normal groups required')
    items=[]
    for i,row in enumerate(selected):
        with Image.open(ROOT/row['image'])as source:image=ImageOps.exif_transpose(source).convert('RGB');image.thumbnail((1280,1280),Image.Resampling.LANCZOS)
        dest=BASE/'prepared'/f'idea-{i:04d}.jpg';dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():raise ValueError('Preserve prepared images')
        image.save(dest,quality=95)
        pixel=BASE/'prepared'/f'idea-{i:04d}.npz'
        np.savez_compressed(pixel,mask=np.zeros((7,80,80),np.uint8),known=np.zeros(7,np.uint8))
        items.append({'image':dest.relative_to(ROOT).as_posix(),'targets':row['targets'],'domain':'idea',
            'pixel_target':pixel.relative_to(ROOT).as_posix(),'image_sha256':sha(dest),'pixel_target_sha256':sha(pixel),
            'source_image':row['image'],'source_annotation':row['annotation'],'source_image_sha256':row['source_image_sha256'],
            'source_annotation_sha256':row['annotation_sha256'],'group_id':row['group'],'stratum':row['stratum'],
            'source_size':row['source_size'],'scene_unknown':True,'source_split':'train_only_unsplit'})
    public={'schema':'idea_research_photo_preparation_v1','license':LICENSE,'local_noncommercial_research_only':True,
        'photos':len(items),'strata':dict(Counter(r['stratum']for r in items)),
        'known_target_counts':{c:{'positive':sum(r['targets'][k]==1 for r in items),'negative':sum(r['targets'][k]==0 for r in items),
            'unknown':sum(r['targets'][k]==-1 for r in items)}for k,c in enumerate(SAFETY_CLASSES)},
        'source_url':'https://zenodo.org/records/15120522','author':'EUCENTRE / Senaldi, Casarotti, Mandirola, Cantoni',
        'source_audit_sha256':sha(BASE/'source-audit.json'),'crop_review_sha256':sha(BASE/'crop-review.json'),
        'new_spatial_or_auxiliary_labels':0,'new_ai_labels':0,'new_human_labels':0,'industrial_factory_ground_truth':False,
        'unknown_policy':'Explicit no-damage whole-photo primary negatives only. Missing damage annotations unknown. Other5/pixel/aux unknown.',
        'distribution_policy':'Do not distribute source images/adapted images or this research checkpoint. Reports and software only.',
        'limitations':['Normal-photo material is not established.','Buildings include seismic/ordinary inspections, not verified factory scenes.',
            'Publisher no-damage ontology has not been independently relabelled against SafeLog visible-surface definitions.',
            'No original scene IDs; near-photo grouping and local-view screening cannot prove independence.',
            'No new independently heldout evaluation; original public VAL repeatedly explored.']}
    write_new(path,{'split':'train','classes':list(SAFETY_CLASSES),'items':items,'audit':public})
    write_new(ROOT/'reports/facility-idea-data.json',{**public,'manifest_sha256':sha(path)});print(json.dumps(public),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['acquire','audit','cache','screen','prepare']);args=parser.parse_args();globals()[args.mode]()
