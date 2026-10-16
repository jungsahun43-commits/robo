"""Preregister and verify partial positive VT plus full-mask Ostrava supervision."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from safelog_ai.corrosion_partial_data import partial_target,mask_view,pixel_counts
from safelog_ai.corrosion_json_pairs import polygon_labels
from safelog_ai.process_probe import process_exists
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read

VT=ROOT/'data/industrial-corrosion-json-pairs-v68/manifest.json'
OST=ROOT/'data/industrial-ostrava-masks-v61/manifest.json'
VT_SHA='af40f9f6a0cc91709c0006ec5c9e21c9d7dd94a93f079fe2401dc1dab56888ce'
OST_SHA='90ab369b28d8ad6955ca0f3114e04c1fd87f7fc464c66a28fbccc763407e4f8c'
OUT=ROOT/'data/industrial-corrosion-partial-v72'
RUN=ROOT/'runs/industrial-corrosion-partial-data-v72'
POLICY=ROOT/'reports/industrial-corrosion-partial-v72-data-policy.json'
SOURCES=('scripts/prepare_corrosion_partial.py','safelog_ai/corrosion_partial_data.py','tests/test_corrosion_partial_data.py')
EXPECTED={'ostrava':{'fit':60,'validation':15,'cal':15,'test':15},'vt':{'fit':276,'validation':60,'cal':60,'test':44}}


def utc():return datetime.now(timezone.utc).isoformat()


def freeze():
    assert not OUT.exists() and not RUN.exists() and not POLICY.exists()
    assert sha(VT)==VT_SHA and sha(OST)==OST_SHA and read(VT)['audit']['ready'] and read(OST)['audit']['ready']
    files=[VT,OST,ROOT/'data/industrial-corrosion-json-pairs-v68/new-pixel-links.json',ROOT/'data/industrial-corrosion-condition-v66/manifest.json',ROOT/'data/industrial-ostrava-masks-v61/decoded-pairs.json',ROOT/'data/industrial-ostrava-masks-v61/new-pixel-links.json',ROOT/'reports/industrial-corrosion-json-pair-v68-data-policy.json',ROOT/'safelog_ai/corrosion_json_pairs.py']
    # Bind the previously completed graph lineage, rather than decode/recompare it.
    for n in read(ROOT/'reports/industrial-corrosion-json-pair-v68-data-policy.json')['inputs_sha256']:
        if n.endswith(('.json','.jsonl')):files.append(ROOT/n)
    files=sorted(set(files));OUT.mkdir();RUN.mkdir()
    p={'UTC':utc(),'fixed_before_new_derived_GT':True,'source_sha256':{n:sha(ROOT/n) for n in SOURCES},'inputs_sha256':{v.relative_to(ROOT).as_posix():sha(v) for v in files},
       'projection':'VT raw1Fair/2Poor/3Severe -> explicitpositive1; raw0Good/background(none OR minimalcorrosion) and255ignore ->unknown255. Ostrava nativeL0 ->knownnegative0; L169 ->positive1 (original L>0 contract). Preserve raw states/order/coordinates/files. No nativePNG repair, invented negatives, severity transfer or synthetic GT.',
       'geometry':'Native annotation SHA verified. Aspectletterbox512 scale=min(512/w,512/h), round resizedsize, centeredfloorpadding; targetNEAREST only. Padding255 reported separately from nativeunknown. No RGB required for this GT-only stage, no new feature or source contents view.',
       'reuse':'Immutable v68/v61 rows/groups/phases plus completed graph lineage bound by SHA. No downloads/newRGB decode/pHash pairdistances/reallocation. Their allheld exclusions and role conflicts stayfixed. No new inputs or views, so no new graph nodes. OriginalTEST44/220files + Ostrava15 preserved.',
       'expected_entries':EXPECTED,'minimum_groups_each_source_phase':3,
       'minimum_positive_photos':{'ostrava':{'fit':30,'validation':10,'cal':10,'test':10},'vt':{'fit':40,'validation':10,'cal':10,'test':20}},
       'minimum_512_positive_pixels_each_source_phase':1024,'minimum_512_knownnegative_pixels_Ostrava_each_phase':1024,
       'VT_knownnegative_pixels_must_equal':0,'minimum_total_FIT_Ostrava_fullnegative_photos':2,
       'error':'Any missing/changed annotationSHA, size, rawclass, derivedGT validity, split overlap, count/group/pixel minimum failure blocks ready. Preserve failure/journal; do not move photos, weaken conditions or treat failed data as normal.',
       'storage':'Sequential annotations only to RAM, no imageData base64 decode/RGB or full archive expansion. Native onephoto masks plus arrays <512MiB, metadata only <32MiB disk, >=1GiB diskreserve. No derivedmask image files/denseencoder cache/parent copies/deletion.',
       'scope':'Existing336FIT(60Ostrava+276VT),75VAL,75CAL,59originalTEST reused, notnewnaturalphotos. VT unknown area has no measurable binaryFP; wholehealthyTEST insufficient for appwholephoto20percentpass. Broad manualpolygons not expert-verified allrustmicroboundaries; proxygroups not physicalsite independence.',
       'training':'None in v72. A separate singlebranch protocol/commit is required before features/gradient. Preservev16/approved3heads/v62 tensor-buffer-meta/maps.',
       'licenses':'VT Bianchi/HebdonV2 dataCC0, cite10.7294/16624663.v2 and10.1061/(ASCE)CP.1943-5487.0001045; Ostrava Fric ZenodoV2 dataCC BY4.0 DOI10.5281/zenodo.11235637. Attribution/source/version/derivedprojection changes and allparentterms inclInsPLADCC BY-NC3.0 retained.',
       'new_units_passed':3,'independent_field_validation':False}
    assert shutil.disk_usage(ROOT).free>1024**3+32*1024**2
    new(POLICY,p);print({'policy_sha256':sha(POLICY)},flush=True)


def validate():
    p=read(POLICY)
    for n,h in {**p['source_sha256'],**p['inputs_sha256']}.items():assert sha(ROOT/n)==h,n
    git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
    for n in (*SOURCES,POLICY.relative_to(ROOT).as_posix()):assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])).hexdigest()==sha(ROOT/n),n
    return p


def native_target(archive,r,source):
    if source=='vt':
        raw=archive.read(r['source_json_member']);assert hashlib.sha256(raw).hexdigest()==r['source_json_encoded_sha256']
        original,_=polygon_labels(json.loads(raw),tuple(r['size']))
        assert hashlib.sha256(original.tobytes()).hexdigest()==r['class_raster_sha256']
    else:
        raw=archive.read(r['mask_member']);assert hashlib.sha256(raw).hexdigest()==r['mask_encoded_sha256']
        with Image.open(BytesIO(raw)) as im:
            assert im.size==tuple(r['size']) and getattr(im,'n_frames',1)==1
            original=np.array(im.convert('L'))
        assert hashlib.sha256(original.tobytes()).hexdigest()==r['mask_L_sha256']
    return partial_target(source,original)


def audit():
    p=validate();rows=[];counts={};roles=defaultdict(set);ids=set()
    with (RUN/'annotation-events.jsonl').open('x',encoding='utf-8',buffering=1) as events:
        for source,path in (('ostrava',OST),('vt',VT)):
            prior=read(path)['rows'];archives={r['archive'] for r in prior};assert len(archives)==1
            with zipfile.ZipFile(ROOT/next(iter(archives))) as archive:
                for r in prior:
                    assert r['id'] not in ids;ids.add(r['id']);roles[r['group_id']].add(r['phase'])
                    native=native_target(archive,r,source);target,padding,g=mask_view(native)
                    value={**r,'source':source,'source_manifest_sha256':sha(path),'derived_native_sha256':hashlib.sha256(native.tobytes()).hexdigest(),
                           'derived_512_sha256':hashlib.sha256(target.tobytes()).hexdigest(),'derived_native_pixels':pixel_counts(native),
                           'derived_512_pixels':pixel_counts(target,padding),'input_geometry':g,'partial_annotation':source=='vt'}
                    rows.append(value);events.write(json.dumps({'UTC':utc(),'id':r['id'],'source':source,'annotation_only':True,'RGB_decode':False})+'\n')
            counts[source]={}
            for phase,n in EXPECTED[source].items():
                selected=[r for r in rows if r['source']==source and r['phase']==phase]
                aggregate={scale:{k:sum(r['derived_'+scale+'_pixels'][k] for r in selected) for k in ('positive','known_negative','unknown','padding')} for scale in ('native','512')}
                photos=sum(r['derived_512_pixels']['positive']>0 for r in selected);groups=len({r['group_id'] for r in selected})
                assert len(selected)==n and groups>=p['minimum_groups_each_source_phase']
                assert photos>=p['minimum_positive_photos'][source][phase] and aggregate['512']['positive']>=1024
                if source=='vt':assert aggregate['native']['known_negative']==aggregate['512']['known_negative']==0
                else:assert aggregate['512']['known_negative']>=1024 and aggregate['native']['unknown']==aggregate['512']['unknown']==0
                counts[source][phase]={'entries':n,'groups':groups,'positive_photos_512':photos,'fullnegative_photos_512':sum(r['derived_512_pixels']['positive']==0 and r['derived_512_pixels']['known_negative']>0 for r in selected),**aggregate}
    assert all(len(v)==1 for v in roles.values())
    assert counts['ostrava']['fit']['fullnegative_photos_512']>=2
    result={'UTC':utc(),'ready':True,'policy_sha256':sha(POLICY),'counts':counts,'locked_source_roles_exact':True,'new_graph_nodes_or_distances':0,'source_RGB_decodes':0,'new_encoder_features_scores_gradient_optimizer':0,'new_natural_photos':0,'wholephoto_app_gate_passed':False,'training_protocol_fixed':False,'independent_field_validation':False,'new_units_passed':3,'disk_free_bytes':shutil.disk_usage(ROOT).free}
    assert shutil.disk_usage(ROOT).free>=1024**3
    new(OUT/'manifest.json',{'rows':rows,'audit':result});new(ROOT/'reports/industrial-corrosion-partial-v72-data-audit.json',{**result,'manifest_sha256':sha(OUT/'manifest.json')})
    print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','audit'));args=parser.parse_args()
    if args.stage=='freeze':freeze()
    else:
        for path in (ROOT/'runs').glob('**/*.lock'):
            if process_exists(int(path.read_text(encoding='utf-8').strip())):raise RuntimeError('Live job '+str(path))
        with (RUN/'operation.lock').open('x',encoding='utf-8') as f:f.write(str(os.getpid()))
        try:audit()
        except BaseException as e:
            new(RUN/('failure-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'UTC':utc(),'error':repr(e)});raise
        finally:(RUN/'operation.lock').unlink()
