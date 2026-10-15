"""Reserve original/copy TEST identities, then inspect three TRAIN JSON schemas."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path,PurePosixPath
import subprocess
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read

SOURCE=ROOT/'data/industrial-corrosion-condition-v64-source'
OUT=ROOT/'data/industrial-corrosion-condition-v65-metadata'
POLICY=ROOT/'reports/industrial-corrosion-condition-v65-metadata-policy.json'
SCRIPT='scripts/prepare_corrosion_condition_metadata.py'


def utc():return datetime.now(timezone.utc).isoformat()


def freeze():
    assert not OUT.exists() and not POLICY.exists()
    audit=read(SOURCE/'archive-audit.json');assert audit['all_inner_CRC_and_encodedSHA_checked']
    groups={}
    for r in read(SOURCE/'zip-members-before-pixels.json')['rows']:
        if r['directory'] or r['member'].endswith('.pdf'):continue
        path=PurePosixPath(r['member']);assert len(path.parts)==5
        _,variant,role,kind,_=path.parts
        assert role in ('Train','Test') and variant in ('original','512x512')
        key=(role,path.stem);value=groups.setdefault(key,{'id':'ccs/'+role+'/'+path.stem,'publisher_role':role,'stem':path.stem,'members':{}})
        label=variant+'/'+kind;assert label not in value['members'];value['members'][label]=r
    expected={'original/images','original/json','original/masks','512x512/images_512','512x512/mask_512'}
    rows=sorted(groups.values(),key=lambda r:(r['publisher_role'],r['stem']))
    assert len(rows)==440 and Counter(r['publisher_role'] for r in rows)=={'Train':396,'Test':44}
    assert all(set(r['members'])==expected for r in rows)
    OUT.mkdir();new(OUT/'all-pairs-before-json-or-pixels.json',{'rows':rows,'copy_variants_not_independent_photos':True})
    new(OUT/'reserved-original-and-resized-TEST-before-pixels.json',{'rows':[r for r in rows if r['publisher_role']=='Test'],'original_photos':44,'all_associated_files':220,'no_images_or_masks_decoded_or_viewed':True})
    chosen=sorted([r for r in rows if r['publisher_role']=='Train'],key=lambda r:r['members']['original/json']['member'])[:3]
    p={'UTC':utc(),'fixed_before_source_JSON_read_or_pixels':True,'script_sha256':sha(ROOT/SCRIPT),'archive_sha256':sha(SOURCE/'publisher-ccs-v2.zip'),
       'archive_audit_sha256':sha(SOURCE/'archive-audit.json'),'allpairs_sha256':sha(OUT/'all-pairs-before-json-or-pixels.json'),
       'reserved_TEST_sha256':sha(OUT/'reserved-original-and-resized-TEST-before-pixels.json'),
       'fixed_first3_TRAIN_JSON':[r['members']['original/json']['member'] for r in chosen],
       'inspect':'datajson.loads only<=32MiB, preserve exact labels,shape_types,dimensions,imagePath asuntrustedtext; no imageData base64decode/pixels/geometryrender/codeexecution',
       'copy_identity':'group samepublisherrole+stem original/resized/JSON/masks. Same numericstem inTrainandTest alone is not physicalidentity or grounds to merge; inspect actualimagePath/pixels inlaterprotocol.',
       'label_semantics':'officialguideline textreadonly: Good includes no OR minimal corrosion; peelingpaintalone notcorrosion; fair/poor/severe visiblecorrosion severityannotations. Good/background is NOT verifiedabsence of allrust. Do not useallGood asbinaryrustnegative.',
       'mapping':'authorREADME BGRmapping is not anRGB/palette declaration foractualfiles; keep raw labelnames/colors, verify nativePNGmode/palette/JSONmapping before any maskconversion',
       'future':'no finalFIT_VAL_CAL orlearnprotocol. Original396TRAIN/44TESTfixed; pixel/physicalgroups/currentTESTexclusions before onefixedsplit, firstFIT QAonlyafterdatagate.',
       'new_source_pixeldecode_contentview_features_scores_gradient_optimizer':0,'license':'sourceuniversityCC0,codeMITseparate, cite bothdatasetDOI10.7294/16624663.v2 andpaper10.1061/(ASCE)CP.1943-5487.0001045',
       'independent_field_validation':False}
    new(POLICY,p);print({'policy_sha256':sha(POLICY),'reserved_TEST_sha256':p['reserved_TEST_sha256'],'original_photos':440},flush=True)


def inspect():
    p=read(POLICY);assert sha(ROOT/SCRIPT)==p['script_sha256'] and sha(SOURCE/'publisher-ccs-v2.zip')==p['archive_sha256']
    git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
    for n in (SCRIPT,POLICY.relative_to(ROOT).as_posix()):
        assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])).hexdigest()==sha(ROOT/n),n
    assert sha(OUT/'reserved-original-and-resized-TEST-before-pixels.json')==p['reserved_TEST_sha256']
    assert sha(OUT/'all-pairs-before-json-or-pixels.json')==p['allpairs_sha256'] and not (OUT/'first3TRAIN-JSON-inspection.json').exists()
    byname={v['member']:v for r in read(OUT/'all-pairs-before-json-or-pixels.json')['rows'] for v in r['members'].values()};result=[]
    with zipfile.ZipFile(SOURCE/'publisher-ccs-v2.zip') as archive:
        for name in p['fixed_first3_TRAIN_JSON']:
            assert '/original/Train/json/' in name
            raw=archive.read(name);assert len(raw)<=32*1024**2 and hashlib.sha256(raw).hexdigest()==byname[name]['encoded_sha256']
            value=json.loads(raw);assert isinstance(value,dict) and isinstance(value['shapes'],list)
            result.append({'member':name,'JSON_keys':sorted(value),'imagePath_untrusted_text':value.get('imagePath'),
                'imageWidth':value.get('imageWidth'),'imageHeight':value.get('imageHeight'),'shape_labels':dict(Counter(s['label'] for s in value['shapes'])),
                'shape_types':dict(Counter(str(s.get('shape_type')) for s in value['shapes'])),'group_ids':sorted({str(s.get('group_id')) for s in value['shapes']}),
                'imageData_present_not_decoded':bool(value.get('imageData')),'flags':value.get('flags')})
    summary={'UTC':utc(),'policy_sha256':sha(POLICY),'original_photo_pairs':440,'publisher_TRAIN':396,'reserved_TEST':44,
        'associated_TESTfiles':220,'declared_original_resized_variants_perphoto':2,'TRAIN_JSON_examples':result,
        'TEST_JSON_contents_read':0,'new_source_images_masks_or_guideline_example_images_viewed':0,'new_pixel_decodes_features_scores_gradient_optimizer':0,
        'final_pixel_class_mapping_verified':False,'final_development_split_fixed':False,'training_protocol_fixed':False,
        'Good_may_include_minimal_rust':True,'physical_part_or_report_groups_verified':False,'independent_field_validation':False}
    new(OUT/'first3TRAIN-JSON-inspection.json',summary);new(ROOT/'reports/industrial-corrosion-condition-v65-metadata-audit.json',summary);print(summary,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','inspect'));args=parser.parse_args();globals()[args.stage]()
