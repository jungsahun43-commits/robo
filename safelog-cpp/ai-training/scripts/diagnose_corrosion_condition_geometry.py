"""Header inventory plus fixed TRAIN-only audit of an authoritative raster canvas."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import numpy as np
from PIL import Image, ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from safelog_ai.corrosion_condition_data import native_classes,polygon_classes,LABELS
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read

SOURCE=ROOT/'data/industrial-corrosion-condition-v64-source'
META=ROOT/'data/industrial-corrosion-condition-v65-metadata'
RUN=ROOT/'runs/industrial-corrosion-condition-geometry-v67'
POLICY=ROOT/'reports/industrial-corrosion-condition-v67-diagnostic-policy.json'
SCRIPT='scripts/diagnose_corrosion_condition_geometry.py'


def utc():return datetime.now(timezone.utc).isoformat()


def freeze():
    assert not RUN.exists() and not POLICY.exists()
    selected=sorted([r for r in read(META/'all-pairs-before-json-or-pixels.json')['rows'] if r['publisher_role']=='Train'],key=lambda r:r['members']['original/json']['member'])[:3]
    assert [r['stem'] for r in selected]==['0','1','10']
    files=[SOURCE/'publisher-ccs-v2.zip',SOURCE/'author-labelme2voc-static.py.txt',SOURCE/'labelme-v3.16.7-shape-static.py.txt',META/'all-pairs-before-json-or-pixels.json',ROOT/'safelog_ai/corrosion_condition_data.py',ROOT/'reports/industrial-corrosion-condition-v66-data-policy.json',ROOT/'reports/industrial-corrosion-condition-v66-r1-recovery-policy.json']
    files+=list((ROOT/'runs/industrial-corrosion-condition-data-v66').glob('failure-*.json'))
    p={'UTC':utc(),'fixed_before_new_header_inventory_or_TRAIN_reference_comparison':True,'script_sha256':sha(ROOT/SCRIPT),'inputs_sha256':{v.relative_to(ROOT).as_posix():sha(v) for v in files},
       'header_inventory':'all440 original/copy image/mask encodedSHA checked; PIL open/getexif/size/mode/format only, no image.load/convert/autorotate or pixels for headers. Preserve actual format, EXIF orientation and coordinate mismatch; do not fix extensions, masks, or orientation.',
       'fixed_TRAIN_only_reference_ids':[r['id'] for r in selected],
       'reference':'Author converter calls labelme shapes_to_label. Official labelme v3.16.7 static source uses uint8 zeros -> PIL L canvas, then ordered draw.polygon outline1/fill1. Independently implement this exact canvas; original v66 used PIL mode1. Compare existing native PNG to both without choosing algorithms from TEST. Representative version is not proof of the publisher installed version. No source code execution.',
       'geometry':'Native mask/JSON dimensions and exact classmapping preserved, unknownshape or invalidcoordinate stops TRAIN reference. No GT replacement, no tolerance sweep, no learning/thresholds/TEST polygon comparisons.',
       'scope':'Header metadata for reservedTEST is exclusion/format diagnosis only; decode three fixed originalTRAIN PNG masks and reference rasters, sourceRGB content views0, neuralfeatures/score/gradient/optimizer0. No new split or relaxation of original v66 minimums/gate.',
       'future':'If auditor defect independently confirmed, fix code in a new preregistered recovery while preserving failures; remaining unsupported originals must not silently become valid. Coordinate mismatch needs separate evidence, not forced resizing.',
       'storage':'smallmetadata <8MiB, >=1GiBreserve, no full archive extraction/cache/modelcopy',
       'independent_field_validation':False,'new_optimizer_updates':0}
    assert shutil.disk_usage(ROOT).free>1024**3+8*1024**2
    RUN.mkdir();new(POLICY,p);print({'policy_sha256':sha(POLICY)},flush=True)


def run():
    p=read(POLICY);assert sha(ROOT/SCRIPT)==p['script_sha256']
    for n,h in p['inputs_sha256'].items():assert sha(ROOT/n)==h,n
    git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
    for n in (SCRIPT,POLICY.relative_to(ROOT).as_posix()):assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])).hexdigest()==sha(ROOT/n)
    assert not (RUN/'diagnosis.json').exists()
    rows=read(META/'all-pairs-before-json-or-pixels.json')['rows'];inventory=[];reference=[]
    with zipfile.ZipFile(SOURCE/'publisher-ccs-v2.zip') as archive:
        def get(item):
            assert item['bytes']<=32*1024**2
            raw=archive.read(item['member']);assert hashlib.sha256(raw).hexdigest()==item['encoded_sha256'];return raw
        for r in rows:
            headers={}
            for key in ('original/images','original/masks','512x512/images_512','512x512/mask_512'):
                with Image.open(io.BytesIO(get(r['members'][key]))) as im:
                    headers[key]={'size':list(im.size),'mode':im.mode,'format':im.format,'EXIF_orientation':im.getexif().get(274)}
            problems=[]
            for key,v in headers.items():
                fmt='JPEG' if key.endswith(('images','images_512')) else 'PNG'
                if v['format']!=fmt:problems.append({'kind':'format','variant':key,'expected':fmt,'actual':v['format']})
                if v['EXIF_orientation'] not in (None,1):problems.append({'kind':'EXIF_orientation','variant':key,'actual':v['EXIF_orientation']})
                if v['size'][0]*v['size'][1]>40_000_000:problems.append({'kind':'size_limit','variant':key,'actual':v['size']})
            for a,b in [('original/images','original/masks'),('512x512/images_512','512x512/mask_512')]:
                if headers[a]['size']!=headers[b]['size']:problems.append({'kind':'coordinate_size','image':headers[a]['size'],'mask':headers[b]['size'],'variant':a})
            inventory.append({'id':r['id'],'headers':headers,'problems':problems})
            if r['id'] in p['fixed_TRAIN_only_reference_ids']:
                annotation=json.loads(get(r['members']['original/json']))
                with Image.open(io.BytesIO(get(r['members']['original/masks']))) as mask:
                    native=native_classes(mask);size=mask.size
                # Read only static source; independently implement the documented uint8/L canvas.
                result=np.zeros((size[1],size[0]),dtype=np.uint8)
                for shape in annotation['shapes']:
                    assert shape['label'] in LABELS and shape.get('shape_type','polygon')=='polygon'
                    canvas=Image.fromarray(np.zeros((size[1],size[0]),dtype=np.uint8))
                    ImageDraw.Draw(canvas).polygon([tuple(v) for v in shape['points']],outline=1,fill=1)
                    result[np.asarray(canvas,dtype=bool)]=LABELS[shape['label']]
                mode1=polygon_classes(annotation,size)
                reference.append({'id':r['id'],'JSON_version':annotation.get('version'),'original_mask_mode':mask.mode,
                    'native_vs_v66_mode1_mismatch':int(np.count_nonzero(native!=mode1)),
                    'native_vs_official_L_canvas_mismatch':int(np.count_nonzero(native!=result)),
                    'v66_vs_official_canvas_mismatch':int(np.count_nonzero(mode1!=result)),
                    'native_class_counts':dict(zip(*[v.tolist() for v in np.unique(native,return_counts=True)]))})
    summary={'UTC':utc(),'policy_sha256':sha(POLICY),'original_entries':len(rows),'header_files_read_no_pixel_decode':len(rows)*4,
             'header_problem_entries':sum(bool(r['problems']) for r in inventory),
             'header_problem_types':dict(Counter(v['kind'] for r in inventory for v in r['problems'])),
             'fixed_TRAIN_mask_reference_comparisons':reference,'new_RGB_pixel_decodes_or_image_views':0,
             'new_TRAIN_native_mask_decodes':3,'new_TEST_mask_pixel_or_polygon_comparisons':0,
             'new_neural_features_scores_gradient_optimizer':0,'v66_minimum_or_TEST_policy_changes':0,
             'can_claim_full_data_ready':False,'training_protocol_fixed':False,'independent_field_validation':False}
    new(RUN/'headers.json',{'rows':inventory});new(RUN/'diagnosis.json',summary)
    new(ROOT/'reports/industrial-corrosion-condition-v67-diagnosis.json',summary);print(summary,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','run'));args=parser.parse_args();globals()[args.stage]()
