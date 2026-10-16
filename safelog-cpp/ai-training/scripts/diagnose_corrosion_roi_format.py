"""Header-only format diagnosis; cannot qualify the failed integer ROI contract."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path,PurePosixPath
import stat
import struct
import subprocess
import sys
import zipfile
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import prepare_corrosion_background as prior
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read
RUN=ROOT/'runs/industrial-corrosion-roi-format-v77'
POLICY=ROOT/'reports/industrial-corrosion-roi-format-v77-policy.json'
RECOVERY=ROOT/'reports/industrial-corrosion-roi-format-v77-r1-policy.json'
SCRIPT='scripts/diagnose_corrosion_roi_format.py'


def utc():return datetime.now(timezone.utc).isoformat()


def freeze():
    assert not RUN.exists() and not POLICY.exists();assert len(list(prior.RUN.glob('failure-*.json')))==2 and not list(prior.OUT.glob('decoded-*.json'))
    RUN.mkdir();paths=[prior.POLICY,prior.RECOVERY,prior.OUT/'reserved-TEST-and-copies-before-pixels.json',prior.ARCHIVE,prior.SOURCE/'publisher-labeled-members.json',prior.SOURCE/'imagej-static-receipt.json']+[prior.SOURCE/n for n in ('imagej-RoiDecoder.java.txt','imagej-Roi.java.txt')]
    new(POLICY,{'UTC':utc(),'fixed_before_new_header_diagnosis':True,'source_sha256':{SCRIPT:sha(ROOT/SCRIPT)},'inputs_sha256':{v.relative_to(ROOT).as_posix():sha(v) for v in paths},
        'scope':'All70nestedannotationZIPs/ROIheaders only, plus34originalTIFFIFDheaders. No RGBload/convert/fingerprints/model/features/scores/gradient/optimizer/visualcontent. Missing83annotations remainunpaired/nottraining.',
        'ROI':'Read boundednestedZIP<=1MiB/16members/8MiBexpanded/ROI<=1MiB, safe paths/no symlink/encryption/duplicate; CRC+encodedSHA. Inspect64byteIoutheader fields and float32slots18..33 only as metadata, includingversion/type/ncoords/shape/subtype/options/position/hdr2. Recordnonrectangle/extensionfields; no raster/coordinateGT construction oracceptance.',
        'TIFF':'OnlyImage.openmetadata n_frames/size/mode/orientation; no load/convert/array/EXIFrotation. VerifyencodedSHA; max128MiBencoded, noexpandedfilewriting.',
        'preserve':'v76integer-onlycontractfailedandstaysnotready. Do notignoreactualflags, shiftcoords, changeGT, weaken minimums, moveTEST/roles orstartlearning. A separatelypreregistereddocumentedformatcontractwouldbeneededafterthisdiagnosis.',
        'new_pixels_features_scores_gradient_optimizer':0,'new_visual_content_views':0,'independent_field_validation':False})
    print({'policy_sha256':sha(POLICY)},flush=True)


def run():
    active=RECOVERY if RECOVERY.exists() else POLICY;p=read(active)
    if RECOVERY.exists():assert sha(POLICY)==p['original_policy_sha256']
    for n,h in {**p['source_sha256'],**p['inputs_sha256']}.items():assert sha(ROOT/n)==h,n
    git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
    for n in (SCRIPT,active.relative_to(ROOT).as_posix()):assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])).hexdigest()==sha(ROOT/n),n
    records=[];headers=[]
    with zipfile.ZipFile(prior.ARCHIVE) as archive:
        for item in read(prior.SOURCE/'publisher-labeled-members.json')['rows']:
            name=item['member']
            if '/Labels/' in name and name.endswith('.zip'):
                raw=archive.read(name);assert hashlib.sha256(raw).hexdigest()==item['encoded_SHA256'] and len(raw)<=1024**2;seen=set()
                with zipfile.ZipFile(BytesIO(raw)) as inner:
                    assert len(inner.infolist())<=16 and sum(v.file_size for v in inner.infolist())<=8*1024**2
                    for v in inner.infolist():
                        path=PurePosixPath(v.filename);assert not path.is_absolute() and '..' not in path.parts and ':' not in v.filename and '\\' not in v.filename and not stat.S_ISLNK(v.external_attr>>16) and not v.flag_bits&1
                        assert v.filename.casefold() not in seen;seen.add(v.filename.casefold())
                        if v.is_dir():continue
                        assert v.filename.endswith('.roi') and v.file_size<=1024**2
                        b=inner.read(v);assert len(b)>=64 and b[:4]==b'Iout';u=lambda i:struct.unpack_from('>H',b,i)[0];s=lambda i:struct.unpack_from('>h',b,i)[0];integer=lambda i:struct.unpack_from('>i',b,i)[0]
                        records.append({'outer_member':name,'roi_member':v.filename,'bytes':len(b),'SHA256':hashlib.sha256(b).hexdigest(),'CRC32':f'{v.CRC:08x}',
                          'version':u(4),'type':b[6],'int_top_left_bottom_right':[s(i) for i in (8,10,12,14)],'n_coordinates':u(16),'shape_size':integer(36),'subtype':u(48),'options':u(50),'arc':u(54),'position':integer(56),'hdr2':integer(60),'float32_slots_18_22_26_30':list(struct.unpack_from('>ffff',b,18))})
            elif '/Original images/' in name and name.endswith('.tif'):
                raw=archive.read(name);assert hashlib.sha256(raw).hexdigest()==item['encoded_SHA256'] and len(raw)<=128*1024**2
                with Image.open(BytesIO(raw)) as im:headers.append({'member':name,'size':list(im.size),'mode':im.mode,'frames':im.n_frames,'orientation':im.getexif().get(274),'format':im.format})
    new(RUN/'ROI-headers.json',{'rows':records});new(RUN/'TIFF-headers.json',{'rows':headers})
    result={'UTC':utc(),'ROI_headers':len(records),'expected_ROI_count_if5perclass':350,'ROI_count_matches_publisher5perclass':len(records)==350,'per_zip_ROI_counts':dict(Counter(r['outer_member'] for r in records)),'nestedZIPs':70,'TIFF_IFD_headers':len(headers),'versions':dict(Counter(r['version'] for r in records)),'types':dict(Counter(r['type'] for r in records)),'options':dict(Counter(r['options'] for r in records)),'ncoords':dict(Counter(r['n_coordinates'] for r in records)),
      'subpixel_flag':sum(bool(r['options']&128) for r in records),'nonzero_shape_size':sum(r['shape_size']!=0 for r in records),'nonzero_subtype':sum(r['subtype']!=0 for r in records),'nonzero_arc':sum(r['arc']!=0 for r in records),'float_rect_width_height_counts':dict(Counter(str(r['float32_slots_18_22_26_30'][2:]) for r in records)),
      'RGB_pixels_loaded':0,'ROI_rasters_constructed':0,'new_features_scores_gradient_optimizer':0,'visual_content_views':0,'v76_integer_contract_still_failed':True,'training_ready':False}
    new(RUN/'diagnosis.json',result);new(ROOT/'reports/industrial-corrosion-roi-format-v77-diagnosis.json',result);print(result,flush=True)


def freeze_recovery():
    assert not RECOVERY.exists() and not (RUN/'ROI-headers.json').exists()
    assert sha(POLICY)=='335aea842a28a9c6c18c28e65ef057b1c3f94b9433974cd9473808f21c1dc6b4'
    new(RUN/'failure-initial-count-assumption.json',{'UTC':utc(),'error':'AssertionError len(records)==350 and len(headers)==34 before output persistence','actual_counts_not_preserved_by_original_script':True,'all_header_reads_had_completed_before_assertion':True,'RGB_pixel_decodes':0,'ROI_rasters':0,'optimizer':0,'learning_minima_not_changed':True})
    p=read(POLICY);p.update(UTC=utc(),original_policy_sha256=sha(POLICY),source_sha256={SCRIPT:sha(ROOT/SCRIPT)},recovery='Diagnostic output repair only: persist actual ROI/header inventory/count differences ratherthan assert publisherdescriptioncount beforewrite. Re-readheadersencodedbytes, noRGBpixels. This doesnotqualifyv76orrelax5patch learningcontract. Preservefirstscript inGit/failure; no training orGTchanges.')
    new(RECOVERY,p);print({'recovery_policy_sha256':sha(RECOVERY)},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','freeze-recovery','run'));args=parser.parse_args()
    if args.stage=='freeze':freeze()
    elif args.stage=='freeze-recovery':freeze_recovery()
    else:
        try:run()
        except BaseException as error:
            new(RUN/('failure-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'UTC':utc(),'error':repr(error)});raise
