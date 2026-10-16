"""FIT-only annotation geometry, not another model or TEST-score trial."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from safelog_ai.native_corrosion_grid import grid,annotation_tile,fine_cells
from safelog_ai.corrosion_subpixel_roi import provided_target
from safelog_ai.corrosion_background_roi import target512
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read
MANIFEST=ROOT/'data/industrial-corrosion-subpixel-v78/manifest.json'
RUN=ROOT/'runs/industrial-corrosion-native-geometry-v80'
POLICY=ROOT/'reports/industrial-corrosion-native-geometry-v80-policy.json'
SOURCES=('scripts/probe_corrosion_native_geometry.py','safelog_ai/native_corrosion_grid.py','tests/test_native_corrosion_grid.py')


def utc():return datetime.now(timezone.utc).isoformat()


def freeze():
    assert not RUN.exists() and not POLICY.exists();assert read(MANIFEST)['audit']['ready'];RUN.mkdir()
    files=[MANIFEST,ROOT/'safelog_ai/corrosion_subpixel_roi.py',ROOT/'safelog_ai/corrosion_background_roi.py',ROOT/'runs/industrial-corrosion-background-v79/verification.json',ROOT/'runs/industrial-corrosion-background-v79/selection.json']
    assert not read(files[-1])['combined_source_research_gate_passed']
    new(POLICY,{'UTC':utc(),'fixed_before_new_geometry_probe':True,'source_sha256':{n:sha(ROOT/n) for n in SOURCES},'inputs_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in files},
       'scope':'FIT15only, reconstructprovidedGTfromcachedv78ROI/sizeandverifySHAs. No originalRGB/ZIPreads/imagecontentview/encoderfeatures/modelscore/gradient/optimizer ornewVAL_CAL_TEST calculations. No weight/threshold/loss tuning.',
       'one_geometry':'Input-onlynative512tiles,stride384,start0regularstepsthenappendaxislength-512ifmissing; axis<=512start0 withright/bottomRGBpad124,116,104/targetunknown255. NativeRGBno resize; futureinferenceallgridtiles/overlapprobabilitymean. GTmaymarkeligibleFITtrainingtiles laterbutneverchooseCAL_TESTinferencetiles.',
       'measure':'Onlynative/wholeletterbox512labelcounts,stride4geometricbins,andallnativegridlabelcounts/overlapmultiplicity. Stride4bin support isnotencoderresponse/receptivefield/accuracy orperformancepromise. RepeatedoverlapknownpixelsnotnewGT/naturalphotos.',
       'contract':'Sourcecoordinates/classes/unknown/TEST/group roles/minimaexactunchanged. VerifyeachFITknownnativepixelcoveredbygridandmatchingoverlaplabels. No GTdilation/patchsize5assumption/wholebackgroundorhealthlabels.',
       'next':'No modelprotocolfixedbythisprobe. If geometry plausible, separatelypreregisteroneexactmodelinitializer/trainparams/normalization/batch-sampling/loss/epoch/VAL/CAL/retention/sparsepointguards/device/delta beforefeatures; no multiplegrids/structures/TESTcuttrials. Existingv79failedandalloldmodels/outputsretained.',
       'storage':'SequentialnativeGTbuffers; metadataonly<32MiB/no densefeatures/newimages/parentclone/deletion; >=1GiBreserve. Threecoordinateunitschecked.',
       'license':'Rios/Netto/RoehlMendeleyV1CC BY4.0 DOI10.17632/kcyn4nhv2c.1/author-source-version-geometrychangesandallparenttermsretained','new_units_passed':3,'independent_field_validation':False})
    print({'policy_sha256':sha(POLICY)},flush=True)


def run():
    p=read(POLICY)
    for n,h in {**p['source_sha256'],**p['inputs_sha256']}.items():assert sha(ROOT/n)==h,n
    git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
    for n in (*SOURCES,POLICY.relative_to(ROOT).as_posix()):assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])).hexdigest()==sha(ROOT/n),n
    result=[]
    for r in [r for r in read(MANIFEST)['rows'] if r['phase']=='fit']:
        native,_=provided_target(tuple(r['size']),r['positive_ROIs'],r['negative_ROIs']);small,_=target512(native)
        assert hashlib.sha256(native.tobytes()).hexdigest()==r['native_sparse_SHA256'] and hashlib.sha256(small.tobytes()).hexdigest()==r['target512_SHA256']
        tiles=[];coverage=np.zeros_like(native,dtype=np.uint8)
        for b in grid(tuple(r['size'])):
            target=annotation_tile(native,b);coverage[b['top']:b['top']+b['height'],b['left']:b['left']+b['width']]+=1
            assert np.array_equal(target[:b['height'],:b['width']],native[b['top']:b['top']+b['height'],b['left']:b['left']+b['width']])
            tiles.append({**b,'provided_pixels':{str(k):int((target==k).sum()) for k in (0,1)},'stride4_label_bins':{str(k):fine_cells(target,k) for k in (0,1)}})
        assert np.all(coverage>0)
        result.append({'id':r['id'],'size':r['size'],'group_id':r['group_id'],'whole512_pixels':r['pixels512'],'whole512_stride4_label_bins':{str(k):fine_cells(small,k) for k in (0,1)},'native_unique_pixels':r['native_pixels'],'tiles':tiles,'all_input_grid_tiles':len(tiles),'tiles_with_positive':sum(t['provided_pixels']['1']>0 for t in tiles),'tiles_with_negative':sum(t['provided_pixels']['0']>0 for t in tiles),'all_known_pixels_covered':True,'overlap_multiplicity_min_max':[int(coverage.min()),int(coverage.max())]})
    assert len(result)==15
    summary={'UTC':utc(),'FIT_photos':15,'original_RGB_decodes':0,'new_encoder_features_model_scores_gradient_optimizer':0,'new_VAL_CAL_TEST':0,'new_source_visual_views':0,'original_native_GT_and_existing512_SHA_exact':True,'input_grid_tiles':sum(r['all_input_grid_tiles'] for r in result),'positive_support_tiles':sum(r['tiles_with_positive'] for r in result),'negative_support_tiles':sum(r['tiles_with_negative'] for r in result),
       'whole512_stride4_label_bins':{str(k):sum(r['whole512_stride4_label_bins'][str(k)] for r in result) for k in (0,1)},'nativegrid_stride4_label_bin_exposures_including_overlap':{str(k):sum(t['stride4_label_bins'][str(k)] for r in result for t in r['tiles']) for k in (0,1)},'source_unique_native_pixels':{str(k):sum(r['native_unique_pixels'][str(k)] for r in result) for k in (0,1)},'nativegrid_provided_pixel_exposures_including_overlap':{str(k):sum(t['provided_pixels'][str(k)] for r in result for t in r['tiles']) for k in (0,1)},'all_known_pixels_covered':True,'new_training_protocol_fixed':False,'performance_improvement_proven':False,'new_units_passed':3}
    new(RUN/'probe.json',{'summary':summary,'rows':result});new(ROOT/'reports/industrial-corrosion-native-geometry-v80-probe.json',{'summary':summary,'rows':result});print(summary,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','run'));args=parser.parse_args()
    if args.stage=='freeze':freeze()
    else:run()
