"""Reuse the immutable API/C++ smoke with explicit v16 composition differences."""
import inspect
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import study_real_mask_transfer as study
from scripts import release_loco_context as previous
from scripts.fetch_equipment_expansion import sha


def main():
    p=study.validate();assert sha(ROOT/'scripts/release_loco_context.py')==p['legacy_source_sha256']['scripts/release_loco_context.py']
    target=ROOT/'scripts/release_real_mask_transfer.py';assert not target.exists()
    text=inspect.getsource(previous.smoke)
    text=text.replace('safelog_ai.loco_context_server','safelog_ai.real_mask_server')
    text=text.replace("REPORTS/'industrial-ksdd-photo-only-v11-release.json'","REPORTS/'industrial-loco-context-v13-release.json'")
    text=text.replace("elif task != 'loco_splicing_connectors'", "elif task != 'mpdd_bracket_black'")
    text=text.replace("('ksdd2_surface', 'ball_screw_pitting', 'loco_splicing_connectors')", "('ksdd2_surface', 'ball_screw_pitting', 'loco_splicing_connectors', 'mpdd_bracket_black')")
    marker="    point = study.read(trained.RUN/'calibration.json')['point'];"
    insertion="""    candidates = study.read(RUN/'candidate-evaluation.json')['rows']
    prior_rows = {r['id']: r for r in study.read(study.PARENT_EVAL)['rows']}
    newly = [r for r in candidates if r['category']=='mpdd_bracket_black' and r.get('method')=='max_transfer_local' and r['candidate_detected'] and not prior_rows[r['id']]['candidate_detected']]
    black_expected = max(newly, key=lambda r:r['score'])
    photos['mpdd_bracket_black'] = ROOT/next(r['image'] for r in study.rows_for('test','mpdd_bracket_black') if r['id']==black_expected['id'])
"""
    assert marker in text;text=text.replace(marker,insertion+marker)
    marker='            for key in [k for k in reply if k.endswith(\'PngBase64\')]: reply.pop(key)'
    insertion="""            if task == 'mpdd_bracket_black':
                from safelog_ai.loco_context_server import ContextEngine
                parent = ContextEngine(study.START, study.EXPECTED, study.PARENT_EVAL); original = parent.analyze(task, encoded)
                assert reply['heatmapPngBase64'] == original['heatmapPngBase64']
                assert abs(reply['classResults'][0]['score']-black_expected['score']) < 2e-4
                assert reply['candidateDetected'] and not original['candidateDetected']
                assert reply['originalLocalizationChanged'] is False and not reply['sourceHeldoutPhotoBenchmarkPassed']
                assert reply['transferIsValidatedPixelLocalization'] is False
                del parent
"""
    assert marker in text;text=text.replace(marker,insertion+marker)
    text=text.replace("write(REPORTS/'industrial-loco-context-v13-release.json'", "write(REPORTS/'industrial-real-mask-transfer-v16-release.json'")
    text=text.replace("'actual_current_CPP_operations': 9", "'actual_current_CPP_operations': 12")
    text=text.replace("'actual_CPP_operations': 9", "'actual_CPP_operations': 12")
    text=text.replace("'all_other_exposed_policy_SHA_preserved': True", "'all_other_exposed_policy_SHA_preserved': True, 'new_black_photo_candidate_and_original_heatmap_checked': True, 'new_source_self_benchmark_failure_explicit': True")
    header='''"""Current ZIP API15/C++12 checks. Prior inspection counts are not reused."""
import argparse,base64,hashlib,io,json,os,shutil,socket,subprocess,sys,time,zipfile
from pathlib import Path
from urllib.error import HTTPError
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import study_real_mask_transfer as study
from scripts import study_ksdd_supervised as trained
from scripts.fetch_equipment_expansion import sha
from scripts.release_metal_calibration import write,http
from safelog_ai.equipment_expansion import CATEGORIES
from safelog_ai.industrial_supervised import WEAR_CATEGORY
RUN,REPORTS=study.RUN,study.REPORTS;REPO=ROOT.parents[1];OUTPUTS=ROOT.parents[3]/'outputs'
STAGE=RUN/'handoff-stage';ISOLATED=RUN/'isolated-zip'
ARCHIVE=OUTPUTS/'unified-real-mask-transfer-role4-handoff-2026-10-15-v16.zip'

def checked():
    study.validate();value=study.read(RUN/'package.json')
    assert study.read(RUN/'verification.json')['status']=='passed'
    assert value['selected']=={'mpdd_bracket_black':'max_transfer_local'}
    assert sha(ROOT/value['checkpoint_path'])==value['checkpoint_sha256']
    return value

'''
    footer='''
if __name__=='__main__':
    torch.set_num_threads(4)
    lock=RUN/'release.lock'
    with lock.open('x',encoding='utf-8') as f:f.write(str(os.getpid()))
    try:smoke()
    except BaseException as error:
        write(RUN/('release-failure-'+str(int(time.time()))+'.json'),{'error':repr(error),'pid':os.getpid()});raise
    finally:lock.unlink()
'''
    with target.open('x',encoding='utf-8',newline='\n') as f:f.write(header+text+footer)
    print({'release_script_created':str(target)},flush=True)


if __name__=='__main__':main()
