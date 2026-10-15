"""Compose only selected black-bracket photo behavior into one research ZIP."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import zipfile
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import study_real_mask_transfer as study
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
RUN=study.RUN;OUTPUTS=ROOT.parents[3]/'outputs'


def main():
    study.validate();selection=study.read(RUN/'selection.json');train=study.read(RUN/'trained/training.json')
    assert selection['selected']=={'mpdd_bracket_black':'max_transfer_local'}
    assert study.read(RUN/'verification.json')['status']=='passed'
    state=torch.load(RUN/'selected/final.pt',map_location='cpu',weights_only=True)
    state.update(transfer_training_actual_updates=train['actual_optimizer_updates'],transfer_selected_epoch=train['selected_epoch'],
        transfer_selected_epoch_updates=train['selected_epoch_updates'],transfer_source_holdout_benchmark=selection['categories']['magnetic_tile_surface'])
    state['trained_dataset_sources']=[*state['trained_dataset_sources'],'Author magnetic tile masks, academic research only: https://github.com/abin24/Magnetic-tile-defect-datasets.']
    release=RUN/'released';assert not release.exists();release.mkdir();torch.save(state,release/'final.pt');digest=sha(release/'final.pt')
    evaluation=deepcopy(study.read(study.PARENT_EVAL));c='mpdd_bracket_black';method=selection['selected'][c];metric=selection['categories'][c][method]
    old=evaluation['categories'][c]
    old.update(parent_photo_rates=deepcopy(old['photo_rates']),photo_rates=metric['rates'],by_defect=metric['by_defect'],auroc=metric['auroc'],
        stage_gate=study.stage_gate(metric['rates']['fpr'],metric['rates']['fnr'],metric['rates']['negative_photos'],metric['rates']['positive_photos']),
        photo_threshold=state['transfer_surface_points'][c]['photo_threshold'],normal_CAL_photos=30,photoScoreSource=method,
        original_location_preserved=True,transfer_is_evaluated_pixel_localization=False)
    selected_rows={r['id']:r for r in study.read(RUN/'candidate-evaluation.json')['rows'] if r['category']==c and r.get('method')==method}
    for row in evaluation['rows']:
        if row['id'] in selected_rows:
            r=selected_rows[row['id']];row.update(score=r['score'],candidate_detected=r['candidate_detected'])
    evaluation.update(checkpoint_sha256=digest,source_mask_holdout_benchmark=selection['categories']['magnetic_tile_surface'],
        known_TEST_reused=True,adaptive_development_on_known_TEST=True,independent_industrial_field_validated=False)
    evaluation['stage20_categories']=[c for c,r in evaluation['categories'].items() if r.get('stage_gate',{}).get('ready_for_next_stage')]
    assert len(evaluation['stage20_categories'])==8
    new(RUN/'evaluation.json',evaluation)
    stage=RUN/'handoff-stage';assert not stage.exists();stage.mkdir();(stage/'safelog_ai').mkdir()
    shutil.copyfile(release/'final.pt',stage/'model.pt');shutil.copyfile(RUN/'evaluation.json',stage/'evaluation.json')
    previous=ROOT/'runs/industrial-loco-context-v13/handoff-stage'
    for source in (previous/'safelog_ai').glob('*.py'):shutil.copyfile(source,stage/'safelog_ai'/source.name)
    for name in ('magnetic_surface.py','real_mask_server.py'):shutil.copyfile(ROOT/'safelog_ai'/name,stage/'safelog_ai'/name)
    shutil.copyfile(previous/'requirements-api.txt',stage/'requirements-api.txt')
    model_info={'path':'model.pt','sha256':digest,'checkpoint_count':1,'backbone_count':1,'method':selection['selected'],
        'actual_new_optimizer_updates':train['actual_optimizer_updates'],'selected_epoch':train['selected_epoch'],'selected_epoch_updates':train['selected_epoch_updates'],
        'new_source_self_benchmark_passed':False,'known_target_TEST_reused':True,'independent_field_validated':False,'app_default_promoted':False}
    new(stage/'model.json',model_info)
    for path in (ROOT.parent/'docs/REAL_MASK_TRANSFER_HANDOFF_KO.md',ROOT/'reports/REAL_MASK_TRANSFER_RESULTS_KO.md'):
        target=stage/path.relative_to(ROOT.parents[1]);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    (stage/'README_KO.md').write_text('# SafeLog 실제 마스크 학습 v16\n\n검정 브래킷 사진 미탐 부분 개선. 목표20%는 미달이며 기존 위치 지도를 보존한다.\n\n- [역할4 실행 안내](safelog-cpp/docs/REAL_MASK_TRANSFER_HANDOFF_KO.md)\n- [학습·평가 결과](safelog-cpp/ai-training/reports/REAL_MASK_TRANSFER_RESULTS_KO.md)\n',encoding='utf-8')
    hashes={p.relative_to(stage).as_posix():sha(p) for p in stage.rglob('*') if p.is_file()};new(stage/'package-sha256.json',hashes)
    archive=OUTPUTS/'unified-real-mask-transfer-role4-handoff-2026-10-15-v16.zip';assert not archive.exists()
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for path in sorted(stage.rglob('*')):
            if path.is_file():z.write(path,path.relative_to(stage).as_posix())
    record={'checkpoint_path':(release/'final.pt').relative_to(ROOT).as_posix(),'checkpoint_sha256':digest,'candidate_before_metadata_SHA256':sha(RUN/'selected/final.pt'),
        'archive':str(archive),'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'selected':selection['selected'],'neural_new_updates_during_packaging':0,'CPU_CPP_release_verified':False}
    new(RUN/'package.json',record);print(record,flush=True)


if __name__=='__main__':main()
