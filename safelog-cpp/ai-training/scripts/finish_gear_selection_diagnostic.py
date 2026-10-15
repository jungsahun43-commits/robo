"""Preserve the VAL-only result and prepare a data-feasible next hypothesis."""
from datetime import datetime,timezone
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import diagnose_gear_selection as study
from scripts import study_gear_task_adapter as trained
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read


def main():
    diag=read(study.RUN/'result.json');previous=read(trained.RUN/'final-audit.json')
    assert diag['actual_new_optimizer_updates']==0 and diag['new_CAL_TEST_scores']==0
    assert not diag['development_exact_RGB_opposite_label_groups'] and not diag['cross_FIT_VAL_near_pairs']
    support={str(y):[g for g,c in diag['FIT_VAL_group_label_counts']['fit'].items() if c.get(str(y),0)>=2] for y in (0,1)}
    fit_pairs=[r for r in diag['development_near_opposite_label_pairs'] if r['phase']=='fit']
    feasibility={'UTC':datetime.now(timezone.utc).isoformat(),'stage':'metadata_and_synthetic_loss_only_not_training_protocol',
        'manifest_sha256':sha(trained.MANIFEST),'source_diagnostic_sha256':sha(study.RUN/'result.json'),
        'FIT_group_support_with_atleast2patches_perlabel':support,'proposed_four_groups_two_distinct_patches_perlabel_feasible':all(len(v)>=4 for v in support.values()),
        'possible_batch':'4differentFITgroups perlabel,2distinctpatches pergroup,16total; groupids used for sampling/loss only, no inference input or invented embedding',
        'loss_prototype':'same-label different-group positives; exclude same-group pairs from all contrastive terms; normalizedcosine logsumexp soft-neighbor loss. TechnicaldefaultT.2, full lambda/temperature/optimizer/VAL/CAL protocol not yet fixed.',
        'loss_source_sha256':sha(ROOT/'safelog_ai/gear_cross_group_loss.py'),'new_synthetic_unit_tests_passed':3,
        'original_GT_unchanged':True,'pending_FIT_only_near_opposite_label_pairs':fit_pairs,
        'actual_source_gradients_or_optimizer':0,'new_model_scores':0,'new_training_protocol_fixed':False,
        'paper_motivation_only_not_method_reproduction':'https://arxiv.org/pdf/2511.15440 sectionsIII/IV discuss per-component imbalance and contrastive generalization; this is a distinct cross-group prototype, no author performance claimed'}
    new(ROOT/'reports/industrial-gear-cross-group-v33-feasibility.json',feasibility)
    r={'UTC':datetime.now(timezone.utc).isoformat(),'status':'v29_v30_v31_complete_rejected_v32_VAL_diagnostic_complete_v33_loss_prototype_only',
        'active_model':previous['active_model'],'active_model_sha256':previous['active_model_sha256'],
        'active_role4_ZIP':previous['active_role4_ZIP'],'active_role4_ZIP_sha256':previous['active_role4_ZIP_sha256'],
        'v29_actual_optimizer':1920,'v29_selected_epoch':10,'v29_selected_updates':640,'v29_TEST_FPR':239/937,'v29_TEST_FNR':333/1242,
        'v30_actual_optimizer':1920,'v30_selected_epoch':0,'v31_actual_optimizer':1920,'v31_selected_epoch':0,
        'v30_v31_selected_epoch0_not_new_learning_improvement':True,
        'v31_research_delta':previous['research_delta'],'v31_research_delta_sha256':previous['research_delta_sha256'],
        'v31_research_delta_bytes':42125,'v31_delta_needs_parent_not_standalone_app':True,
        'this_turn_v29_v30_v31_actual_optimizer':5760,'tracked_partial_research_optimizer_including_failed':37840,'tracked_sum_not_entire_history':True,
        'new_API_CPP_ZIP_checks':0,'new_ZIPs':0,'v32_diagnostic':diag,'v33_feasibility':feasibility,
        'v33_actual_optimizer_or_source_gradients':0,'v33_training_protocol_fixed':False,
        'next':'Read GEAR_CROSS_GROUP_NEXT_KO.md. Preregister one cross-group contrastive regularization candidate on unchanged v28 gear split/shared adapter, fix positive masks/sampling/embedding/temperature/coefficient/init/cache/optimizer/epochs/VAL/CAL/guards/delta restoration before actual source gradients. Do not rerun old heads or promote last checkpoints from v32.',
        'old_app_original_positions_KS_BS_LOCO_and8of12_targets_unchanged':True,'new_independent_field_validation':False,
        'all_originals_failed_weights_labels_reports_preserved':True,'awake_helper_restarted':False,'free_disk_bytes':shutil.disk_usage(ROOT).free}
    new(ROOT.parents[3]/'outputs/INDUSTRIAL_GEAR_SELECTION_RESUME_2026-10-15-v32-DIAGNOSIS.json',r)
    new(ROOT/'reports/industrial-gear-selection-v32-preparation.json',r)
    lines=['# v32 VAL 전용 진단 결과','',
        '후보8개(selected/last 각4종)를 사전 고정한 뒤 같은 VAL1,440개에서만 비교했다. 새 optimizer·RGB추출·CAL/TEST점수·checkpoint반영은0이다. 원래 선택결과와 실패기록은 유지한다.','',
        '| 모델 | 선택epoch | 선택 BCE | 마지막 BCE | 선택 가중AUROC | 마지막 가중AUROC | 차이 |','|---|---:|---:|---:|---:|---:|---:|']
    for name,value in diag['results'].items():
        a,b=value['selected'],value['last'];lines.append(f"| {name} | {a['epoch']} | {a['VAL_macro_label_group_BCE']:.6f} | {b['VAL_macro_label_group_BCE']:.6f} | {a['VAL_label_group_weighted_AUROC']:.6f} | {b['VAL_label_group_weighted_AUROC']:.6f} | {value['last_minus_selected_weighted_AUROC']:+.6f} |")
    lines+=['','마지막 가중치의 가중AUROC 증가 폭은최대 .01163이며 v31은 .00190 낮았다. 선택기준만 바꾸면 큰 개선이 나온다고 결론낼 근거가 부족하다. 마지막 checkpoint를 새 TEST에 평가하거나 반영하지 않았다.','',
        'VAL은 nOK744개만 있는 묶음35, OK497/nOK7인 묶음54, OK192개만 있는 묶음72다. 부품별 클래스 불균형과 드문 사례7개의 불확실성을 남긴다. 정확RGB의 반대라벨 묶음0개, FIT/VAL 사이 근접쌍0개다. 근접 반대라벨은7쌍(FIT5/VAL2)이며 오류로 단정하지 않았고 원래 라벨을 바꾸지 않았다. VAL 예시 이미지는 열람하지 않았다.','',
        '다음 가설은 다른 부품에서도 같은 마모 상태의 특징이 가까워지도록 돕는 cross-group 보조손실이다. 현재 FIT에서 라벨별2장 이상을 가진 그룹이 OK6개/nOK8개이므로 라벨별4그룹×2장 샘플링을 구성할 수 있다. 학습전 코드와 JSON에 구조·초기값·배치·embedding·temperature·loss계수·학습/선택/CAL/guard를 별도 고정해야 한다. 이번에는 수학 helper와 synthetic unit3개만 검증했고 실제 sourcegradient·optimizer는0이다.','',
        '[원저자 논문](https://arxiv.org/pdf/2511.15440)이 설명하는 부품별 클래스 불균형·외형 상관과 대조 학습 문제의식을 참고한다. 구현은 별도 변형이며 원 논문의 실험을 재현한 결과로 표시하지 않는다. 기존128패치 마모허용 범위·CC BY4.0/원저자 인용·공개TEST 반복개발/독립현장 미검증의 한계도 유지한다.']
    with (ROOT/'reports/GEAR_SELECTION_V32_RESULTS_KO.md').open('x',encoding='utf-8',newline='\n') as out:out.write('\n'.join(lines)+'\n')
    print({'diag_VAL_inputs':11520,'new_optimizer':0,'new_TEST_scores':0,'FIT_groups_support':support,'new_training_protocol_fixed':False},flush=True)


if __name__=='__main__':main()
