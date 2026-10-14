"""Audit actual training and retain accepted model after three rejected studies."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from safelog_ai.component_discriminator import DiscriminatorMain
from safelog_ai.component_reference import NEW_CATEGORIES
from safelog_ai.retention_distillation import state_sha256
from scripts import study_component_discriminator as study
from scripts.study_equipment_expansion import read
from scripts.prepare_equipment_expansion import new
from scripts.fetch_equipment_expansion import sha

REPORTS = ROOT/'reports'; OUTPUTS = ROOT.parents[3]/'outputs'
CURRENT_ZIP = OUTPUTS/'unified-equipment-reference-role4-handoff-2026-10-14-v2.zip'
CURRENT_ZIP_SHA = '416b0845554344f9707f391ae0c232434eec602d15072ccf562e376443bf5722'


def pct(value): return f'{value*100:.2f}%'


def main():
    p = study.validate(); verification = read(REPORTS/'equipment-component-discriminator-v5-verification.json'); assert verification['status'] == 'passed'
    selections = {n: read(REPORTS/f'equipment-component-{suffix}-selection.json') for n, suffix in [(3, 'reference-v3'), (4, 'detail-v4'), (5, 'discriminator-v5')]}
    assert all(not v['newly_selected_categories'] for v in selections.values())
    assert sha(study.START) == study.EXPECTED and sha(CURRENT_ZIP) == CURRENT_ZIP_SHA
    results = {n: read(REPORTS/f'equipment-component-{suffix}-validation.json')['categories'] for n, suffix in [(3, 'reference-v3'), (4, 'detail-v4'), (5, 'discriminator-v5')]}
    training = read(study.RUN/'train/training.json'); history = [json.loads(line) for line in (study.RUN/'train/epochs.jsonl').read_text().splitlines()]
    progress = torch.load(study.RUN/'train/progress.pt', map_location='cpu', weights_only=True)
    steps = sorted({int(v['step'].item()) for v in progress['optimizer']['state'].values()}); assert steps == [2560]
    parent = torch.load(study.START, map_location='cpu', weights_only=True); trained = torch.load(study.RUN/'train/final-trained.pt', map_location='cpu', weights_only=True)
    torch.manual_seed(study.SEED); initial = DiscriminatorMain(parent['reference_counts'], list(NEW_CATEGORIES))
    initial_head = initial.component_discriminator.state_dict(); final_head = {n.removeprefix('component_discriminator.'): v for n, v in trained['state_dict'].items() if n.startswith('component_discriminator.')}
    changed = [n for n, v in initial_head.items() if not torch.equal(v, final_head[n])]; assert len(changed) == len(initial_head)
    assert all(torch.equal(v, trained['state_dict'][n]) for n, v in parent['state_dict'].items())
    draws = torch.load(study.RUN/'train/draws.pt', map_location='cpu', weights_only=True)
    sampled = {c: {'unique_normal_source_photos': len(torch.unique(indices//256)), 'unique_normal_patches': len(torch.unique(indices)), 'repeated_patch_draws': indices.numel()} for c, indices in draws.items()}
    assert sum(v['unique_normal_source_photos'] for v in sampled.values()) == 527
    audit = {'status': 'passed_rejected_candidates_retained_parent', 'audited_utc': datetime.now(timezone.utc).isoformat(),
        'active_checkpoint': str(study.START), 'active_checkpoint_sha256': study.EXPECTED, 'retained_role4_ZIP': str(CURRENT_ZIP), 'retained_role4_ZIP_sha256': CURRENT_ZIP_SHA,
        'new_role4_ZIP_created': False, 'newly_selected_categories': [], 'stage20_category_count_unchanged': 4, 'stage15_category_count_unchanged': 3,
        'actual_neural_training_epochs': 40, 'actual_neural_optimizer_updates': 2560, 'optimizer_state_step_values': steps,
        'normal_FIT_photos_sampled': 527, 'normal_patch_draws': 1966080, 'synthetic_patch_draws': 1966080, 'sampled_normal_source_counts': sampled,
        'initial_discriminator_SHA': state_sha256(initial_head), 'trained_discriminator_SHA': state_sha256(final_head), 'changed_discriminator_tensor_count': len(changed),
        'all_parent_tensors_and_reference_buffers_exact': True, 'all_original_location_outputs_preserved': True,
        'final_epoch_only_no_TEST_epoch_selection': True, 'normal_CAL_used_after_training_only': True,
        'protected_files_SHA_and_mtime_preserved': verification['protected_files'], 'legacy_Python_source_SHA_preserved': verification['legacy_sources'],
        'unit_tests_passed': 16, 'new_unit_tests': 9, 'unit_test_optimizer_updates': 0,
        'new_CPP_smoke_operations': 0, 'reason_no_new_CPP_smoke': 'No candidate selected and no runtime artifact/app adapter changed; previous ZIP21operation receipt remains',
        'new_source_photos_or_real_defect_labels': 0, 'known_TEST_reused_for_development_and_candidate_selection': True,
        'independent_field_validated': False, 'app_default_promoted': False,
        'next_priority': 'Native detail preservation plus actual defect training data; preregister fresh data split before a new training study'}
    new(REPORTS/'equipment-component-rounds-v3-v5-final-audit.json', audit)
    lines = ['# 나사·연결부 보강: 실제 추가 학습과 후보 검증', '', '기준일: 2026-10-14', '',
        '## 이번 결론', '', '**세 방식의 후보 모두 반영 조건을 충족하지 못해 기존 통과 모델을 유지했다.** 신규11유형 중 사진20% 이내4개, 그중15% 이내3개 상태가 그대로다. 역할4에게 새 모델 ZIP을 보내거나 기존 ZIP을 교체할 필요가 없다.', '',
        '정상 참조(v3), 세부 특징/전체 배치 비교(v4), 실제 학습 판별기(v5)를 각각 사전에 조건을 기록하고 검증했다. 이번에는 v5에서 **실제40epoch·2,560번 optimizer 업데이트**를 수행했다. 단순 경고 기준 변경을 신경망 학습 횟수로 세지 않았다.', '',
        '## 사용한 자료와 평가 범위', '', '| 유형 | 정상FIT | 정상CAL | 정상 평가 | 결함 평가 |', '|---|---:|---:|---:|---:|',
        '| 나사 | 96 | 24 | 41 | 119 |', '| 전선 연결부 | 330 | 54 | 119 | 193 |', '| 금속 연결부 | 101 | 25 | 30 | 14 |', '',
        '모두 앞서 확보한 공개 자료이며 새 독립 현장 사진·실제 고장 라벨을 추가한 작업은 아니다. 원래TEST 사진을 반복 사용해 개발 후보를 비교했다. 새 촬영 현장의 성능이나 물리적 부품·촬영장소가 독립된 성능이 검증된 것은 아니다.',
        '금속 연결부는 결함 평가14장으로 사전 최소20장 조건에 부족하다. 오탐·미탐이 낮게 측정돼도 목표 통과로 승격하지 않았다.', '',
        '## 모든 후보 결과', '', '| 유형 | 방식 | 정상CAL 경고 예산 | 사진 오탐 | 사진 미탐 | 판정 |', '|---|---|---:|---:|---:|---|']
    labels = {'ad_screw': '나사', 'loco_splicing_connectors': '전선 연결부', 'mpdd_connector': '금속 연결부'}
    names = {3: '정상 지역 특징', 4: '세부 특징/전체 배치', 5: '학습 판별기'}
    for category in NEW_CATEGORIES:
        b = p['baseline'][category]['photo_rates']; lines.append(f"| {labels[category]} | 현재 사용 모델 | 기존 기준 | {pct(b['fpr'])} | {pct(b['fnr'])} | 유지 |")
        for version, values in results.items():
            if category not in values: continue
            for name in ('normalCAL10', 'normalCAL05'):
                v = values[category][name]; r = v['photo_rates']; decision = '평가 표본 부족' if category == 'mpdd_connector' else '오탐·미탐 조건 미달'
                lines.append(f"| {labels[category]} | {names[version]} | {int(v['photo_normalCAL_alpha']*100)}% | {r['fp']}/{v['normal_test_photos']} = {pct(r['fpr'])} | {r['fn']}/{v['defect_test_photos']} = {pct(r['fnr'])} | {decision} |")
    lines += ['', 'CAL 예산은 정상 보정 자료의 순위 기준이며 실제 오탐률 보장치가 아니다. 모든 후보에서alpha.10/.05 두 기준을 평가 전에 고정했고, TEST 점수의 분위수로 임계값을 고르지 않았다. 후보 선택과 다음 회차 설계에는 이미 알려진 평가 결과가 사용되므로 이 표는 개발 비교 결과다.', '',
        '## 실제 신경망 학습', '',
        '- 기존 공통 MobileNet 분석부·시설/벨트/이물질/설비 출력·정상 케이블/너트 참조를 고정했다. 같은 모델에 유형 조건을 받는 판별기 하나를 추가했다.',
        '- 정상FIT527장의134,912개 특징만 학습 자료로 사용했다. 매 업데이트마다 유형별256개, 합768개 정상 지역 특징을 뽑았다.',
        '- 정상 특징은label0, 작은 Gaussian 변형을 주고 다시 정규화한 특징은label1로 학습했다. 가상 이상은 실제 고장 정답이 아니다.',
        '- 정상 지역 특징1,966,080회와 같은 수의 가상 이상을 반복 학습했다. 반복draw를 새로운 사진으로 세지 않았다.',
        '- 공유 판별기는조건embedding과1016→256→128→1 MLP다. AdamW·BCE·고정40epoch로 학습하고 마지막 회차만 평가했다. TEST·CAL로 학습하거나 TEST로 epoch를 선택하지 않았다.',
        f"- 첫epoch 평균학습BCE {history[0]['training_BCE']:.4f} → 마지막 {history[-1]['training_BCE']:.4f}. 이는 정상/가상 변형 학습 손실이며 실제 고장 정확도가 아니다.",
        f"- optimizer state의step이모두2,560이고, 초기 판별기와 비교해{len(changed)}개tensor가모두변했다. 기존부모모델tensor와케이블/너트참조는정확히같다.", '',
        '[SimpleNet 논문](https://arxiv.org/abs/2303.15140)의 정상 특징·Gaussian 가상 이상·판별기 아이디어를 참고했다. 논문의feature adapter·backbone·구조·학습 설정·성능을 재현한 구현은 아니다. [저자 코드](https://github.com/DonaldRR/SimpleNet)는 방식 확인용으로 읽었고 다운로드해 실행하지 않았다.', '',
        '## 결함 종류별로 남은 문제', '', '| 전선 연결부 후보(CAL10) | 구성·관계 이상 미탐 | 구조·외관 이상 미탐 |', '|---|---:|---:|']
    for version, values in results.items():
        groups = values['loco_splicing_connectors']['normalCAL10']['by_defect']
        lines.append('| '+names[version]+' | '+' | '.join(f"{groups[k]['fn']}/{groups[k]['defect_photos']} = {pct(groups[k]['fnr'])}" for k in ('logical_anomalies', 'structural_anomalies'))+' |')
    lines += ['', '지역 특징 비교와 전체 배치 비교는 서로 다른 결함에서 실패했다. 전체 배치 비교만으로 세밀한 외관 결함까지 해결되지 않았다. 가상 이상을 잘 구분하는 판별기도 실제 손상 전체에 충분히 일반화하지 못했다.',
        '나사는 원본 해상도 보존과 실제 손상 학습자료를 다음 검증 대상으로 잡는다. 이번 결과만으로 원인을 해상도 하나로 확정하지 않는다.', '',
        '## 다음 개발 순서', '',
        '1. 원본 세부 정보를 유지하는 확대·타일 입력을 준비한다. 평가 정답을 보고 타일 위치를 고르지 않는다.',
        '2. 실제 손상 사진이 학습에 포함되는 추가 자료를 확보하고, 중복·촬영그룹을 확인해 학습/보정/평가를 분리한다. 현재TEST 정답을 학습용으로 옮기지 않는다.',
        '3. 새 판별기에 실제 손상 신호를 학습시키는 실험을 새 사전 조건으로 시작한다. 현재 통과한 네 유형의 결과를 함께 보존·검증한다.',
        '4. 금속 연결부는 특히 새로운 결함 평가 표본을 확보한다. 표본을 확보하기 전까지평가 부족 상태를 유지한다.', '',
        '## 기존 앱 연결과 보존', '',
        '현재 역할4 전달물은기존케이블·너트ZIP이다. 이번회차는새전달ZIP·APK·기본앱모델을만들거나교체하지않았다. 새후보C++연결검사는0회이며, 이전ZIP의21개성공기록을이번회차검사로세지않았다.',
        f"자료·모델·기존보고서{verification['protected_files']:,}개의SHA와mtime,기존Python소스{verification['legacy_sources']}개의SHA를보존했다. 후보의기존위치heatmap과pixel기준도같다. 사진판별실험을위치정확도개선으로표시하지않았다.",
        '각runs의selected/final.pt는후보선택과정의메타데이터사본이며,이번처럼선택항목이없으면활성모델이아니다. 아래활성SHA를기준으로이어간다.', '',
        '현재 활성 연구 모델SHA: `'+study.EXPECTED+'`', '현재 역할4 ZIP SHA: `'+CURRENT_ZIP_SHA+'`', '',
        '![후보 비교](equipment-component-rounds-v3-v5-comparison.png)', '',
        '- [v3 정상 참조 조건](equipment-component-reference-v3-protocol.json)', '- [v3 전체 결과](equipment-component-reference-v3-validation.json)',
        '- [v4 세부/배치 조건](equipment-component-detail-v4-protocol.json)', '- [v4 전체 결과](equipment-component-detail-v4-validation.json)',
        '- [v5 실제 학습 조건](equipment-component-discriminator-v5-protocol.json)', '- [v5 전체 결과](equipment-component-discriminator-v5-validation.json)',
        '- [v5 보존 검증](equipment-component-discriminator-v5-verification.json)', '- [최종 감사](equipment-component-rounds-v3-v5-final-audit.json)',
        '- [현재 역할4 연결 안내](../../docs/EQUIPMENT_REFERENCE_HANDOFF_KO.md)', '- [자료 출처·연구 사용 조건](EQUIPMENT_EXPANSION_ATTRIBUTION_KO.md)', '']
    with (REPORTS/'COMPONENT_DEVELOPMENT_RESULTS_KO.md').open('x', encoding='utf-8', newline='\n') as out: out.write('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.3))
    for j, (category, label) in enumerate(zip(NEW_CATEGORIES, ('Screw', 'Splicing connectors', 'Metal connector (defect TEST14)'))):
        values = [('Prior', p['baseline'][category]['photo_rates'])]+[(f'v{n} CAL10', v[category]['normalCAL10']['photo_rates']) for n, v in results.items() if category in v]
        for i, key in enumerate(('fpr', 'fnr')):
            ax = axes[i, j]; bars = ax.bar([v[0] for v in values], [100*v[1][key] for v in values], color=['#71808e', '#7581af', '#b89b59', '#458575'][:len(values)])
            ax.axhline(20, color='#b34545', linestyle='--', linewidth=1); ax.set_ylim(0, 105); ax.set_title(label if i == 0 else ''); ax.set_ylabel('Photo '+key.upper()+' (%)'); ax.tick_params(axis='x', labelrotation=15)
            for bar in bars: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.2, f'{bar.get_height():.1f}', ha='center', fontsize=8)
    fig.suptitle('Component studies: no candidate selected; accepted parent retained')
    fig.text(.5, .018, 'Known TEST reused for development. CAL05 results are in the report. Actual training: v5, 40 epochs / 2,560 updates.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .05, 1, .95]); fig.savefig(REPORTS/'equipment-component-rounds-v3-v5-comparison.png', dpi=160); plt.close(fig)
    resume = {**audit, 'repo': str(ROOT.parents[1]), 'completed_studies': ['equipment-component-reference-v3', 'equipment-component-detail-v4', 'equipment-component-discriminator-v5'],
        'active_checkpoint_is_unchanged_parent': True, 'do_not_use_rejected_selected_folder_metadata_copies_as_active_model': True,
        'result_report': str(REPORTS/'COMPONENT_DEVELOPMENT_RESULTS_KO.md'), 'next': audit['next_priority']}
    new(OUTPUTS/'COMPONENT_DEVELOPMENT_RESUME_2026-10-14.json', resume)
    with (OUTPUTS/'COMPONENT_DEVELOPMENT_RESUME_2026-10-14.md').open('x', encoding='utf-8', newline='\n') as out:
        out.write('# 나사·연결부 실험 재개 기록\n\n3개 방식은 모두 미채택. 기존활성모델·케이블/너트ZIP유지.\n'
            'v5실제학습40epoch·2560업데이트,정상FIT527장,가상변형은실제고장라벨이아님.\n'
            '새자료0,알려진TEST재사용,금속연결부결함TEST14장으로평가부족.\n'
            '현재모델SHA: '+study.EXPECTED+'\n다음: 원본세부정보보존·실제손상학습자료확보후새사전조건.\n')
    repo = ROOT.parents[1]; header = '## 나사·연결부 실제 추가 학습 검증 · 2026-10-14'
    block = '\n'.join([header, '', '정상 참조·세부/배치 비교·새 판별기를 비교했다. 판별기는 실제40epoch·2,560회 학습했지만 후보가 오탐·미탐20%와 표본 조건을 함께 충족하지 못해 기존 통과 모델·전달ZIP을 유지했다.',
        '신규11유형 중4개 사진20% 이내 상태가 유지된다. 정상FIT527장을 사용했으며 새 독립 현장 사진·실제 고장 라벨은 추가하지 않았다. 다음은 원본 세부 정보 보존과 실제 손상 학습자료 보강이다.', ''])
    for path, link in [(repo/'README.md', 'safelog-cpp/ai-training/reports/COMPONENT_DEVELOPMENT_RESULTS_KO.md'), (ROOT/'README_KO.md', 'reports/COMPONENT_DEVELOPMENT_RESULTS_KO.md'),
        (repo/'safelog-cpp/docs/EQUIPMENT_SCOPE_KO.md', '../ai-training/reports/COMPONENT_DEVELOPMENT_RESULTS_KO.md'), (repo/'safelog-cpp/docs/DATASET_MODEL_MAP_KO.md', '../ai-training/reports/COMPONENT_DEVELOPMENT_RESULTS_KO.md')]:
        text = path.read_bytes().decode('utf-8'); assert header not in text; nl = '\r\n' if '\r\n' in text else '\n'; title, body = text.split(nl+nl, 1)
        insert = block+'\n- [이번 실제 학습과 미채택 결과]('+link+')\n'; path.write_bytes((title+nl+nl+insert.replace('\n', nl)+nl+body).encode('utf-8'))
    print(json.dumps({'audit': 'passed', 'active_model_retained': True, 'actual_gradient_updates': 2560, 'discriminator_tensors_changed': len(changed), 'new_ZIP': False}), flush=True)


if __name__ == '__main__': torch.set_num_threads(4); main()
