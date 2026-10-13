"""Report the fixed, matched per-photo protection study from measured artifacts."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.unified_main_data import read, sha
from scripts.study_unified_belt_photo_guard import BASE, validate, new
REPO = ROOT.parents[1]; REPORTS = ROOT/'reports'


def text_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        f.write(value)


def pct(value):
    return '미측정' if value is None else f'{100*value:.2f}%'


def main():
    p = validate()
    t = read(REPORTS/'unified-belt-photo-guard-v1-training.json')
    r = read(REPORTS/'unified-belt-photo-guard-v1-validation.json')
    c = read(REPORTS/'unified-belt-photo-guard-v1-selection.json')
    v = read(REPORTS/'unified-belt-photo-guard-v1-verification.json')
    old = read(REPORTS/'unified-belt-recall-step-v2-validation.json')
    original = read(REPORTS/'unified-belt-fine-global_tile_mean-validation.json')
    assert t['optimizer_updates'] == v['unique_committed_updates'] == 512
    assert sha(BASE/'train/final.pt') == t['weights_sha256']
    cases = [original['summary'], old['known_summary'], r['known_summary']]
    roi = [r['normal_ROI_CAL']['accepted'], old['normal_ROI_CAL']['recall_projected'], r['normal_ROI_CAL']['photo_guard']]
    before = {x['id']: x for x in original['rows']}
    after = {x['id']: x for x in r['known_belt_rows']}
    assert set(before) == set(after) and len(before) == 353
    missed_before = {k for k, x in before.items() if x['score'] <= .5}
    missed_after = {k for k, x in after.items() if x['score'] <= .5}
    assert len(missed_before) == cases[0]['photo_rates']['fn'] and len(missed_after) == cases[-1]['photo_rates']['fn']
    changes = {'threshold': .5, 'definition': 'score>.5 is an alarm; fused valid-pixel max, uncalibrated',
               'recovered_prior_misses': sorted(missed_before-missed_after),
               'new_misses': sorted(missed_after-missed_before),
               'persistent_misses': sorted(missed_before & missed_after),
               'rows': [{'id': k, 'accepted_score': before[k]['score'], 'candidate_score': after[k]['score']}
                        for k in sorted(missed_before | missed_after)]}
    new(REPORTS/'unified-belt-photo-guard-v1-photo-changes.json', changes)
    decision = ('기존 선택 모델과 역할4 전달 ZIP을 유지한다.' if c['selected'] == 'prior_accepted_tiles'
                else '새 연구 후보가 사전 조건을 통과했다. 연결용 전달물을 별도로 검증한다. 앱 기본 모델은 자동 교체하지 않는다.')
    d = p['train_mask_diagnostic']
    lines = ['# 사진별 손상 탐지 보호 · 실제 학습 비교', '', '기준일: 2026-10-13', '',
             '## 이번 변경', '',
             '정상 영역의 오탐을 줄이면서 손상을 놓치지 않도록, 기존의 배치 전체 양성 pixel 합산 보호를 **손상 사진별 보호 6개**로 바꿨다. 같은 공통 모델·같은 손상 분석부를 학습하며 시설·이물질 분석부는 보존한다.', '',
             f"학습 전 TRAIN 정답만 확인한 결과, 배치 내 가장 큰/작은 손상 pixel 수의 중앙값 차이는 {d['GT_pixels_largest_to_smallest_ratio_quantiles']['median']:.2f}배다. 가장 작은 사진의 합산 양성 pixel 비중 중앙값은 {100*d['smallest_photo_share_of_pooled_GT_quantiles']['median']:.2f}%다. 큰 손상이 기존 보호 계산을 지배할 가능성을 조사한 것으로, 이전 성능 저하의 원인을 확정한 결과는 아니다.", '',
             '사진마다 양성 pixel 손실을 자기 손상 크기로 나눈 뒤, 실제 Adam 가중치 변화가 6개 중 어느 사진의 손실도 일차 근사로 늘리지 않도록 보정했다. float64 계산 후 실제 저장 dtype에서 모든 조건을 재검사한다. 64회 이내 계산에서 조건을 만족하지 못하면 해당 가중치 변화 전체를 정확히 복원한다. optimizer의 원래 momentum은 유지한다.', '',
             '이 조건은 **현재 TRAIN 배치의 일차 근사**다. 실제 손실, 검증 사진 미탐, 산업현장 성능을 보장하지 않으므로 검증을 별도로 수행했다.', '',
             '## 실제 학습 조건', '',
             f"- 실제 4epoch·512 optimizer 업데이트. 실제 journal {v['actual_journal_updates']}개, 각 업데이트의 사진별 조건 총 {v['photo_constraints_checked']:,}개를 확인했다.",
             f"- 보정 {v['projected_updates']}회, 정확한 이전 파라미터 복원 {v['reverted_parameter_steps']}회. 학습 시간 {t['elapsed_training_minutes']:.2f}분.",
             '- 초기 모델은 현재 선택 native_mix이며, 앞선 미채택 모델에서 이어 학습하지 않았다. 이전 정상 학습/합산 보호 실험과 동일한 배열·seed443·학습률·손실·4epoch로 비교했다.',
             '- 배치8: 전체 손상 사진4개·기존 contextual crop2개·작성자 정상 ROI2개. 정상 FIT14개를1,024회 사용하고 양성 TRAIN3,072draw를 사용했다. 반복 사용을 새 사진으로 계산하지 않는다.',
             '- BCE posweight20＋Dice＋teacher KL10＋정상 ROI 손실0.5 유지. 이번에 새 사진·새 정답을 추가하지 않았다. 정상 ROI를 정상 전체사진이나 정상 dense pixel mask로 취급하지 않는다.',
             '- 학습 전 protocol을 commit하고 마지막 epoch만 비교했다. 검증 결과를 보고 epoch·임계값·선택 조건을 바꾸지 않았다.', '',
             '## 같은 알려진 손상VAL353장 비교', '',
             '| 항목 | 현재 선택 | 앞선 합산 보호 | 이번 사진별 보호 |', '|---|---:|---:|---:|']
    for label, fn in [('위치 IoU', lambda x: x['pixel_iou']), ('손상 pixel 미탐', lambda x: x['pixel_fnr']),
                      ('배경 pixel 오탐', lambda x: x['pixel_fpr']), ('손상 사진 미탐', lambda x: x['photo_rates']['fnr'])]:
        lines.append('| '+label+' | '+' | '.join(pct(fn(x)) for x in cases)+' |')
    lines += ['| 놓친 손상 사진 | '+' | '.join(str(x['photo_rates']['fn'])+'/353' for x in cases)+' |',
              '| 정답과 겹친 예측 없음 | '+' | '.join(str(x['positive_photos_without_any_gt_overlap']) for x in cases)+' |', '',
              f"현재 선택 대비 기존 미탐 회복 {len(changes['recovered_prior_misses'])}장, 새 미탐 {len(changes['new_misses'])}장, 지속 미탐 {len(changes['persistent_misses'])}장이다. 같은 미탐 장수라도 놓친 사진이 바뀔 수 있어 사진 ID와 점수를 별도 보관한다.", '',
              '배경pixel 오탐의 분모는 손상사진 안의 정답 mask 밖이다. 정상 전체사진 오탐률과 다르다. 기존에 사용한353장·4개 촬영 구간을 반복 비교했으므로 새로운 독립 산업현장 시험으로 해석하지 않는다. 원본 전체＋기존6구역 결합 정책과0.5 기준을 그대로 사용했다.', '',
              '### 작은 손상과 촬영 구간', '',
              '| 구분 | 현재 손상pixel 미탐 | 이번 손상pixel 미탐 | 현재 IoU | 이번 IoU |', '|---|---:|---:|---:|---:|']
    for key in original['slices']:
        a, b = original['slices'][key], r['known_slices'][key]
        lines.append('| '+key+' | '+pct(a['pixel_fnr'])+' | '+pct(b['pixel_fnr'])+' | '+pct(a['pixel_iou'])+' | '+pct(b['pixel_iou'])+' |')
    lines += ['', '## 정상 ROI 비교 · CAL3개', '', '| 항목 | 현재 선택 | 앞선 합산 보호 | 이번 사진별 보호 |', '|---|---:|---:|---:|',
              '| 이상 후보 발생 | '+' | '.join(str(x['false_alarms'])+'/3' for x in roi)+' |',
              '| 정상 ROI 오탐 비율 | '+' | '.join(pct(x['ROI_FPR']) for x in roi)+' |',
              '| 정상 전체사진 오탐 | 미측정 | 미측정 | 미측정 |', '',
              'CAL3은 작성자가 normal belt로 지정한 bbox 내부 구역이고 전체사진·기계적 정상 정답은 없다. 보관 정상7개와 예약BeltCrack TEST의 모델 점수를 계산하지 않았다. 이3개에서0건이어도 산업현장 오탐0% 또는5% 미만의 근거는 아니다.', '',
              '## 사전 선택 조건과 결론', '', decision, '', '| 조건 | 결과 |', '|---|---|']
    for name, passed in c['known_metric_guards'].items():
        lines.append('| '+name+' | '+('통과' if passed else '실패')+' |')
    lines += ['| 정상CAL 이상 후보 감소 | '+('통과' if c['normal_ROI_CAL_alarm_reduction'] else '실패')+' |', '',
              '현재 선택 대비 IoU0.5%p 이상 개선과 오탐·미탐·작은 손상·각 촬영 구간의 보존 조건8개 및 정상CAL 후보 감소를 모두 요구했다. 일부 지표만 좋아진 가중치는 반영하지 않는다.', '',
              f"- 기존 파일 {v['protected_files']:,}개의 SHA·mtime와 기존Python소스 {v['legacy_sources']}개의 SHA를 유지했다. progress/final 일치와 모든 비벨트tensor 정확 일치를 확인했다.",
              '- 시설·이물질 CPU 출력은 이전과 정확히 같고,3기능과 원본 구역의 CPU/GPU 수치 일치를 확인했다. 이 실험에서 새로운 C++ HTTP 연결 검사는 수행하지 않았다.',
              '- 사진에 보이는 벨트 손상 분석을 보강하는 연구다. 내부 모터 고장·가동 상태·수리 완료·법적 안전 판정을 학습한 것으로 표현하지 않는다.',
              '- 모델 하나·공통 분석부 하나를 사용한다. 교사는 TRAIN 제약용이며 추론에 함께 배포하지 않는다.', '',
              '선택 모델SHA: `'+c['checkpoint_sha256']+'`', '정책SHA: `'+c['policy']['sha256']+'`',
              '이번 학습 모델SHA: `'+t['weights_sha256']+'`', '',
              '![측정 결과](unified-belt-photo-guard-v1-comparison.png)', '',
              '- [사전 조건](unified-belt-photo-guard-v1-protocol.json)', '- [실제 학습](unified-belt-photo-guard-v1-training.json)',
              '- [사진별 검증](unified-belt-photo-guard-v1-validation.json)', '- [미탐 변화](unified-belt-photo-guard-v1-photo-changes.json)',
              '- [선택](unified-belt-photo-guard-v1-selection.json)', '- [보존·수치 검증](unified-belt-photo-guard-v1-verification.json)',
              '- [앞선 합산 보호 결과](UNIFIED_BELT_RECALL_STEP_RESULTS_KO.md)', '- [정상 자료 검수](UNIFIED_BELT_NORMAL_RESULTS_KO.md)',
              '- [출처·라이선스](UNIFIED_MAIN_ATTRIBUTION_KO.md)', '- [실행 순서](../../docs/UNIFIED_BELT_PHOTO_GUARD_KO.md)', '']
    text_new(REPORTS/'UNIFIED_BELT_PHOTO_GUARD_RESULTS_KO.md', '\n'.join(lines))
    text_new(REPO/'safelog-cpp/docs/UNIFIED_BELT_PHOTO_GUARD_KO.md', '''# 사진별 손상 탐지 보호 비교

같은6개 양성 TRAIN 사진을 하나로 합산하던 보호 계산을 사진별6개 조건으로 분리한다. 매 조건은 해당 사진의 작성자 양성 pixel 수로 정규화한다. 실제 Adam 변화·float64 dual 좌표 계산·FP32 쓰기 후 조건 검사를 수행한다. 64cycle 안에 만족하지 못하면 정확히 이전 파라미터를 복원하고 raw optimizer moment는 유지한다. 이 조건은 TRAIN 일차 근사이며 검증 미탐을 보장하지 않는다.

## 완료된 실행 순서

1. test_belt_photo_guard.py: 작은/큰mask 정규화, 유효pixel, 다중조건·직교방향, 안전변화 보존, 반대조건 복원, Adam moment, 고차원FP32의7검사.
2. study_unified_belt_photo_guard.py --mode freeze: 기존control과같은draw·입력SHA·기존파일SHA/mtime·TRAIN정답크기 진단·사전8조건을 고정. 소스/protocol을commit한다.
3. --mode train:4epoch512optimizer업데이트.64단계마다 가중치·optimizer·scheduler·cursor·CPU/CUDA RNG저장. WindowsPID는 읽기전용process_exists사용. 완료final은 재학습하지 않는다.
4. --mode eval: knownVAL353와normalCAL3만 평가. 보관7·예약BeltCrack TEST 점수는 계산하지 않는다.
5. --mode verify:512journal의6개조건총3072개, final/progress, 비벨트tensor, 기존파일·소스 보존과 사전선택조건을 확인한다.
6. report_unified_belt_photo_guard.py: 전체/작은손상/촬영구간과 사진미탐ID변화를 보고한다. 선택된새후보가있으면 별도연결전달물검증을 진행한다. 앱기본모델을 자동교체하지 않는다.

```powershell
./.venv/Scripts/python.exe -m unittest discover -s tests -p test_belt_photo_guard.py
./.venv/Scripts/python.exe scripts/study_unified_belt_photo_guard.py --mode freeze
# 소스/protocol commit 후
./.venv/Scripts/python.exe scripts/study_unified_belt_photo_guard.py --mode train
./.venv/Scripts/python.exe scripts/study_unified_belt_photo_guard.py --mode eval
./.venv/Scripts/python.exe scripts/study_unified_belt_photo_guard.py --mode verify
./.venv/Scripts/python.exe scripts/report_unified_belt_photo_guard.py
```

완료된 freeze/eval/report를 다시 실행해 덮어쓰지 않는다. 새 학습은 새 run·사전 protocol·학습 전 commit을 사용한다. 사진 전체 정상 FPR은 미측정이다. 원자료를 Git/전달 ZIP에 추가하지 않는다.

[실제 결과](../ai-training/reports/UNIFIED_BELT_PHOTO_GUARD_RESULTS_KO.md)
''')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    names = ['Accepted', 'Pooled guard', 'Photo guards']
    values = [[x['pixel_iou'] for x in cases], [x['pixel_fnr'] for x in cases],
              [x['photo_rates']['fnr'] for x in cases], [x['ROI_FPR'] for x in roi]]
    for ax, nums, title in zip(axes, values, ['Pixel IoU (higher better)', 'Damage pixel FNR', 'Damage photo FNR', 'Normal ROI FPR (CAL3 only)']):
        nums = [100*x for x in nums]
        bars = ax.bar(names, nums, color=['#54738e', '#caa34b', '#519473'])
        for bar, value in zip(bars, nums):
            ax.text(bar.get_x()+bar.get_width()/2, value, f'{value:.2f}%', ha='center', va='bottom', fontsize=8)
        ax.set_title(title, fontsize=11); ax.set_ylim(0, max(max(nums)*1.22, .1)); ax.set_ylabel('%')
        ax.tick_params(axis='x', labelsize=8)
    fig.suptitle('Matched 4 epochs /512 updates: pooled versus per-photo TRAIN displacement protection')
    fig.text(.5, .02, 'Known VAL353 reused; normal CAL3 is ROI-only, not whole-photo FPR or field5% proof', ha='center')
    fig.tight_layout(rect=[0, .05, 1, .93]); fig.savefig(REPORTS/'unified-belt-photo-guard-v1-comparison.png', dpi=160); plt.close(fig)
    header = '## 사진별 손상 보호 비교 · 2026-10-13'
    block = '\n'.join([header, '',
        f"실제 4epoch·512회 학습에서 배치 내 손상 사진 6개를 각각 보호했다. TRAIN 일차 근사 조건을 총 {v['photo_constraints_checked']}개 검사했다. 검증 미탐 보장 조건은 아니므로 같은 VAL353장과 정상 CAL3개로 비교했다.", '',
        f"현재/이번 손상 사진 미탐 {cases[0]['photo_rates']['fn']}/{cases[-1]['photo_rates']['fn']}장(각353장), 정상 ROI 후보 {roi[0]['false_alarms']}/{roi[-1]['false_alarms']}개(각3개). "+decision, '',
        '정상 전체사진 오탐과 독립 산업현장 전체5%는 미검증이다. 기존 시설·이물질 tensor를 보존했다. 보관7개·예약TEST의 모델 점수를 계산하지 않았다.', ''])
    for path, link in [(REPO/'README.md', 'safelog-cpp/ai-training/reports/UNIFIED_BELT_PHOTO_GUARD_RESULTS_KO.md'),
                       (ROOT/'README_KO.md', 'reports/UNIFIED_BELT_PHOTO_GUARD_RESULTS_KO.md'),
                       (REPO/'safelog-cpp/docs/EQUIPMENT_SCOPE_KO.md', '../ai-training/reports/UNIFIED_BELT_PHOTO_GUARD_RESULTS_KO.md'),
                       (REPO/'safelog-cpp/docs/DATASET_MODEL_MAP_KO.md', '../ai-training/reports/UNIFIED_BELT_PHOTO_GUARD_RESULTS_KO.md')]:
        content = path.read_bytes().decode('utf-8'); assert header not in content
        nl = '\r\n' if '\r\n' in content else '\n'; title, body = content.split(nl+nl, 1)
        addition = block+'\n- [사진별 보호 측정 결과]('+link+')\n'
        path.write_bytes((title+nl+nl+addition.replace('\n', nl)+nl+body).encode('utf-8'))
    print(json.dumps({'report': 'UNIFIED_BELT_PHOTO_GUARD_RESULTS_KO.md', 'selection': c['selected'], 'updates': 512}))


if __name__ == '__main__':
    main()
