"""Actual learning counts, retained prior results, and a reproducible next-step record."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.unified_main_data import read,sha
from scripts.study_unified_belt_positive import validate,BASE
REPO=ROOT.parents[1]; REPORTS=ROOT/'reports'; OUTPUTS=REPO.parents[1]/'outputs'


def pct(x): return '미측정' if x is None else f'{100*x:.2f}%'


def text_new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f: f.write(value)


def main():
    p=validate(); training=read(REPORTS/'unified-belt-positive-training.json'); r=read(REPORTS/'unified-belt-positive-validation.json')
    choice=read(REPORTS/'unified-belt-positive-selection.json'); v=read(REPORTS/'unified-belt-positive-verification.json')
    a,b=p['baseline'],r['summary']; accepted=choice['selected']=='positive_retention'
    decision='추가 학습 후보가 사전8조건을 모두 통과해 연구 연결용 모델로 선택했다.' if accepted else '추가 학습 후보가 사전 조건을 모두 통과하지 못해 기존 accepted_tiles 모델·정책·전달 ZIP을 유지했다.'
    lines=['# 벨트 손상 구역 비중·예측 보존 · 실제 추가 학습 결과','','기준일: 2026-10-13','','## 이번 학습','','**실제6epoch·768회 가중치 업데이트**를 완료했다. 배포 모델 하나의 벨트 출력만 학습했고 공통 분석부·시설·이물질 출력과 모든 BN 통계는 고정했다. 추론은 이전과 동일한 전체사진＋원본 구역6개의 결합 설정이다. 앞서 비교한 세밀 구역2후보는 위치 성능이 악화해 적용하지 않았다.','',
        f"- 작성자 TRAIN {p['scheduled_unique_TRAIN_photos']:,}장을 반복 샘플링한6,144draw이다. 새 독립 사진0장·새 정답0개다.",
        f"- 배치8개: 전체사진4개, 기존 contextual crop2개, 작성자 손상 mask가 있는 uniform crop2개. 양성 uniform 구역은 실제{p['positive_uniform_draws']:,}draw/{p['positive_uniform_unique_views']:,}고유 구역을 사용했고, 양성 구역이 준비되지 않은{p['uniform_positive_fallback_to_existing_native_crop_draws']}draw는 기존 양성 crop을 썼다.",
        '- 선택에는 TRAIN 작성자 mask만 사용했다. 검증·예약 TEST 사진을 학습에 넣지 않았다. 빈 local mask를 정상 전체 사진 정답으로 만들지 않았다.',
        '- 기존 BCE 양성 비중20＋Dice에 기존 모델과의 유효 pixel KL 보존항10을 더했다. 교사 점수는 제약용이며 새 정답이 아니다. 교사는 학습 때만 사용하고 전달 모델에는 넣지 않는다.',
        '- 학습률5e-5,6회 cosine 최저5e-6,AdamW,FP32. 마지막 epoch6만 평가했으며 중간 검증·학습률 반복 탐색은 하지 않았다.',
        f"- 실제 학습 시간{training['elapsed_training_minutes']:.2f}분; journal {v['actual_journal_updates']:,}회, 고유 완료 업데이트{v['committed_unique_updates']:,}회. 실제 학습한 파일SHA `{training['weights_sha256']}`.",'',
        '## 같은 알려진 검증353장 비교','','| 항목 | 기존 선택 모델 | 추가 학습 후보 |','|---|---:|---:|']
    for label,fn in [('위치 IoU',lambda x:x['pixel_iou']),('손상 pixel 미탐',lambda x:x['pixel_fnr']),('배경 pixel 오탐',lambda x:x['pixel_fpr']),('손상 사진 미탐',lambda x:x['photo_rates']['fnr'])]:
        lines.append('| '+label+' | '+pct(fn(a))+' | '+pct(fn(b))+' |')
    lines += [f"| 정답과 겹친 예측이 전혀 없는 손상 사진 | {a['positive_photos_without_any_gt_overlap']} | {b['positive_photos_without_any_gt_overlap']} |",'| 정상 벨트 사진 오탐 | 미측정·정상 정답0장 | 미측정·정상 정답0장 |','',
        '손상사진 미탐은 각각 '+f"{a['photo_rates']['fn']}/353과{b['photo_rates']['fn']}/353이다. 배경pixel 오탐은 손상사진 안의 작성자 mask 밖을 분모로 삼으며, 정상사진 오탐과 다르다. VAL353장·4개 촬영 구간은 이미 반복 사용한 자료다. 새 현장 검증 또는 산업현장 전체 오차율로 해석하지 않는다.",'',
        '## 작은 손상·촬영 구간 확인','','| 묶음·장수 | 기존 IoU / pixel 미탐 | 학습 후 IoU / pixel 미탐 |','|---|---:|---:|']
    for k,s in p['baseline_slices'].items():
        t=r['slices'][k]; lines.append('| '+k+f" · {s['photos']}장 | "+pct(s['pixel_iou'])+' / '+pct(s['pixel_fnr'])+' | '+pct(t['pixel_iou'])+' / '+pct(t['pixel_fnr'])+' |')
    lines += ['', '## 사전 조건과 선택','',decision,'','| 사전 조건 | 통과 여부 |','|---|---|']
    for k,passed in choice['guards'].items(): lines.append('| '+k+' | '+('통과' if passed else '실패')+' |')
    lines += ['',f"선택한 모델SHA `{choice['checkpoint_sha256']}`, 정책SHA `{choice['inference_policy']['sha256']}`. IoU0.5%p 개선과 오탐·미탐 유지, 작은 손상·각 구간 보존 조건은 학습 전에 고정했다. 실패 조건을 결과에 맞춰 완화하지 않았다.",'',
        '## 검증과 다음 우선순위','',f"기존 파일{v['protected_files']:,}개의 SHA·수정시간과 Python 소스{v['legacy_sources']}개의 SHA를 보존했다. 마지막 progress와 최종 가중치가 정확히 같고 모든 비벨트 tensor는 직전 모델과 같다. TRAIN 입력으로3기능 CPU/GPU 출력 일치와 원본 구역 분석의 CPU/GPU 일치를 확인했다.",'',
        '정상 벨트 전체사진의 crack 없음 정답과 촬영 구간을 확인할 수 있는 자료를 확보하는 것이 다음 우선순위다. 정상 전체사진 오탐을 측정할 수 없으므로, 알려진 양성 검증의 숫자만 보고 전체5% 미만이라고 판단하지 않는다. 인터넷의 normal 표기는 해당 데이터의 정답 범위를 확인한 후 사용한다. 기계 정상·모터 내부 고장 정답으로 확대하지 않는다.','',
        '예약 BeltCrack TEST는 사용하지 않았다. 시설·이물질 성능은 기존 가중치에서 계승하며 새 평가가 아니다. 위험 등급·조치·요약은 규칙 템플릿이고 점검자의 확인을 전제로 한다. 앱 기본 모델을 자동으로 교체하지 않았다.','',
        '![실제 학습 비교](unified-belt-positive-comparison.png)','',
        '- [사전 조건](unified-belt-positive-protocol.json)','- [실제 학습 기록](unified-belt-positive-training.json)','- [전체 계수·분모·각 사진](unified-belt-positive-validation.json)','- [선택 조건](unified-belt-positive-selection.json)','- [보존·재개 확인](unified-belt-positive-verification.json)',
        '- [앞선 세밀 분석 실패 결과](UNIFIED_BELT_FINE_RESULTS_KO.md)','- [출처·사용 조건](UNIFIED_MAIN_ATTRIBUTION_KO.md)','- [실행 파이프라인](../../docs/UNIFIED_BELT_POSITIVE_STUDY_KO.md)','']
    text_new(REPORTS/'UNIFIED_BELT_POSITIVE_RESULTS_KO.md','\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(16,5))
    for ax,key,title in zip(axes,['pixel_iou','pixel_fnr','pixel_fpr','photo'],['Pixel IoU (higher better)','Damage pixel FNR (lower better)','Background pixel FPR (lower better)','Damage photo FNR (lower better)']):
        values=[100*(x['photo_rates']['fnr'] if key=='photo' else x[key]) for x in (a,b)]
        bars=ax.bar(['Accepted','Trained6'],values,color=['#54738e','#519473'])
        for bar,value in zip(bars,values): ax.text(bar.get_x()+bar.get_width()/2,value,f'{value:.2f}%',ha='center',va='bottom',fontsize=9)
        ax.set_ylim(0,max(max(values)*1.22,.1)); ax.set_title(title,fontsize=11); ax.set_ylabel('%')
    fig.suptitle('ONE shared model / actual 6 epochs, 768 updates / known reused VAL353',fontsize=13)
    fig.text(.5,.02,'Normal whole-photo belt FPR unavailable; no independent field validation',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.05,1,.93]); fig.savefig(REPORTS/'unified-belt-positive-comparison.png',dpi=160); plt.close(fig)
    text_new(REPO/'safelog-cpp/docs/UNIFIED_BELT_POSITIVE_STUDY_KO.md','''# 벨트 양성 구역·예측 보존 학습 파이프라인

기존 선택 가중치에서 벨트 출력만 실제6회 학습하고 현재 선택한 전체＋원본 구역 정책으로 비교한다. 새 데이터셋이나 새 정답은 추가하지 않는다.

1. `tests/test_belt_positive_retention.py`로 교사 gradient 차단·유효 좌표 제외·동일 점수·unknown 손실을 확인한다.
2. `study_unified_belt_positive.py --mode freeze`로 TRAIN draw, 양성 구역 선택, 고정6epoch, 기존 모델, 입력·소스SHA와 사전 선택 기준을 기록한다.
3. 소스·protocol을 Git에 먼저 commit한다. 아래 학습 명령은 현재HEAD와SHA가 같아야 실행된다.
4. `--mode train`: 실제768업데이트,64업데이트마다 모델·optimizer·scheduler·cursor·CPU/CUDA RNG를 보관한다. Windows PID 확인은 읽기 전용 process_exists를 사용한다. 완료된 final은 재학습하지 않는다.
5. `--mode eval`: 알려진 VAL353에 동일한0.5 기준·작성자640 mask를 적용한다. 예약 TEST는 사용하지 않는다.
6. `--mode select-verify`: 현재 선택 정책 대비8조건과 입력·기존 모델·소스 보존, 실제 journal, progress 일치, 비벨트 tensor를 확인한다.
7. `report_unified_belt_positive.py`: 실제 개선·악화·장수·학습 횟수와 선택 여부를 기록한다. 채택된 경우에만 역할4용 전달물을 새 버전으로 만들고 독립 ZIP의 CPU API·Qt/C++ 연결을 시험한다.

```powershell
./.venv/Scripts/python.exe -m unittest discover -s tests -p test_belt_positive_retention.py
./.venv/Scripts/python.exe scripts/study_unified_belt_positive.py --mode freeze
# 소스와 protocol commit 후 실행
./.venv/Scripts/python.exe scripts/study_unified_belt_positive.py --mode train
./.venv/Scripts/python.exe scripts/study_unified_belt_positive.py --mode eval
./.venv/Scripts/python.exe scripts/study_unified_belt_positive.py --mode select-verify
./.venv/Scripts/python.exe scripts/report_unified_belt_positive.py
```

freeze·평가·보고서는 새 파일을 배타적으로 작성한다. 완료된 연구를 재실행해 기존 결과를 덮어쓰지 않는다. 동일 이름의 completed run을 새 실험으로 세지 않는다. 교사는 학습 제약용으로만 사용하며 추론은 모델 하나를 사용한다.

[실제 결과](../ai-training/reports/UNIFIED_BELT_POSITIVE_RESULTS_KO.md), [앞선 추론 비교](../ai-training/reports/UNIFIED_BELT_FINE_RESULTS_KO.md), [출처·사용 조건](../ai-training/reports/UNIFIED_MAIN_ATTRIBUTION_KO.md).
''')
    header='## 양성 구역·예측 보존 추가 학습 · 2026-10-13'
    selected_summary=b if accepted else a
    block='\n'.join([header,'','**실제6epoch·768회** 벨트 출력 추가 학습. 새 독립 사진·정답0개. '+decision,'',
        f"현재 선택 설정: 위치 IoU {pct(selected_summary['pixel_iou'])}, 손상pixel 미탐 {pct(selected_summary['pixel_fnr'])}, 배경pixel 오탐 {pct(selected_summary['pixel_fpr'])}, 손상사진 미탐 {pct(selected_summary['photo_rates']['fnr'])}. 정상사진 오탐은 미측정이고 전체 산업현장5% 검증은 없다.",''])
    for path,link in [(REPO/'README.md','safelog-cpp/ai-training/reports/UNIFIED_BELT_POSITIVE_RESULTS_KO.md'),(ROOT/'README_KO.md','reports/UNIFIED_BELT_POSITIVE_RESULTS_KO.md'),
        (REPO/'safelog-cpp/docs/EQUIPMENT_SCOPE_KO.md','../ai-training/reports/UNIFIED_BELT_POSITIVE_RESULTS_KO.md'),(REPO/'safelog-cpp/docs/DATASET_MODEL_MAP_KO.md','../ai-training/reports/UNIFIED_BELT_POSITIVE_RESULTS_KO.md')]:
        content=path.read_bytes().decode('utf-8'); assert header not in content; nl='\r\n' if '\r\n' in content else '\n'; title,body=content.split(nl+nl,1)
        addition=block+'\n- [실제 학습·비교·선택]('+link+')\n'
        path.write_bytes((title+nl+nl+addition.replace('\n',nl)+nl+body).encode('utf-8'))
    print(json.dumps({'report':'UNIFIED_BELT_POSITIVE_RESULTS_KO.md','selected':choice['selected'],'actual_updates':v['actual_journal_updates']}),flush=True)


if __name__=='__main__': main()
