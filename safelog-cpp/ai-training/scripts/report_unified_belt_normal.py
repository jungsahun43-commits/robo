"""Publish actual normal-ROI pilot evidence, without a whole-photo FPR claim."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.unified_main_data import read,sha
from scripts.study_unified_belt_normal import validate,BASE,NORMAL,new,START_SHA
from scripts.release_unified_main import local_links
REPO=ROOT.parents[1]; REPORTS=ROOT/'reports'; OUTPUTS=REPO.parents[1]/'outputs'
ZIP_NAME='unified-main-belt-tiles-role4-handoff-2026-10-13-v1.1.zip'
ZIP_SHA='3e8235e8d3cc626bf0476a05a6f8686872fd90d34ed517c6d081e707cfdbd7c9'


def pct(x): return '미측정' if x is None else f'{100*x:.2f}%'


def text_new(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f: f.write(text)


def main():
    p=validate(); n=read(NORMAL); t=read(REPORTS/'unified-belt-normal-training.json'); r=read(REPORTS/'unified-belt-normal-validation.json')
    v=read(REPORTS/'unified-belt-normal-verification.json'); choice=read(REPORTS/'unified-belt-normal-selection.json')
    assert choice['selected']=='prior_accepted_tiles' and t['optimizer_updates']==v['actual_journal_updates']==512
    assert sha(BASE/'train/final.pt')==t['weights_sha256']
    source_summary={k:value for k,value in n.items() if k not in ('rows','preparation_source_sha256')}
    source_summary['source_images_redistributed_in_Git']=False
    new(REPORTS/'unified-belt-normal-source-summary.json',source_summary)
    a,b=p['baseline'],r['known_summary']; old_roi,new_roi=r['normal_ROI_CAL']['accepted'],r['normal_ROI_CAL']['normal_pilot']
    lines=['# 정상 벨트 구역 추가 자료 · 초기 학습 결과','','기준일: 2026-10-13','','## 실제로 완료한 작업','',
        '**공개 데이터를 새로 내려받아 검사하고, 기존 통합 모델의 벨트 출력에 실제4epoch·512회 추가 학습했다.** 학습은 모델 하나의 벨트 출력만 바꾼다. 학습용 교사는 예측 보존 제약에만 사용하며 추론에 포함하지 않는다. 공통 분석부·시설·이물질 출력과 BN 통계는 보존했다.','',
        f"- 실제 학습 시간{t['elapsed_training_minutes']:.2f}분. journal {v['actual_journal_updates']}회, 고유 완료 업데이트{v['unique_committed_updates']}회.",
        f"- 기존 작성자 양성 TRAIN {p['positive_unique_TRAIN_photos']:,}장에3,072draw, 새 작성자 정상 구역14개에1,024draw를 사용했다. 반복 사용 횟수는 새 사진 수가 아니다.",
        '- 새 정상 전체사진 정답0장·새 정상 pixel mask0개다. 정상 bbox 안의 구역 부재 분류 정답을 사용했으며, 사진 전체나 내부 기계 정상으로 확대하지 않았다.',
        f"- 실제 학습 파일SHA `{t['weights_sha256']}`. 이 파일은 비교용 pilot으로 보관하고 현재 선택 모델은 그대로 유지한다.",'',
        '## 출처·자료량·중복 검사','',
        '[Gofaones Workspace의 belt defects v2](https://universe.roboflow.com/gofaones-workspace/belt-defects/dataset/2), [v1](https://universe.roboflow.com/gofaones-workspace/belt-defects/dataset/1). 사용 조건은 작성자가 표시한[CC BY4.0](https://creativecommons.org/licenses/by/4.0/)이고 ZIP 안 README.dataset.txt에서도 확인했다. 원래 촬영 기관·장비·촬영 구간 정보는 제공되지 않아 독립 현장 여부를 검증하지 못했다.','',
        '- v1 ZIP497장·19,280,690bytes, v2 ZIP1,548장·56,537,347bytes를 실제 수신했다. 증강본을 포함한 파일 수다. v2를 학습 재료로 쓰고 v1은 삭제된 worn surface 라벨 등의 검수에만 사용했다.',
        f"- v2 원본 이름 힌트{n['source_parent_hints']}개 중{n['publisher_cross_split_parent_hints']}개가 작성자 학습·검증·시험 묶음에 걸쳐 있었다. 동일 이름·정확 RGB·회전/반전 pHash 근접성을 합쳐{n['dedup_similarity_groups']}개 보수적 유사 그룹으로 묶었다. 이름 힌트 수와 그룹 수를 독립 사진·독립 촬영 횟수로 부르지 않는다.",
        '- 정상 bbox의 중앙60% 정사각형을 사용하고 최소64pixel을 요구했다. 어떤 다른 라벨 bbox와도 겹치지 않아야 한다. v1의 마모 라벨이 있는 부모 이름은 제외했고, 제안된 묶음을 가로지르는 유사 그룹도 제외했다. 이전 BeltCrack TRAIN/알려진 VAL과 회전·반전 pHash를 비교했다.',
        '- 남은 정상 구역24개: FIT14·CAL3·보관7. 한 유사 그룹에서 하나만 골랐다. 보관7개는 원본 검수로 조회했지만 가중치 학습·모델 점수 비교에는 쓰지 않았다.',
        '- FIT는 작성자TRAIN의 frame0..149, CAL은frame180..239, 보관은frame270이상 이름 힌트를 사용하고 사이 구간은 제외했다. 이름 숫자가 실제 촬영 순서·영상 ID라는 보장은 없다. 작성자VAL/TEST 원본을 FIT에 넣지 않았다. 작성자 원래 split을 그대로 독립 평가로 믿지 않고 제안된 구간으로 재분리했다.',
        '- 자료는512×512로 변환되고 회전·crop·노출·noise 증강을 포함한다. 고해상도 카메라 원본으로 설명하지 않는다. 작성자 정상 라벨도 현장 점검자의 인증이나 완전한 기계 정상 정답이 아니다.','',
        '## 정상 구역 오탐 · 비교용3개만','',
        '| 측정 | 기존 선택 모델 | 정상 구역 추가 학습 후보 |','|---|---:|---:|',
        f"| 정상 구역의 이상 후보 발생 | {old_roi['false_alarms']}/3 | {new_roi['false_alarms']}/3 |",
        f"| 구역 오탐 비율 | {pct(old_roi['ROI_FPR'])} | {pct(new_roi['ROI_FPR'])} |",
        '| 정상 전체사진 오탐 | 미측정 | 미측정 |','',
        '**3개 구역의 비율은 초기 동작 확인 수치다. 전체 산업현장 오탐·미탐5% 미만이나 일반적인 정상 벨트 사진 오탐으로 해석하지 않는다.** 정상 구역도 축소된 bbox에서 나온 부분사진이다. CAL은 이전 미측정 항목을 작은 범위로 처음 비교한 것이며 독립 현장 시험이 아니다.','',
        '## 기존 손상 검증353장 · 같은 구역 정책','',
        '| 항목 | 기존 선택 모델 | 정상 구역 추가 학습 후보 |','|---|---:|---:|']
    for label,fn in [('위치 IoU',lambda x:x['pixel_iou']),('손상 pixel 미탐',lambda x:x['pixel_fnr']),('배경 pixel 오탐',lambda x:x['pixel_fpr']),('손상 사진 미탐',lambda x:x['photo_rates']['fnr'])]:
        lines.append('| '+label+' | '+pct(fn(a))+' | '+pct(fn(b))+' |')
    lines += [f"| 놓친 손상 사진 | {a['photo_rates']['fn']}/353 | {b['photo_rates']['fn']}/353 |",
        f"| 정답에 겹친 예측이 전혀 없는 손상 사진 | {a['positive_photos_without_any_gt_overlap']} | {b['positive_photos_without_any_gt_overlap']} |",'',
        '353장·4개 작성자 구간은 이미 반복 사용한 알려진 VAL이다. 배경pixel 오탐은 손상사진 안에서 mask 밖 pixel을 분모로 한 값이고, 정상 사진 오탐과 다르다. 손상사진 점수는 유효 grid의 전체·구역 결합 출력 최대값이다. 보정된 고장 확률이 아니다. 예약 BeltCrack TEST는 사용하지 않았다.','',
        '## 선택과 보존','',
        '**사전에 정한 pilot 자료 품질 조건에 따라 현재 모델·정책·역할4 ZIP을 유지했다.** CAL3개와 원래 촬영 구간 정보 부족으로 새 후보는 적용하지 않는다. 이 조건은 학습·새 점수 비교 전에 고정했다.','',
        '| 기존 손상 성능 확인 조건 | 결과 |','|---|---|']
    for name,passed in choice['known_metric_guards'].items(): lines.append('| '+name+' | '+('통과' if passed else '실패')+' |')
    lines += ['',f"기존 파일{v['protected_files']:,}개와 Python소스{v['legacy_sources']}개의 SHA를 보존했고 입력·이전 보고서 수정시간도 유지했다. 마지막 progress와 final 가중치가 같고 모든 비벨트 tensor는 직전과 같다. TRAIN 입력의3개 기능 CPU/GPU 출력 일치, 이전 시설·이물질 CPU 출력 정확 일치, 원본 구역 CPU/GPU 일치를 확인했다. 새 Qt/C++·Android 실행 시험은 수행하지 않았다.",'',
        f"현재 선택 모델SHA `{START_SHA}`, 정책SHA `{choice['policy']['sha256']}`. 기존 앱 기본 모델도 자동 교체하지 않았다. 역할4는 기존 `{ZIP_NAME}`을 사용한다.",'',
        '## 다음 효율적인 작업','',
        '현재는 새 데이터를 전부 학습에 넣는 것보다, 원래 촬영 구간을 확인할 수 있는 정상 전체사진과 어려운 손상 사진의 정답을 더 확보하는 것이 우선이다. 이번 수집·검수 코드는 정상 구역만 허용하고 중복과 unknown 범위를 기록하므로 추가 자료를 같은 기준으로 비교할 수 있다. 정상 구역 점수를 사진 전체 정상·수리 완료·모터 내부 고장으로 바꾸지 않는다.','',
        '![실제 초기 추가 학습 비교](unified-belt-normal-comparison.png)','',
        '- [다운로드·정답 범위·분리 요약](unified-belt-normal-source-summary.json)','- [사전 조건](unified-belt-normal-protocol.json)','- [실제 학습 기록](unified-belt-normal-training.json)',
        '- [모든 계수와 구역 점수](unified-belt-normal-validation.json)','- [현재 모델 유지](unified-belt-normal-selection.json)','- [보존·재개 확인](unified-belt-normal-verification.json)',
        '- [실행 방법](../../docs/UNIFIED_BELT_NORMAL_STUDY_KO.md)','- [이전 모델의 출처·조건](UNIFIED_MAIN_ATTRIBUTION_KO.md)','']
    text_new(REPORTS/'UNIFIED_BELT_NORMAL_RESULTS_KO.md','\n'.join(lines))
    text_new(REPO/'safelog-cpp/docs/UNIFIED_BELT_NORMAL_STUDY_KO.md','''# 정상 벨트 구역 · 신규 자료 검수와 초기 학습

## 자료와 정답 범위

Gofaones Workspace의 belt defects v1·v2를 Roboflow 로그인 후 COCO ZIP으로 내려받았다. 원본은 data/belt-normal-roboflow-v1 및 data/belt-normal-rois-v1에 보관하며 Git에 올리지 않는다. CC BY4.0이고 실제 출처·라이선스와 변경 내역은 결과 보고서에 기록한다.

prepare_belt_normal_rois.py는 두 버전의 실제 라벨을 검사하고, 명시적 normal belt bbox의 중앙 구역만 고른다. 마모 라벨 부모, 다른 라벨과 겹치는 구역, 그룹 중복, 이전 검증 자료와의 유사성을 검사한다. 사진 전체 정상과 dense mask 정답은 만들지 않는다. 본 연구 준비 경로는 제작 환경의 Downloads와 고정된 data 경로를 사용한다.

## 실제 흐름

1. tests/test_belt_normal_roi.py: 정상 구역 범위·금지 겹침·frame gap·holdout·증강 그룹 확인.
2. prepare_belt_normal_rois.py: 원본 ZIP은 보존하고 FIT14·CAL3·보관7 정상 구역과 source SHA를 생성한다.
3. tests/test_belt_normal_learning.py: 구역 부재 분류 손실과 single-class 출력 조건 확인.
4. study_unified_belt_normal.py --mode freeze: TRAIN draws, 실제4epoch512업데이트, source·입력 SHA, 비교 조건과 적용 금지를 고정한다.
5. 소스와 protocol을 먼저 commit한 뒤 --mode train으로 학습한다. 가중치·optimizer·scheduler·cursor·CPU/CUDA RNG를64업데이트마다 저장한다. Windows PID는 읽기 전용process_exists로 확인한다. 완료된final은 다시 학습하지 않는다.
6. --mode eval: 같은 알려진 손상VAL353장과 새 정상CAL3개를 비교한다. 보관7개와 예약BeltCrack TEST 모델 점수는 계산하지 않는다.
7. --mode verify: 실제 journal, final/progress, 비벨트 tensor와 기존 입력·보고서·소스를 보존했는지 확인한다.
8. report_unified_belt_normal.py: 결과·한계·현재 전달물SHA·재개 기준을 작성한다.

```powershell
./.venv/Scripts/python.exe -m unittest discover -s tests -p test_belt_normal_roi.py
./.venv/Scripts/python.exe -m unittest discover -s tests -p test_belt_normal_learning.py
./.venv/Scripts/python.exe scripts/prepare_belt_normal_rois.py
./.venv/Scripts/python.exe scripts/study_unified_belt_normal.py --mode freeze
# 소스와 protocol commit 후
./.venv/Scripts/python.exe scripts/study_unified_belt_normal.py --mode train
./.venv/Scripts/python.exe scripts/study_unified_belt_normal.py --mode eval
./.venv/Scripts/python.exe scripts/study_unified_belt_normal.py --mode verify
./.venv/Scripts/python.exe scripts/report_unified_belt_normal.py
```

완료된 준비·protocol·평가·보고서는 배타적으로 생성한다. 같은 연구를 새 실험으로 다시 실행하거나 기존 결과를 덮어쓰지 않는다. 새 추가 자료나 손실 비교는 새 namespace와 사전조건으로 진행한다. 현 단계는 CAL3개와 촬영 구간 미검증으로 적용 금지된pilot이다. 전체 정상 사진 오탐은 여전히 미측정이다.

[실제 결과](../ai-training/reports/UNIFIED_BELT_NORMAL_RESULTS_KO.md), [새 자료의 실제 출처](https://universe.roboflow.com/gofaones-workspace/belt-defects/dataset/2), [이전 모델 출처·사용 조건](../ai-training/reports/UNIFIED_MAIN_ATTRIBUTION_KO.md).
''')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(16,5))
    pairs=[(a['pixel_iou'],b['pixel_iou']),(a['pixel_fnr'],b['pixel_fnr']),(a['photo_rates']['fnr'],b['photo_rates']['fnr']),(old_roi['ROI_FPR'],new_roi['ROI_FPR'])]
    titles=['Known pixel IoU (higher better)','Known damage pixel FNR','Known damage photo FNR','Normal ROI alarms (ONLY 3 cores)']
    for ax,pair,title in zip(axes,pairs,titles):
        values=[100*x for x in pair]; bars=ax.bar(['Accepted','Pilot4'],values,color=['#54738e','#519473'])
        for bar,value in zip(bars,values): ax.text(bar.get_x()+bar.get_width()/2,value,f'{value:.2f}%',ha='center',va='bottom',fontsize=9)
        ax.set_title(title,fontsize=11); ax.set_ylim(0,max(max(values)*1.2,.1)); ax.set_ylabel('%')
    fig.suptitle('Actual 4 epochs / 512 updates / 14 normal FIT cores / current model RETAINED',fontsize=13)
    fig.text(.5,.02,'ROI CAL3 is provisional; whole-photo normal belt FPR remains unknown; no field5% proof',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.05,1,.93]); fig.savefig(REPORTS/'unified-belt-normal-comparison.png',dpi=160); plt.close(fig)
    header='## 신규 정상 구역 자료·초기 학습 · 2026-10-13'
    block='\n'.join([header,'',
        'belt defects COCO ZIP497장·1,548장을 실제 받았고, 중복·라벨 겹침 검사 후 정상 구역24개(FIT14·CAL3·보관7)를 분리했다. **실제4epoch·512회 추가 학습**을 수행했다. 증강본과 반복draw를 새 독립 사진으로 세지 않는다. 사진 전체 정상·정상pixel mask 정답은 만들지 않았다.','',
        f"정상 CAL 구역 이상 후보 {old_roi['false_alarms']}/3→{new_roi['false_alarms']}/3. 기존 손상VAL 위치 IoU {pct(a['pixel_iou'])}→{pct(b['pixel_iou'])}, 손상사진 미탐{a['photo_rates']['fn']}/353→{b['photo_rates']['fn']}/353. CAL3개와 원래 촬영 구간 정보 부족으로 현재 모델·정책·전달ZIP을 유지했다.",'',
        '정상 전체사진 오탐은 미측정이고 산업현장 전체5% 검증은 없다. 시설·이물질 가중치를 보존했다. 보관7개와 예약BeltCrack TEST의 모델 점수는 계산하지 않았다.',''])
    docs=[REPORTS/'UNIFIED_BELT_NORMAL_RESULTS_KO.md',REPO/'safelog-cpp/docs/UNIFIED_BELT_NORMAL_STUDY_KO.md']
    for path,link in [(REPO/'README.md','safelog-cpp/ai-training/reports/UNIFIED_BELT_NORMAL_RESULTS_KO.md'),(ROOT/'README_KO.md','reports/UNIFIED_BELT_NORMAL_RESULTS_KO.md'),
        (REPO/'safelog-cpp/docs/EQUIPMENT_SCOPE_KO.md','../ai-training/reports/UNIFIED_BELT_NORMAL_RESULTS_KO.md'),(REPO/'safelog-cpp/docs/DATASET_MODEL_MAP_KO.md','../ai-training/reports/UNIFIED_BELT_NORMAL_RESULTS_KO.md')]:
        content=path.read_bytes().decode('utf-8'); assert header not in content; nl='\r\n' if '\r\n' in content else '\n'; title,body=content.split(nl+nl,1)
        addition=block+'\n- [실제 다운로드·검수·학습 결과]('+link+')\n'
        path.write_bytes((title+nl+nl+addition.replace('\n',nl)+nl+body).encode('utf-8')); docs.append(path)
    path=OUTPUTS/ZIP_NAME; assert sha(path)==ZIP_SHA
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None and sum(x.endswith('.pt') for x in archive.namelist())==1
        hashes=json.loads(archive.read('package-sha256.json')); assert all(hashlib.sha256(archive.read(n)).hexdigest()==h for n,h in hashes.items())
        model=json.loads(archive.read('model.json')); assert model['sha256']==START_SHA and model['inferencePolicySha256']==choice['policy']['sha256']
    links=0
    for doc in docs:
        for link in local_links(doc):
            assert link.is_relative_to(REPO) and link.exists(),link
            if link.suffix: assert link.is_file(),link
            links+=1
    audit={'status':'passed','downloaded_exports':[{'version':1,'images':497,'sha256':n['v1_archive_sha256']},{'version':2,'images':1548,'sha256':n['archive_sha256']}],
        'normal_ROIs':n['split_counts'],'performed_neural_epochs':4,'performed_unique_updates':512,'actual_journal_updates':v['actual_journal_updates'],
        'normal_whole_photo_labels':0,'dense_normal_pixel_masks':0,'normal_whole_photo_FPR':None,'normal_CAL_ROI_results':{k:{n:v for n,v in values.items() if n!='rows'} for k,values in r['normal_ROI_CAL'].items()},
        'known_positive_VAL_reused':True,'reserved_normal7_model_scores_computed':False,'original_recording_groups_verified':False,
        'old_BeltCrack_TEST_used':False,'new_selected_checkpoint':False,'retained_model_sha256':START_SHA,'retained_policy_sha256':choice['policy']['sha256'],
        'pilot_weights_sha256':t['weights_sha256'],'old_files_preserved':v['protected_files'],'old_sources_preserved':v['legacy_sources'],
        'CPU_GPU_three_task_and_native_tiles_checked':True,'new_CPP_HTTP_tests':0,'retained_handoff':ZIP_NAME,'retained_handoff_sha256':ZIP_SHA,
        'ZIP_CRC_weights_policy_and_members_checked':True,'local_links_checked':links,'new_handoff':None,'app_default_promoted':False,
        'industrial_all_tasks_5percent_verified':False,'raw_slice_helper_score_scope':'Actual photo scores are fused global/tile maxima over valid pixels, not calibrated; helper shorthand does not alter raw counts'}
    new(REPORTS/'unified-belt-normal-final-audit.json',audit); new(OUTPUTS/'UNIFIED_BELT_NORMAL_RESUME_2026-10-13.json',audit)
    text_new(OUTPUTS/'UNIFIED_BELT_NORMAL_RESUME_2026-10-13.md','\n'.join(['# 정상 구역 초기 추가 학습 재개 기준 · 2026-10-13','',block,'',
        '현재 선택 모델SHA: '+START_SHA,'정책: global_tile_mean / '+choice['policy']['sha256'],'역할4 ZIP: '+ZIP_NAME,'ZIP SHA: '+ZIP_SHA,'',
        '이번 normal-roi-pilot-v1은 완료된 연구다. 같은 namespace로 prepare/freeze/eval/report를 반복하지 않는다. Windows PID 확인에는 읽기 전용process_exists를 사용한다. 새 연구는 별도 run·protocol과 학습 전commit을 사용한다.','',
        '다음 우선순위: 원래 촬영 구간을 확인할 수 있는 정상 전체사진·어려운 손상 사진 정답 확보. normal bbox에서 전체 사진 정상 정답을 만들지 않는다. 현재 정상CAL3만으로5%를 선언하지 않는다. 보관7개는 모델 점수 미사용이고 새로운독립 현장 시험으로 주장하지 않는다.','',
        '결과: safelog-cpp/ai-training/reports/UNIFIED_BELT_NORMAL_RESULTS_KO.md','파이프라인: safelog-cpp/docs/UNIFIED_BELT_NORMAL_STUDY_KO.md',
        '데이터: data/belt-normal-rois-v1/manifest.json / source.coco.zip (Git제외)','pilot weights: runs/unified-belt-normal-roi-pilot-v1/train/final.pt (배포 제외)','']))
    print(json.dumps({'reported':True,'actual_updates':512,'normal_rois':24,'CAL_only':3,'new_handoff':False,'links_checked':links}),flush=True)


if __name__=='__main__': main()
