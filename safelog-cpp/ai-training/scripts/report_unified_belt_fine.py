"""Describe measured fine-view comparisons without implying additional training."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.unified_main_data import read,sha
from scripts.study_unified_belt_fine import MODES,BASELINE,new,validate
REPORTS=ROOT/'reports'; REPO=ROOT.parents[1]


def pct(x): return '미측정' if x is None else f'{100*x:.2f}%'


def main():
    validate(); choice=read(REPORTS/'unified-belt-fine-selection.json'); verified=read(REPORTS/'unified-belt-fine-verification.json')
    results={m:read(REPORTS/f'unified-belt-fine-{m}-validation.json') for m in MODES}
    labels=['기존 전체＋구역','전체＋세밀 구역','전체＋기존·세밀 구역']
    selected=choice['selected']; decision='기존 설정과 전달 ZIP을 유지한다.' if selected==BASELINE else f'사전 조건을 모두 통과한 **{selected}**를 연구 연결용 설정으로 선택했다.'
    lines=['# 벨트 작은 손상 · 세밀 구역 분석 비교', '', '기준일: 2026-10-13', '',
        '## 이번 작업', '',
        '**추가 가중치 학습0회·새 독립 사진0장·새 정답0개**다. 현재 선택한 native_mix 가중치를 동일하게 사용하고, 원본 구역 분석 크기와 결합 방식만 사전에 정한2개 후보로 비교했다. 모델 파일 하나·공통 MobileNet 분석부 하나를 반복 사용한다. 이번 비교를 모델 재학습으로 세지 않는다.', '',
        '기존 알려진 VAL353장·4개 촬영 구간을 반복 사용한 연구 비교다. 실제 산업현장 일반화·새로운 독립 검증의 결과가 아니다. 예약 TEST는 사용하지 않았다. 기준0.5와 작성자640 mask 평가 좌표를 유지했다.', '',
        '## 전체 비교', '', '| 항목 | '+' | '.join(labels)+' |', '|---|---:|---:|---:|']
    for label,fn in [('위치 IoU',lambda r:r['pixel_iou']),('손상 pixel 미탐',lambda r:r['pixel_fnr']),('배경 pixel 오탐',lambda r:r['pixel_fpr']),('손상 사진 미탐',lambda r:r['photo_rates']['fnr'])]:
        lines.append('| '+label+' | '+' | '.join(pct(fn(results[m]['summary'])) for m in MODES)+' |')
    lines += ['| 손상 사진에서 정답과 겹친 예측이 전혀 없음 | '+' | '.join(str(results[m]['summary']['positive_photos_without_any_gt_overlap']) for m in MODES)+' |',
        '| 정상 벨트 사진 오탐 | 미측정 | 미측정 | 미측정 |', '',
        '배경 pixel 오탐은 손상 사진 안에서 mask 밖의 pixel을 분모로 한 값이다. 정상 전체 사진의 오탐과 다르다. 모든 사진이 작성자 손상 사진이라 정상 사진 오탐 분모는0이며 값은 null이다. 사진 미탐은 최대 출력이0.5를 넘는지로 측정하므로 위치 IoU와 함께 확인해야 한다.', '',
        '## 작은 손상과 촬영 구간별 오류', '',
        '손상 영역이 유효 사진의0.25% 이하인55장이 작은 손상 묶음이다. 이는 작성자 정답 기반의 사후 설명용 구분이며, 서버의 구역 선택에는 정답·손상 크기·모델 점수를 넣지 않는다.', '',
        '| 묶음·장수 | 기존 IoU / pixel 미탐 | 세밀 IoU / pixel 미탐 | 두 크기 IoU / pixel 미탐 |', '|---|---:|---:|---:|']
    for k,b in results[BASELINE]['slices'].items():
        cells=[pct(results[m]['slices'][k]['pixel_iou'])+' / '+pct(results[m]['slices'][k]['pixel_fnr']) for m in MODES]
        lines.append('| '+k+f" · {b['photos']}장 | "+' | '.join(cells)+' |')
    lines += ['', '## 사전 선택 조건과 결론', '', decision, '',
        '현재 선택한 구역 분석을 기준으로 IoU 최소0.5%p 개선, 배경 오탐·pixel 미탐·사진 미탐·정답 겹침 실패 유지, 작은 손상의 IoU·pixel 미탐 유지, 각 촬영 구간 pixel 미탐 악화1%p 이내를 요구했다. 평가 후에 통과 기준을 완화하지 않았다.', '',
        '| 조건 | 세밀 구역 | 두 크기 |', '|---|---|---|']
    for name in choice['guards'][MODES[1]]:
        lines.append('| '+name+' | '+' | '.join('통과' if choice['guards'][m][name] else '실패' for m in MODES[1:])+' |')
    lines += ['', '## 비용·보존·한계', '',
        '이 검증의 원본 사진은3840×2160 또는2160×3840이다. 기존 모드는전체1회＋구역6회, 세밀 모드는전체1회＋구역15회, 두 크기는전체1회＋6회＋15회로 같은 모델을 호출한다. CPU 서버에서는 처리시간이 늘어날 수 있으므로 별도 TRAIN 입력으로 실제 지연을 확인한 후 전달한다. 640 이하 작은 사진이나 종횡비3 초과 사진은 전체 분석으로 돌아간다.', '',
        f"기존 파일{verified['protected_files']:,}개는 SHA와 수정시간을 보존했고, 기존 Python 소스{verified['legacy_sources']}개도 SHA가 같다. 기존353장 기준 pixel 계수를 정확히 재현했다. 가중치와 시설·이물질 기능은 변경하지 않았다. 모델SHA `{choice['checkpoint_sha256']}`와 선택한 정책SHA `{choice['policy']['sha256']}`를 함께 기록한다.", '',
        '독립 산업현장 전체 오탐·미탐5% 미만은 검증하지 못했다. 모터·롤러 내부 고장과 작동 상태는 not_assessed다. 시설·이물질 성능은 기존 결과를 계승하며 새로 평가한 값이 아니다. 기존 앱 기본 모델은 자동 교체하지 않는다.', '',
        '![실제 세밀 구역 비교](unified-belt-fine-comparison.png)', '',
        '- [사전 조건](unified-belt-fine-protocol.json)', '- [기존 오류 구분](unified-belt-fine-baseline-error-audit.json)',
        '- [선택과 통과 여부](unified-belt-fine-selection.json)', '- [보존 확인](unified-belt-fine-verification.json)',
        '- [출처·사용 조건](UNIFIED_MAIN_ATTRIBUTION_KO.md)', '- [기존 전달 결과](UNIFIED_BELT_TILES_RESULTS_KO.md)', '']
    text='\n'.join(lines)
    dest=REPORTS/'UNIFIED_BELT_FINE_RESULTS_KO.md'
    with dest.open('x',encoding='utf-8',newline='\n') as f: f.write(text)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(16,5))
    titles=['Pixel IoU (higher better)','Damage pixel FNR (lower better)','Background pixel FPR (lower better)','Small-damage IoU (higher better)']
    keys=['pixel_iou','pixel_fnr','pixel_fpr','small']
    for ax,key,title in zip(axes,keys,titles):
        vals=[100*(results[m]['slices']['GT-area:<=0.25%']['pixel_iou'] if key=='small' else results[m]['summary'][key]) for m in MODES]
        bars=ax.bar(['Accepted','Fine','Both'],vals,color=['#54738e','#519473','#caa34b'])
        for bar,v in zip(bars,vals): ax.text(bar.get_x()+bar.get_width()/2,v,f'{v:.2f}%',ha='center',va='bottom',fontsize=9)
        ax.set_ylim(0,max(vals)*1.22); ax.set_title(title,fontsize=11); ax.set_ylabel('%')
    fig.suptitle('Same ONE checkpoint / known reused VAL353 / 0 new neural updates',fontsize=13)
    fig.text(.5,.02,'Normal whole-photo belt FPR unavailable; no independent field validation',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.05,1,.93]); fig.savefig(REPORTS/'unified-belt-fine-comparison.png',dpi=160); plt.close(fig)
    block='\n'.join(['## 세밀 구역 추론 비교 · 2026-10-13','',
        '같은 통합 모델로 더 작은 원본 구역을 보는2개 고정 설정을 비교했다. **추가 학습0회·새 독립 사진0장**이다. '+decision,'',
        f"선택 설정 위치 IoU {pct(choice['summary']['pixel_iou'])}, 손상pixel 미탐 {pct(choice['summary']['pixel_fnr'])}, 배경pixel 오탐 {pct(choice['summary']['pixel_fpr'])}, 손상사진 미탐 {pct(choice['summary']['photo_rates']['fnr'])}. 정상사진 오탐은 미측정이고 전체 산업현장5% 검증은 없다.",''])
    for path,link in [(REPO/'README.md','safelog-cpp/ai-training/reports/UNIFIED_BELT_FINE_RESULTS_KO.md'),(ROOT/'README_KO.md','reports/UNIFIED_BELT_FINE_RESULTS_KO.md'),
        (REPO/'safelog-cpp/docs/EQUIPMENT_SCOPE_KO.md','../ai-training/reports/UNIFIED_BELT_FINE_RESULTS_KO.md'),(REPO/'safelog-cpp/docs/DATASET_MODEL_MAP_KO.md','../ai-training/reports/UNIFIED_BELT_FINE_RESULTS_KO.md')]:
        previous=path.read_bytes().decode('utf-8'); assert block.split('\n')[0] not in previous
        nl='\r\n' if '\r\n' in previous else '\n'; title,body=previous.split(nl+nl,1)
        addition=block+'\n- [실제 비교·오류 분석·선택]('+link+')\n'
        path.write_bytes((title+nl+nl+addition.replace('\n',nl)+nl+body).encode('utf-8'))
    print(json.dumps({'report':str(dest),'selected':selected,'new_neural_updates':0}),flush=True)


if __name__=='__main__': main()
