"""Measured metal20% milestone without hiding the localization tradeoff."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.study_metal_reference import read,validate,REPORTS,RUN,VARIANTS


def text_new(path,text):
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write(text)


def pct(x):return f'{100*x:.2f}%'


def main():
    p=validate();choice=read(REPORTS/'equipment-metal-reference-v1-selection.json');v=read(REPORTS/'equipment-metal-reference-v1-verification.json')
    assert choice['selected']=='calibration_only20'
    results=[p['baseline'],*[read(REPORTS/f'equipment-metal-reference-v1-{n}-validation.json') for n in VARIANTS]]
    lines=['# 금속판 미탐 보강 · 정상 기준과 정상 특징 비교','','기준일: 2026-10-14','','## 결과','',
        '**정상 사진으로 정한 경고 기준 조정안**을 선택했다. 같은 알려진TEST97장(정상26·결함71)에서 오탐1/26=3.85%, 미탐13/71=18.31%로 사진 단위20% 목표를 통과했다. 직전 미탐18/71=25.35%에서5장을 더 발견했으며 정상 사진1장을 추가로 경고했다.',
        '다른 설비의 기준·신경망 가중치·사진 점수·위치heatmap은 유지했다. 금속판에서도 raw사진점수와pixel기준/heatmap은 같다. 추가신경망학습은0회이며 정상자료 기반 보정과참조특징저장 실험이다.','',
        '| 항목 | 기존 시제품 | 기준 조정20 | 정상 특징 비교20 |','|---|---:|---:|---:|']
    for label,key in [('사진 오탐','fpr'),('사진 미탐','fnr')]:lines.append('| '+label+' | '+' | '.join(pct(r['photo_rates'][key]) for r in results)+' |')
    for label,key in [('위치 IoU','pixel_iou'),('손상pixel 미탐','pixel_fnr')]:lines.append('| '+label+' | '+' | '.join(pct(r[key]) for r in results)+' |')
    lines += ['| 사진 미탐 건수 | '+' | '.join(str(r['photo_rates']['fn'])+'/71' for r in results)+' |',
        '| 정상 경고 건수 | '+' | '.join(str(r['photo_rates']['fp'])+'/26' for r in results)+' |','',
        '정상 특징 비교는 사진 미탐1/71=1.41%·오탐2/26=7.69%로 좋아졌지만 위치IoU가58.68%→47.85%로 낮아져 이번 선택에서 제외했다. 좋게 나온 사진 수치만 보고 위치 기능까지 개선됐다고 표시하지 않았다.','',
        '## 평가 전에 정한 조건','',
        '- 두 후보만 비교했다. A는기존신경망·점수·위치결과와pixel기준을유지하고금속판사진경고기준만조정한다. B는같은공통분석부의정상FIT특징과정확한최근접cosine거리로비교한다.',
        '- 두후보의사진경고기준은정상CAL20장만으로순위17번째값을사용한다. ceil((n+1)×0.8), 즉20%정상경고예산을사전에정했다. 기존15%예산은순위18번째였다.',
        '- B는정상FIT34장·8704patch만참조버퍼에저장한다. 입력특징은기존MobileNet low40/high960을따로정규화하고합친뒤3×3지역평균을계산한다. 제2신경망이나원본사진검색서비스를추가하지않았다.',
        '- 모든후보는사진FPR≤20%,사진FNR≤20%,직전보다FNR감소,위치IoU유지,pixelFNR유지의5조건을함께충족해야한다. TEST결함정답으로학습하거나임계값을순회조정하지않았다.',
        '- [PatchCore 논문](https://arxiv.org/abs/2106.08265)의정상patch참조아이디어를참고했지만동일구조·동일학습·동일성능을재현한구현이라고주장하지않는다.','',
        '## 현재 목표 상태','',
        '| 신규 유형 | 사진 목표 상태 |','|---|---|','| 부품 표면 | 기존오탐12.64%·미탐10.91% 유지·15%목표통과 |',
        '| 금속판 | 이번오탐3.85%·미탐18.31%·20%목표통과 |',
        '| 나머지9개 | 기존보강필요/평가표본부족상태유지 |','',
        '같은공개평가사진을재사용한비교다. 정상26장은표본이작고원래촬영장소·물리적부품그룹을검증하지못했으므로독립스마트폰산업현장성능이아니다. 금속판pixel미탐27.96%는그대로라위치기능은사진20%목표와분리해보강해야한다.','',
        '## 실제 보존·연결','',
        f"기존자료·모델·보고서{v['protected_files']:,}개의SHA와mtime,기존Python소스{v['legacy_sources']}개의SHA를보존했다. 두후보의모든기존신경망tensor는직전시제품과정확히같다. 부품표면을포함한다른10유형CPU출력정확일치와11유형CPU/GPU일치를확인했다.",
        '모델가중치가같아도사진경고기준과분석정책이달라질수있으므로modelSha256과inferencePolicySha256을함께저장한다. 새CPU서버는금속판의normalCALAlpha20%/rank17정책을명시한다. 기존3항목과새11유형의요청계약은유지한다.',
        '선택후보는새역할4시제품전달물로구분한다. 기존전달ZIP과일반앱기본모델은자동교체하지않는다. 실제분리ZIP·CPU·C++연결검사는release기록을확인한다.','',
        '선택모델SHA: `'+choice['checkpoint_sha256']+'`','',
        '![금속판 비교](equipment-metal-reference-v1-comparison.png)','',
        '- [사전 조건](equipment-metal-reference-v1-protocol.json)','- [선택 결과](equipment-metal-reference-v1-selection.json)',
        '- [기준 조정 검증](equipment-metal-reference-v1-calibration_only20-validation.json)',
        '- [정상 특징 검증](equipment-metal-reference-v1-normal_reference20-validation.json)',
        '- [보존 검증](equipment-metal-reference-v1-verification.json)','- [원래 전체 유형 성능](EQUIPMENT_EXPANSION_RESULTS_KO.md)',
        '- [같은 자료 출처·조건](EQUIPMENT_EXPANSION_ATTRIBUTION_KO.md)','- [역할4 연결](../../docs/METAL_CALIBRATION_HANDOFF_KO.md)','']
    text_new(REPORTS/'METAL_REFERENCE_RESULTS_KO.md','\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,4.5));labels=['Prior','CAL20','Normal reference20']
    for ax,field,title in [(axes[0],'fpr','Photo FPR'),(axes[1],'fnr','Photo FNR'),(axes[2],'pixel_iou','Location IoU (higher better)')]:
        values=[100*(r['photo_rates'][field] if field!='pixel_iou' else r[field]) for r in results]
        bars=ax.bar(labels,values,color=['#607684','#47864d','#a075b4']);ax.set_title(title);ax.tick_params(axis='x',labelrotation=12);ax.set_ylabel('%');ax.set_ylim(0,max(values)*1.25+1)
        for b,y in zip(bars,values):ax.text(b.get_x()+b.get_width()/2,y,f'{y:.2f}%',ha='center',va='bottom',fontsize=9)
        if field!='pixel_iou':ax.axhline(20,color='#ba4444',linestyle='--',linewidth=1)
    fig.suptitle('One frozen neural model: normal-only metal plate refinement (no gradient updates)')
    fig.text(.5,.02,'Known TEST97 reused: normal26 /defect71; CAL20 selected; not independent field validation',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,.93]);fig.savefig(REPORTS/'equipment-metal-reference-v1-comparison.png',dpi=160);plt.close(fig)
    repo=ROOT.parents[1];header='## 금속판 사진20% 목표 통과 · 2026-10-14'
    block='\n'.join([header,'','정상CAL사진만사용한기준조정으로금속판사진오탐3.85%·미탐18.31%를측정했다. 기존신경망가중치·금속판위치결과·다른유형의출력은유지했다. 이번추가신경망학습0회다.',
        '신규11유형중부품표면15%목표와금속판20%목표가통과했다. 같은공개평가자료를재사용한비교이며독립현장검증은없다. 금속판pixel미탐27.96%는별도보강항목이다.',''])
    for path,link in [(repo/'README.md','safelog-cpp/ai-training/reports/METAL_REFERENCE_RESULTS_KO.md'),(ROOT/'README_KO.md','reports/METAL_REFERENCE_RESULTS_KO.md'),
        (repo/'safelog-cpp/docs/EQUIPMENT_SCOPE_KO.md','../ai-training/reports/METAL_REFERENCE_RESULTS_KO.md'),(repo/'safelog-cpp/docs/DATASET_MODEL_MAP_KO.md','../ai-training/reports/METAL_REFERENCE_RESULTS_KO.md')]:
        s=path.read_bytes().decode('utf-8');assert header not in s;nl='\r\n' if '\r\n' in s else '\n';title,body=s.split(nl+nl,1)
        insert=block+'\n- [금속판 실제 보강 결과]('+link+')\n'
        path.write_bytes((title+nl+nl+insert.replace('\n',nl)+nl+body).encode('utf-8'))
    print(json.dumps({'report':'created','selected':choice['selected'],'new_neural_updates':0}),flush=True)


if __name__=='__main__':main()
