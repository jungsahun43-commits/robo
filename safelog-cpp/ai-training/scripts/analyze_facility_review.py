"""Validation-only abstention tradeoff; never replaces full-coverage error target."""
import argparse
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.train_facility_target import read,save,sha,TARGETS
from scripts.facility_error_target import wilson_interval


def tail_metrics(target,score,cutoff,positive):
    target,score=np.asarray(target),np.asarray(score,dtype=float)
    if (target.ndim!=1 or target.shape!=score.shape or not np.isfinite(score).all()
            or not ((score>=0)&(score<=1)).all() or not ((target==-1)|(target==0)|(target==1)).all()):
        raise ValueError('Invalid targets or probabilities')
    known=target>=0
    accepted=((score>=cutoff) if positive else (score<cutoff))&known
    total=int(accepted.sum());errors=int((accepted&(target!=int(positive))).sum())
    return {'accepted':total,'errors':errors,'known_photos':int(known.sum()),
            'coverage':total/int(known.sum()) if known.any() else 0.,
            'conditional_error':errors/total if total else None,'ci95':wilson_interval(errors,total)}


def choose_tail(domains,anchor,positive,conservative):
    scores=np.unique(np.concatenate([p for _,p in domains.values()]))
    candidates=np.unique(np.r_[0.,scores,np.nextafter(scores,2.),anchor,np.nextafter(1.,2.)])
    candidates=candidates[candidates>=anchor] if positive else candidates[candidates<=anchor]
    choices=[]
    for cutoff in candidates:
        metrics={d:tail_metrics(t,p,cutoff,positive) for d,(t,p) in domains.items()}
        if all(m['accepted']>=10 and (m['ci95'][1] if conservative else m['conditional_error'])<.05 for m in metrics.values()):
            choices.append((sum(m['accepted'] for m in metrics.values()),float(cutoff),metrics))
    if not choices:
        cutoff=float(np.nextafter(1.,2.)) if positive else 0.
        return {'supported':False,'cutoff':cutoff,'domains':{d:tail_metrics(t,p,cutoff,positive) for d,(t,p) in domains.items()}}
    _,cutoff,metrics=max(choices,key=lambda r:(r[0],r[1] if positive else -r[1]))
    return {'supported':True,'cutoff':cutoff,'domains':metrics}


def joint_metrics(targets,scores,cutoffs):
    targets,scores=np.asarray(targets),np.asarray(scores)
    if targets.ndim!=2 or targets.shape!=scores.shape or len(cutoffs)!=targets.shape[1]:
        raise ValueError('Joint target/score/cutoff mismatch')
    masks=[];predictions=[]
    for k,(low,high) in enumerate(cutoffs):
        if low>high:raise ValueError('Overlapping judgment cutoffs')
        # Reuse checks even when a direction is unsupported and accepts nothing.
        tail_metrics(targets[:,k],scores[:,k],high,True)
        masks.append(((scores[:,k]<low)|(scores[:,k]>=high))&(targets[:,k]>=0))
        predictions.append(scores[:,k]>=high)
    accepted=np.stack(masks,axis=1).all(1)
    wrong=(np.stack(predictions,axis=1)!=targets).any(1)&accepted
    total,errors=int(accepted.sum()),int(wrong.sum())
    return {'photos':len(targets),'fully_known_photos':int((targets>=0).all(1).sum()),'accepted':total,
            'review_required':len(targets)-total,'coverage':total/len(targets) if len(targets) else 0.,
            'accepted_wrong_photos':errors,'conditional_photo_error':errors/total if total else None,
            'ci95':wilson_interval(errors,total)}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--name',default='facility-presence-target-spatial')
    args=parser.parse_args();run=ROOT/'runs'/args.name
    selection=read(run/'TARGET-SELECTION.json');training=read(run/'TRAINING.json')
    if selection['selection_split']!='val' or training['status']!='complete' or selection['weights_sha256']!=sha(run/'best.pt'):
        raise ValueError('Only completed, unchanged validation candidate allowed')
    classes=training['classes'];domains={};sources={}
    for domain in ('dacl','damsegment','codebrim'):
        path=run/f'validation-{domain}.json';cache_path=run/f'target-validation-{domain}-grid1.json'
        pred,cache=read(path),read(cache_path)
        if pred['split']!='val' or pred['weights_sha256']!=selection['weights_sha256'] or cache['signature']['weights_sha256']!=selection['weights_sha256'] or cache['signature']['grid']!=1 or cache['classes']!=classes:
            raise ValueError('Validation cache does not belong to selected model')
        t,p=np.array(pred['targets']),np.array(cache['probabilities'])
        if t.shape!=p.shape:raise ValueError('Validation order/shape changed')
        domains[domain]=(t,p);sources[domain]={'targets_sha256':sha(path),'scores_sha256':sha(cache_path)}
    baseline=next(a for a in selection['attempts'] if a['grid']==1)
    policies={}
    for rule,conservative in (('observed_only',False),('wilson_upper',True)):
        result={}
        for label in TARGETS:
            k=classes.index(label);values={d:(t[:,k],p[:,k]) for d,(t,p) in domains.items()}
            anchor=baseline['per_class'][label]['threshold']
            neg,pos=(choose_tail(values,anchor,positive,conservative) for positive in (False,True))
            summary={}
            for d,(t,p) in values.items():
                low,high=neg['domains'][d],pos['domains'][d];known=int((t>=0).sum())
                accepted=low['accepted']+high['accepted'];errors=low['errors']+high['errors']
                summary[d]={'known_photos':known,'accepted':accepted,'review_required':known-accepted,
                            'coverage':accepted/known,'conditional_error':errors/accepted if accepted else None,
                            'accepted_errors':errors,'all_photos_target_replaced':False}
            result[label]={'negative':neg,'positive':pos,'summary':summary}
        policies[rule]=result
    joint={}
    indices=[classes.index(label) for label in TARGETS]
    for rule,result in policies.items():
        cutoffs=[(result[label]['negative']['cutoff'],result[label]['positive']['cutoff']) for label in TARGETS]
        joint[rule]={d:joint_metrics(t[:,indices],p[:,indices],cutoffs) for d,(t,p) in domains.items()}
    output={'run':args.name,'stage':'validation_diagnostic_only','weights_sha256':selection['weights_sha256'],
            'positive_support':{label:{d:{'positive_photos':int((t[:,classes.index(label)]==1).sum()),
                     'zero_error_ci95':wilson_interval(0,int((t[:,classes.index(label)]==1).sum()))}
                     for d,(t,p) in domains.items()} for label in TARGETS},
            'sources':sources,'policies':policies,'joint_photo_diagnostics':joint,'deployed':False,
            'criterion':'Common per-label low/high cutoffs across all three domains. Conditional mistake rate per predicted tail, each <5%; upper rule uses nominal two-sided 95% Wilson upper bound; >=10 accepted/tail/domain.',
            'limitation':'Cutoffs selected on validation; intervals ignore repeated selection and related patches. No independent test/field guarantee. Abstained cases are unjudged, never counted correct. Conditional error is NOT full-coverage FNR/FPR.'}
    save(ROOT/'reports'/f'{args.name}-review-diagnostic.json',output)
    lines=['# 자동 판단·확인 대기 검증 진단','',
           '**전체 사진에 대한 기존5% 목표는 바꾸지 않는다. 앱 적용·독립 시험 전이다.**','',
           '자동 판단은 손상 제안/미검출 판단이며 구조 안전 판정이 아니다. 보류 사진을 정답으로 세지 않는다.',
           '아래 오류는 자동으로 판단한 사진만의 조건부 오답률로, 기존 미탐률·오탐률과 분모가 다르다.',
           '낮은 확률과 높은 확률에 별도 공통 기준을 정하고 그 사이를 확인 대기로 둔다. 각 자료·각 판단 방향에서 지원되는 기준이 없으면 그 방향을 전부 보류한다.',
           '검증 자료로 기준을 고른 결과다. Wilson 상한도 선택 편향과 같은 장면의 상관을 해결하지 않으므로 현장 보장으로 쓸 수 없다.','']
    support=output['positive_support']['concrete_spalling']['damsegment']
    lines += [f"자료 수 한계: Dam 박락 양성은{support['positive_photos']}장이다. 이 양성을 모두 틀리지 않고 판단해도 명목95% 상한은{support['zero_error_ci95'][1]*100:.2f}%이다.",
              '따라서 세 자료의 모든 판단 방향에서95% 상한5%를 요구하는 진단은 완벽한 모델만으로도 해결되지 않는다. 추가 독립 양성 자료가 필요하다. 이것은 기존 관측 미탐·오탐5% 목표와 별도 조건이다.','']
    for rule,title in (('observed_only','검증 관측 오류만5% 미만으로 제한'),('wilson_upper','명목95% 상한까지5% 미만으로 요구')):
        lines += [f'## {title}','','| 항목 | 자료 | 자동 판단/전체 | 자동 비율 | 확인 대기 | 자동 판단 중 오류 |','|---|---|---:|---:|---:|---:|']
        for label,policy in policies[rule].items():
            for domain,m in policy['summary'].items():
                error=f"{m['conditional_error']*100:.2f}% ({m['accepted_errors']}건)" if m['conditional_error'] is not None else '판단 없음'
                lines.append(f"| {label} | {domain} | {m['accepted']}/{m['known_photos']} | {m['coverage']*100:.1f}% | {m['review_required']} | {error} |")
        lines+=['','균열·박락 **둘 다** 자동 판단한 사진(둘 중 하나라도 틀리면 사진 오답):','',
                '| 자료 | 자동 사진/전체 | 자동 비율 | 확인 대기 | 자동 사진 중 오류 |','|---|---:|---:|---:|---:|']
        for domain,m in joint[rule].items():
            error=f"{m['conditional_photo_error']*100:.2f}% ({m['accepted_wrong_photos']}건)" if m['conditional_photo_error'] is not None else '판단 없음'
            lines.append(f"| {domain} | {m['accepted']}/{m['photos']} | {m['coverage']*100:.1f}% | {m['review_required']} | {error} |")
        lines+=['','각 판단 방향의5% 조건이 두 항목을 합친 사진 오답5%를 보장하지 않는다. 결합 오류는 위에 별도로 계산한다.',
                '판단 방향별 미지원 여부·분자·분모·구간·확률 기준은 대응 JSON에 모두 기록한다.','']
    (ROOT/'reports'/f'{args.name}-review-diagnostic_KO.md').write_text('\n'.join(lines),encoding='utf-8')
    print(__import__('json').dumps(joint,indent=2))


if __name__=='__main__':main()
