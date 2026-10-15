"""New narrower task split: keep previously viewed FIT groups in FIT."""
from collections import Counter,defaultdict
import hashlib
import math


def allocate_gear_groups(rows,viewed_groups):
    grouped=defaultdict(list)
    for row in rows:
        if row['phase'] not in ('fit','validation','cal') or row['functional_component_category']!='gear':
            raise ValueError('Only original eligible TRAIN gear groups may be allocated')
        grouped[row['group_id']].append(row)
    n=len(grouped)
    if n<14:raise ValueError('At least14 gear TRAIN groups required')
    small=math.ceil(.15*n);caps={'fit':n-2*small,'validation':small,'cal':small};fractions={'fit':.70,'validation':.15,'cal':.15}
    if not set(viewed_groups)<=set(grouped) or len(viewed_groups)>caps['fit']:raise ValueError('Invalid viewed FIT reservation')
    total=Counter(r['label'] for r in rows)
    if not all(total[y]>=100 for y in (0,1)):raise ValueError('Insufficient labels')
    counts={p:Counter() for p in caps};assigned=Counter();result={}
    for g in sorted(viewed_groups):
        result[g]='fit';assigned['fit']+=1;counts['fit'].update(r['label'] for r in grouped[g])
    for g in sorted(set(grouped)-set(viewed_groups),key=lambda g:(-len(grouped[g]),hashlib.sha256(('gear28:'+g).encode()).hexdigest(),g)):
        labels=Counter(r['label'] for r in grouped[g])
        def cost(p):return sum(((counts[p][y]+labels[y])/total[y]-fractions[p])**2-(counts[p][y]/total[y]-fractions[p])**2 for y in (0,1))
        phase=min((p for p in caps if assigned[p]<caps[p]),key=lambda p:(cost(p),list(caps).index(p)))
        result[g]=phase;assigned[phase]+=1;counts[phase].update(labels)
    assert dict(assigned)==caps
    return result
