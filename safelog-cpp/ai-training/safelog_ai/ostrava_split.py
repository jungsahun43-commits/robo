"""Publisher roles take priority over one deterministic development allocation."""
import hashlib
import math


def assign(rows, externally_held_groups):
    test={r['group_id'] for r in rows if r['publisher_role']=='test'}
    held=test|set(externally_held_groups)
    validation={r['group_id'] for r in rows if r['publisher_role']=='validation'}-held
    train_groups={r['group_id'] for r in rows if r['publisher_role']=='train'}-held-validation
    ordered=sorted(train_groups,key=lambda v:(hashlib.sha256(('ostrava61:'+v).encode()).hexdigest(),v))
    cal=set(ordered[:math.ceil(.2*len(ordered))])
    result=[]
    for r in rows:
        role=r['publisher_role'];key=r['group_id']
        if role=='test':phase='test'
        elif key in held:phase='excluded_TEST_connected'
        elif role=='validation':phase='validation'
        elif role!='train':raise ValueError('Unknown publisher role')
        elif key in validation:phase='excluded_publisher_VAL_connected'
        else:phase='cal' if key in cal else 'fit'
        result.append({**r,'phase':phase})
    return result
