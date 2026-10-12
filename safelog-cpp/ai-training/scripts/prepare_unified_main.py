"""Retain publisher splits/unknowns and freeze the three-task optimization schedule."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.train_facility_target import dacl_items
from safelog_ai.unified_main import FACILITY_CLASSES
DATA = ROOT/'data/unified-main-v1'
EPOCHS = 20


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def cycling(pool, count, epochs, rng):
    result = []; order = []; at = 0
    for _ in range(epochs):
        selected = []
        while len(selected) < count:
            if at == len(order): order = rng.permutation(pool).tolist(); at = 0
            amount = min(count-len(selected), len(order)-at); selected.extend(order[at:at+amount]); at += amount
        result.append(selected)
    return result


def main():
    assert not DATA.exists(); DATA.mkdir()
    core = read(ROOT/'data/facility-spatial-training/train.json'); assert core['classes'] == list(FACILITY_CLASSES)
    facility = [dict(r, id=r['image'], task='facility_surface', split='train') for r in core['items'][:core['full_count']]]
    auxiliary = read(ROOT/'data/facility-auxiliary-training/train.json')
    aux = {r['image']: r['targets'] for r in auxiliary['items']}
    for r in facility: r['auxiliary_targets'] = aux.get(r['image'], [-1]*19)
    belt_base = 'data/equipment-beltcrack/expanded-cached-v2/'
    belt_manifest = read(ROOT/(belt_base+'manifest.json'))
    def belt_row(r):
        return {**r, 'task':'belt_damage', **{k: belt_base+r[k] for k in ('image','mask','valid')}}
    belt = [belt_row(r) for r in belt_manifest['rows'] if r['split'] == 'train']
    belt_val = [belt_row(r) for r in belt_manifest['rows'] if r['split'] == 'val']
    foreign_base = 'data/equipment-conveyor-foreign/train-expansion-v2/'
    f = read(ROOT/(foreign_base+'manifest.json'))
    def foreign_row(r, base=foreign_base):
        return {**r, 'task':'conveyor_foreign', 'label':r['foreign_object_photo_label'],
                'image':base+r['native'], 'image_sha256':r['native_sha256']}
    foreign = [foreign_row(r) for r in f['rows'] if r['split'] == 'train']
    calibration = [dict(foreign_row(r), split='calibration') for r in f['rows'] if r['split'] == 'val']
    common = [foreign_row(r) for r in f['rows'] if r['split'] == 'test']
    extra_base = 'data/equipment-conveyor-foreign/calibration-check-v3/'
    extra = [foreign_row(r, extra_base) for r in read(ROOT/(extra_base+'manifest.json'))['rows']]
    assert {r['sequence'] for r in belt}.isdisjoint({r['sequence'] for r in belt_val})
    group_sets = [{r['recording_group'] for r in rows} for rows in [foreign, calibration, common]]
    assert all(group_sets[i].isdisjoint(group_sets[j]) for i in range(3) for j in range(i+1,3))
    assert {r['recording_group'] for r in extra} <= group_sets[2]
    dacl, classes = dacl_items('val', 640); assert classes == list(FACILITY_CLASSES)
    old_split = read(ROOT/'runs/facility-presence-target-roi-control/SPLIT.json')
    validation = {'dacl':dacl, 'damsegment':old_split['val'], 'codebrim':read(ROOT/'data/codebrim-training/val.json')['items']}
    assert {r['image'] for r in facility}.isdisjoint({r['image'] for rows in validation.values() for r in rows})
    pools = {'facility_surface':facility, 'belt_damage':belt, 'conveyor_foreign':foreign}
    rng = np.random.default_rng(271); draws = {key:[[] for _ in range(EPOCHS)] for key in pools}
    for domain, count in [('dacl',358),('damsegment',52),('codebrim',102)]:
        pool = [i for i,r in enumerate(facility) if r['domain'] == domain]
        for e, row in enumerate(cycling(pool,count,EPOCHS,rng)): draws['facility_surface'][e].extend(row)
    draws['belt_damage'] = cycling(list(range(len(belt))),1024,EPOCHS,rng)
    for label in [0,1]:
        pool = [i for i,r in enumerate(foreign) if r['label'] == label]
        for e,row in enumerate(cycling(pool,512,EPOCHS,rng)): draws['conveyor_foreign'][e].extend(row)
    for key in draws:
        for row in draws[key]: rng.shuffle(row)
    available = {key:len(rows) for key,rows in pools.items()}
    used = {key:len({i for row in draws[key] for i in row}) for key in pools}
    np.savez_compressed(DATA/'draws.npz', **{k:np.array(v,dtype=np.int64) for k,v in draws.items()})
    document = {'schema':'unified-main-partial-truth-v1', 'classes':list(FACILITY_CLASSES), 'fit':pools,
        'facility_known_validation':validation, 'belt_known_validation':belt_val,
        'foreign_calibration':calibration, 'foreign_known_test':common, 'foreign_known_extra_test':extra,
        'available_fit_records':available, 'unique_fit_records_scheduled':used,
        'draws_per_epoch':{'facility_surface':512,'belt_damage':1024,'conveyor_foreign':1024},
        'epochs':EPOCHS, 'draws_sha256':sha(DATA/'draws.npz'), 'no_new_labels':True,
        'other_tasks_unknown_not_negative':True, 'same_belt_normal_photo_count':0,
        'all_evaluation_photos_are_known_reused_benchmarks':True, 'independent_factory_verified':False,
        'facility_main_initialization':'runs/facility-presence-target-roi-control/best.pt',
        'sources':['DACL10k','DamSegment','CODEBRIM','BeltCrack','Mendeley-s25x2bnshz'],
        'IDEA_CC_BY_NC_ND_supplement_not_used':True, 'PCB_not_in_this_main_cycle':True}
    with (DATA/'manifest.json').open('x',encoding='utf-8') as out: json.dump(document,out,indent=2)
    print(json.dumps({'available':available,'actually_scheduled_unique':used,'validation':{k:len(v) for k,v in validation.items()},
                      'belt_val':len(belt_val),'foreign_CAL':len(calibration),'foreign_tests':[len(common),len(extra)]}),flush=True)


if __name__ == '__main__': main()
