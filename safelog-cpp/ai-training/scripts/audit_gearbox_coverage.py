"""Label/part coverage audit only; never change heldout membership or model scores."""
from collections import Counter
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_gearbox_stratified import MANIFEST
from scripts.fetch_equipment_expansion import sha
from scripts.study_equipment_expansion import read
from scripts.prepare_equipment_expansion import new


def main():
    rows=read(MANIFEST)['rows'];categories=sorted({r['functional_component_category'] for r in rows})
    counts={phase:{category:{str(y):sum(r['phase']==phase and r['functional_component_category']==category and r['label']==y for r in rows) for y in (0,1)} for category in categories} for phase in ('fit','validation','cal','test')}
    result={'manifest_sha256':sha(MANIFEST),'counts_by_functional_category_and_label':counts,
        'FIT_has_no_synchronizer_collar':sum(counts['fit']['synchronizer_collar'].values())==0,
        'remaining_nonTEST_spline_OK_total':sum(counts[p]['spline']['0'] for p in ('fit','validation','cal')),
        'interpretation':'Binary-label balance does not ensure category coverage. v26/v27 VAL/CAL/TEST include collar absent in FIT. Spline has only3 OK in all retained nonTEST. Preserve primary fullTEST and strict guards; do not silently omit unsupported types to claim broad pass.',
        'next_data_design':'Before another training, preregister category-by-label coverage gate and group allocation using only original eligible TRAIN labels; broad fivecategory training may be impossible with spline OK3. A separately declared gear-only task must retain original failure and report its narrower scope; no reassignment of TEST.',
        'model_or_threshold_changed':False,'TEST_scoring_used_for_counts':False}
    new(ROOT/'reports/industrial-gearbox-category-coverage-v26-v27.json',result)
    print(result,flush=True)


if __name__=='__main__':main()
