import unittest
from safelog_ai.insplad_split import allocate


def order_independence_and_group_integrity():
    rows = [{'publisher_role':'train', 'is_filename_base':True, 'group_id':str(i), 'label':label}
            for i in range(40) for label in (0, 1)]
    result = allocate(rows)
    assert result == allocate(list(reversed(rows)))
    assert list(result.values()).count('validation') == 6
    assert list(result.values()).count('cal') == 6
    assert list(result.values()).count('fit') == 28


def each_label_stratum_and_sparse_groups():
    rows = [{'publisher_role':'train', 'is_filename_base':True, 'group_id':f'{label}/{i}', 'label':label}
            for label in (0, 1) for i in range(20)]
    result = allocate(rows)
    for label in (0, 1):
        assert sum(p == 'validation' for g,p in result.items() if g.startswith(str(label))) == 3
    assert allocate(rows[:1]) == {'0/0':'fit'}


def holdout_and_variant_rejected():
    with unittest.TestCase().assertRaises(ValueError):
        allocate([{'publisher_role':'val', 'is_filename_base':True, 'group_id':'a', 'label':0}])
    with unittest.TestCase().assertRaises(ValueError):
        allocate([{'publisher_role':'train', 'is_filename_base':False, 'group_id':'a', 'label':0}])


class InspladSplitTests(unittest.TestCase):
    test_order_independence = staticmethod(order_independence_and_group_integrity)
    test_strata = staticmethod(each_label_stratum_and_sparse_groups)
    test_reservations = staticmethod(holdout_and_variant_rejected)
