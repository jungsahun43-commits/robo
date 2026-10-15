"""One deterministic, label-stratified allocation of eligible acquisition groups."""
import hashlib
import math
from collections import defaultdict


def allocate(rows):
    """Rows already exclude TEST-linked groups and filename suffix variants."""
    labels = defaultdict(set)
    for row in rows:
        if row['publisher_role'] != 'train' or not row['is_filename_base']:
            raise ValueError('Only eligible original TRAIN candidates can be split')
        labels[str(row['group_id'])].add(int(row['label']))
    strata = defaultdict(list)
    for group, values in labels.items():
        strata[tuple(sorted(values))].append(group)
    result = {}
    for groups in strata.values():
        ordered = sorted(groups, key=lambda g: (hashlib.sha256(('insplad44:' + g).encode()).hexdigest(), g))
        n = math.ceil(.15 * len(ordered)) if len(ordered) >= 3 else 0
        for i, group in enumerate(ordered):
            result[group] = 'validation' if i < n else 'cal' if i < 2*n else 'fit'
    return result
