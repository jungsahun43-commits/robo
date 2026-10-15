"""Deterministic label-stratified allocation of intact publisher TRAIN groups."""
from collections import Counter, defaultdict
import hashlib
import math


def stratified_allocation(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row['phase'] not in ('fit', 'validation', 'cal'):
            raise ValueError('Only audited nonTEST candidate rows may be allocated')
        if row['label'] not in (0, 1): raise ValueError('Binary author acceptance label required')
        grouped[row['group_id']].append(row)
    if len(grouped) < 14: raise ValueError('At least14 intact TRAIN groups required')
    n = len(grouped); small = math.ceil(.15 * n)
    capacities = {'fit': n - 2 * small, 'validation': small, 'cal': small}
    fractions = {'fit': .70, 'validation': .15, 'cal': .15}
    total = Counter(r['label'] for r in rows)
    if not all(total[y] for y in (0, 1)): raise ValueError('Both author classes required')
    counts = {p: Counter() for p in capacities}; assigned = Counter(); result = {}
    order = sorted(grouped, key=lambda g: (-len(grouped[g]), hashlib.sha256(('gearbox26:' + g).encode()).hexdigest(), g))
    for g in order:
        labels = Counter(r['label'] for r in grouped[g])
        def delta(p):
            return sum(((counts[p][y] + labels[y]) / total[y] - fractions[p]) ** 2
                       - (counts[p][y] / total[y] - fractions[p]) ** 2 for y in (0, 1))
        phase = min((p for p in capacities if assigned[p] < capacities[p]),
                    key=lambda p: (delta(p), list(capacities).index(p)))
        result[g] = phase; counts[phase].update(labels); assigned[phase] += 1
    assert dict(assigned) == capacities
    return result
