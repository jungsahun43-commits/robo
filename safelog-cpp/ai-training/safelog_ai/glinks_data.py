"""Publisher acquisition-id proxies and a single predeclared normal-only split."""
import hashlib
import math
from pathlib import PurePosixPath
import re


def publisher_record(member):
    path = PurePosixPath(member)
    if len(path.parts) != 3 or str(path.parent) not in ('train/good', 'test/good', 'test/defect'):
        raise ValueError('Only original publisher image directories permitted')
    match = re.fullmatch(r'([a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12})-(\d{4})\.jpg', path.name)
    if match is None:
        raise ValueError('Expected publisher UUID and frame number')
    return {'publisher_role': path.parts[0], 'label': int(path.parts[1] == 'defect'),
            'acquisition_uuid_proxy': match[1], 'frame_number': int(match[2])}


def normal_group_split(groups):
    ordered = sorted(set(groups), key=lambda group: (hashlib.sha256(('glinks35:' + str(group)).encode()).hexdigest(), str(group)))
    count = math.ceil(.15 * len(ordered))
    return {group: 'validation' if i < count else 'cal' if i < 2 * count else 'fit'
            for i, group in enumerate(ordered)}
