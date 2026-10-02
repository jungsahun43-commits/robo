"""Allow only an audited TRAIN crop replacement over the immutable core manifest."""
from pathlib import PurePosixPath

RECIPE = 'small_source_region_context_v1'
LOCAL_PREFIX = PurePosixPath('data/facility-small-region-training')


def validate_replacement(core, candidate, core_sha256):
    audit = candidate.get('audit', {})
    if (core['split'] != 'train' or candidate.get('split') != 'train'
            or candidate.get('classes') != core['classes']
            or candidate.get('full_count') != core['full_count']
            or len(candidate.get('items', [])) != len(core['items'])
            or audit.get('status') != 'prepared' or audit.get('recipe') != RECIPE
            or audit.get('source_manifest_sha256') != core_sha256):
        raise ValueError('Crop replacement must preserve an audited core TRAIN manifest')
    n = core['full_count']
    if candidate['items'][:n] != core['items'][:n]:
        raise ValueError('Full TRAIN image order, labels and spatial targets must remain unchanged')
    parents = {row['image']: row for row in core['items'][:n]}
    replaced_parents = set()
    changed = 0
    for old, new in zip(core['items'][n:], candidate['items'][n:]):
        if old == new:
            continue
        changed += 1
        parent = parents.get(new.get('parent_image'))
        if (new.get('replacement_of_image') != old['image']
                or new.get('parent_image') != old['parent_image']
                or new.get('parent_split') != 'train'
                or new.get('domain') != old['domain']
                or parent is None or parent['domain'] not in ('dacl', 'damsegment')
                or new['parent_image'] in replaced_parents):
            raise ValueError('Replacement must belong to its original TRAIN parent, at most once')
        replaced_parents.add(new['parent_image'])
        for field in ('image', 'pixel_target'):
            raw_path = new.get(field, '')
            if not isinstance(raw_path, str) or '\\' in raw_path:
                raise ValueError('Derived paths must use relative POSIX separators, including on Windows')
            path = PurePosixPath(raw_path)
            if path.is_absolute() or '..' in path.parts or not path.is_relative_to(LOCAL_PREFIX):
                raise ValueError('Derived crops and targets must stay inside the ignored local data directory')
        target = new.get('targets', [])
        if len(target) != len(core['classes']) or any(t not in (-1, 0, 1) for t in target):
            raise ValueError('Replacement needs seven asserted/unknown source labels')
        if any((p == -1 and t != -1) or (p == 0 and t == 1) for p, t in zip(parent['targets'], target)):
            raise ValueError('Crop cannot invent unknown/absent parent labels')
        if not any(old['targets'][k] == 1 and target[k] == 1 for k in (0, 1)):
            raise ValueError('Replace only an existing target-positive TRAIN detail crop')
    if changed < 1 or changed != audit.get('replaced_rows'):
        raise ValueError('Recorded replacement count must match changed TRAIN rows')
    return candidate
