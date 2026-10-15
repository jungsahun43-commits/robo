"""Publisher filename/condition semantics, before any image-based estimation."""
from datetime import datetime
from pathlib import PurePosixPath
import re

CONDITIONS = {
    'yoke-suspension': {'good': 0, 'rust': 1},
    'glass-insulator': {'good': 0, 'missing-cap': 1},
    'polymer-insulator-upper-shackle': {'normal': 0, 'good': 0, 'corrosão': 1, 'rust': 1},
    'lightning-rod-suspension': {'good': 0, 'rust': 1},
    'vari-grip': {'good': 0, 'rust': 1, 'bird-nest': 1},
}


def parse_member(member):
    path = PurePosixPath(member)
    if (path.is_absolute() or '..' in path.parts or '\\' in member or ':' in member
            or len(path.parts) != 5 or path.parts[0] != 'defect_supervised'
            or path.suffix.lower() not in ('.jpg', '.jpeg', '.png')):
        raise ValueError('Only original supervised photo entries supported')
    _, asset, role, condition, _ = path.parts
    if role not in ('train', 'val') or asset not in CONDITIONS or condition not in CONDITIONS[asset]:
        raise ValueError('Unknown publisher asset, role or condition')
    found = re.findall(r'(\d{2}-\d{2}-\d{4})_DJI_(\d{4})', path.name)
    if len(found) != 1:
        raise ValueError('Unresolved acquisition frame; quarantine rather than guess')
    date, number = found[0]
    datetime.strptime(date, '%d-%m-%Y')
    return {'asset': asset, 'publisher_role': role, 'condition': condition,
            'label': CONDITIONS[asset][condition], 'frame_proxy': date + '_DJI_' + number,
            'reserved_evaluation': role == 'val'}


def filename_base_proxy(member, stems_in_same_parent):
    """Suffix evidence only: does not certify independent natural photo count."""
    name = PurePosixPath(member).stem
    while ('_' in name and name.rsplit('_', 1)[1].isdigit()
           and name.rsplit('_', 1)[0] in stems_in_same_parent):
        name = name.rsplit('_', 1)[0]
    return name


def annotation_binary_condition(asset, path_and_index):
    # Numeric class ID is multi-class: vari-grip good=1, bird-nest=0, rust=2.
    # Use the actual publisher class path, not numeric ID>0 as an alarm.
    relative, index = path_and_index.rsplit(maxsplit=1)
    condition = PurePosixPath(relative).parts[0]
    if asset not in CONDITIONS or condition not in CONDITIONS[asset]:
        raise ValueError('Unknown author label name')
    return relative, int(index), CONDITIONS[asset][condition]
