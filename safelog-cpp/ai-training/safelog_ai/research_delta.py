"""Lossless research state overlay, cryptographically bound to an immutable parent."""
from copy import deepcopy
import re
import torch
from .retention_distillation import state_sha256


def pack_delta(parent,candidate,parent_sha256,parent_relative_path,allowed_prefixes):
    if not re.fullmatch('[0-9a-f]{64}',parent_sha256):raise ValueError('Exact SHA256 required')
    parts=parent_relative_path.replace('\\','/').split('/')
    if parent_relative_path.startswith(('/','\\')) or any(p in ('','..') or ':' in p for p in parts):raise ValueError('Safe workspace relative parent path required')
    if not allowed_prefixes or any(not p for p in allowed_prefixes):raise ValueError('Explicit added namespace required')
    original,new=parent['state_dict'],candidate['state_dict']
    for name,value in original.items():
        if name not in new or not torch.equal(value,new[name]):raise ValueError('Original parent tensor changed: '+name)
    for name,value in parent.items():
        if name!='state_dict' and (name not in candidate or value!=candidate[name]):raise ValueError('Original metadata changed: '+name)
    extra={n:v.detach().cpu().clone() for n,v in new.items() if n not in original}
    if not extra or any(not n.startswith(tuple(allowed_prefixes)) for n in extra):raise ValueError('Only declared added tensors allowed')
    return {'schema':'safelog.research-delta.v1','parent_relative_path':'/'.join(parts),'parent_sha256':parent_sha256,
        'parent_state_sha256':state_sha256(original),'allowed_prefixes':list(allowed_prefixes),
        'added_state_dict':extra,'added_state_sha256':state_sha256(extra),
        'added_metadata':deepcopy({n:v for n,v in candidate.items() if n not in parent}),
        'combined_state_sha256':state_sha256(new)}


def restore_delta(parent,payload,actual_parent_sha256):
    if payload.get('schema')!='safelog.research-delta.v1' or actual_parent_sha256!=payload['parent_sha256']:raise ValueError('Wrong parent identity')
    if state_sha256(parent['state_dict'])!=payload['parent_state_sha256']:raise ValueError('Parent state mismatch')
    extra=payload['added_state_dict']
    if state_sha256(extra)!=payload['added_state_sha256']:raise ValueError('Overlay state mismatch')
    if set(extra)&set(parent['state_dict']) or any(not n.startswith(tuple(payload['allowed_prefixes'])) for n in extra):raise ValueError('Protected parent overlap')
    if set(payload['added_metadata'])&set(parent):raise ValueError('Protected metadata overlap')
    result=deepcopy(parent);result['state_dict'].update({n:t.clone() for n,t in extra.items()});result.update(deepcopy(payload['added_metadata']))
    if state_sha256(result['state_dict'])!=payload['combined_state_sha256']:raise ValueError('Combined state mismatch')
    return result
