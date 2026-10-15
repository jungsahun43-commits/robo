"""Join frozen source descriptors by identity, never by changing split positions."""
import torch


def select_descriptors(ids,source_ids,features):
    if len(source_ids)!=len(features) or len(set(source_ids))!=len(source_ids) or len(set(ids))!=len(ids):
        raise ValueError('Unique aligned source/target descriptor ids required')
    index={key:i for i,key in enumerate(source_ids)}
    if not set(ids)<=set(index):raise ValueError('Requested ids outside permitted source phases')
    return features[torch.tensor([index[key] for key in ids],dtype=torch.long)]
