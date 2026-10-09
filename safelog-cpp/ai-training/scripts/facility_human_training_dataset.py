"""Original augmentation with explicit human-unknown masks and bounded draws."""
from __future__ import annotations
import numpy as np
from scripts.facility_resolution_study import ResolutionPhotos
from safelog_ai.auxiliary_classifier import AUX_CLASSES


class HumanReviewPhotos(ResolutionPhotos):
    def __getitem__(self,index):
        row=list(super().__getitem__(index));item=self.items[index]
        if item.get("human_pixel_unknown_columns"):
            row[5]=row[5].clone()
            for k in item["human_pixel_unknown_columns"]:row[5][k]=0
        if item.get("human_auxiliary_unknown_tags"):
            row[7]=row[7].clone()
            for tag in item["human_auxiliary_unknown_tags"]:row[7][AUX_CLASSES.index(tag)]=0
        return tuple(row)


def human_exposure_draws(original,items,full_count,verified,seed=73):
    """Add at most2 selected full-photo exposures per epoch and reviewed photo.

Every replaced position preserves its original source and full-photo status.
Other positions and all crop draw positions remain unchanged. This is a
human-label+bounded-exposure package, not an isolated label-only comparison.
    """
    if original.dtype!=np.int64 or original.shape!=(6,14248):raise ValueError("Original six-epoch draw budget required")
    if original.min()<0 or original.max()>=len(items):raise ValueError("Draw leaves original TRAIN rows")
    full={row["image"]:i for i,row in enumerate(items[:full_count])}
    if not verified or any(image not in full for image in verified):raise ValueError("Only verified original full TRAIN rows are eligible")
    pools={}
    for image,columns in verified.items():
        index=full[image]
        if not columns or any(k not in(0,1)or items[index]["targets"][k]not in(0,1)for k in columns):raise ValueError("Actual definite human primary labels required")
        pools.setdefault(items[index]["domain"],[]).append(index)
    result=original.copy();rng=np.random.default_rng(seed);counts=[]
    domains=np.asarray([r["domain"]for r in items])
    for epoch,draws in enumerate(original):
        tally={}
        for domain,pool in sorted(pools.items()):
            positions=np.flatnonzero((draws<full_count)&(domains[draws]==domain))
            additions=np.tile(np.asarray(sorted(pool),dtype=np.int64),2)
            if len(additions)>len(positions):raise ValueError("Human exposure exceeds unchanged source/full-photo budget")
            selected=rng.choice(positions,len(additions),replace=False);rng.shuffle(additions)
            result[epoch,selected]=additions;tally[domain]=len(additions)
        counts.append(tally)
    return result,counts
