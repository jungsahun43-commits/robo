"""Fit a decision threshold from calibration labels only, never test labels."""
import math
import numpy as np


def fit_threshold(rows):
    if not rows or any(r.get('split')!='calibration' for r in rows):
        raise ValueError('Only explicitly designated calibration rows may fit the threshold')
    y = np.array([r['label'] for r in rows])
    scores = np.array([r['score'] for r in rows],dtype=np.float64)
    if not np.isin(y,[0,1]).all() or not {0,1}<=set(y) or not np.isfinite(scores).all() or (scores<0).any() or (scores>1).any():
        raise ValueError('Both classes and finite sigmoid scores between zero and one required')
    unique = np.unique(scores)
    clipped = np.clip(unique,1e-12,1-1e-12)
    logits = np.log(clipped)-np.log1p(-clipped)
    middle = 1/(1+np.exp(-(logits[:-1]+logits[1:])/2))
    candidates = np.r_[unique[0]/2,middle,(1+unique[-1])/2]
    def objective(threshold):
        pred = scores>threshold
        fpr = np.mean(pred[y==0]);fnr = np.mean(~pred[y==1])
        return max(fpr,fnr),fpr+fnr,abs(threshold-.5),threshold
    best = min((float(t) for t in candidates if 0<t<1),key=objective)
    assert 0<best<1 and math.isfinite(best)
    return {'threshold':best,'method':'Minimize max(FPR,FNR), then sum; logit midpoint candidates; CAL only',
            'calibration_photos':len(rows),'calibration_negative_photos':int((y==0).sum()),
            'calibration_positive_photos':int((y==1).sum()),'calibration_max_error_rate':float(objective(best)[0])}
