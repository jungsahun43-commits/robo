"""Photo presence and pixel localization remain separate belt measurements."""
from .equipment_anomaly import classification_rates


def summarize(observations):
    if not observations:
        raise ValueError('An evaluation needs observations')
    counts = {name: sum(r['pixels'][name] for r in observations) for name in ('tp', 'fp', 'tn', 'fn')}
    tp, fp, tn, fn = (counts[k] for k in ('tp', 'fp', 'tn', 'fn'))
    photo = classification_rates([int(r['photo_positive']) for r in observations],
                                 [r['score'] for r in observations], 0.5)
    return {'photos': len(observations), 'sequences': len({r['sequence'] for r in observations}),
        'pixel_counts': counts, 'pixel_iou': tp/(tp+fp+fn) if tp+fp+fn else None,
        'pixel_dice': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        'pixel_fnr': fn/(tp+fn) if tp+fn else None, 'pixel_fpr': fp/(fp+tn) if fp+tn else None,
        'photo_rates': photo,
        'positive_photos_without_any_gt_overlap': sum(r['photo_positive'] and r['pixels']['tp']==0 for r in observations),
        'photo_fpr_fnr_below_5_percent': photo['fpr'] is not None and photo['fnr'] is not None
            and photo['fpr'] < 0.05 and photo['fnr'] < 0.05,
        'photo_score_definition': 'Maximum sigmoid over known, non-padding pixels; not a calibrated probability'}
