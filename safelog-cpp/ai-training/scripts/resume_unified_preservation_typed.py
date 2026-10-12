"""Finish fixed fits with explicit, membership-checked CAL provenance."""
import argparse
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import study_unified_preservation_typed as typed
from scripts.unified_main_data import DATA,read,sha


def finalize_threshold(rows):
    from safelog_ai.foreign_calibration import fit_threshold
    manifest=read(DATA/'manifest.json');allowed=manifest['foreign_calibration']
    if [r['id'] for r in rows]!=[r['id'] for r in allowed] or [r['label'] for r in rows]!=[r['label'] for r in allowed]:
        raise ValueError('Threshold inputs must be exactly the fixed CAL320 IDs and author labels in order')
    if len(rows)!=320 or any(r['split']!='calibration' for r in allowed):
        raise ValueError('Only declared calibration source records may be tagged calibration')
    forbidden={r['id'] for key in ['foreign_known_test','foreign_known_extra_test'] for r in manifest[key]}
    if any(r['id'] in forbidden for r in rows):raise ValueError('TEST cannot fit a threshold')
    return fit_threshold([{**r,'split':'calibration'} for r in rows])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=typed.study.VARIANTS)
    args=parser.parse_args();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    typed.activate();fix=read(ROOT/'reports/unified-preservation-calibration-provenance-fix.json')
    assert fix['resume_source_sha256']==sha(Path(__file__)) and fix['neural_loss_or_sampling_changed'] is False
    typed.study.fit_threshold=finalize_threshold
    for variant in ([args.variant] if args.variant else typed.study.VARIANTS):typed.study.train_variant(variant)
