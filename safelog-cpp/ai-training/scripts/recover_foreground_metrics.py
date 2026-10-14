"""Repair only abstention counts; preserve frozen scorer and CAL thresholds."""
import argparse
from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import study_industrial_foreground as study
from scripts.prepare_equipment_expansion import new
from scripts.fetch_equipment_expansion import sha

RECORD = study.REPORTS/'industrial-foreground-v12-metric-recovery.json'
SOURCES = ('scripts/recover_foreground_metrics.py', 'tests/test_foreground_metric_recovery.py')


def worst_case_rates(labels, scores, threshold):
    if len(labels) != len(scores) or not math.isfinite(threshold) or any(v not in (0, 1) for v in labels):
        raise ValueError('Matching binary labels and finite fixed threshold required')
    tp = fp = fn = tn = 0
    for label, score in zip(labels, scores):
        if not math.isfinite(score):
            # Frozen study uses +/-infinity only as symbols for missing outputs.
            # Never convert these into actual model scores or healthy outputs.
            if score != (-float('inf') if label else float('inf')):
                raise ValueError('Unrecognised missing-score representation')
            if label: fn += 1
            else: fp += 1
        elif label:
            if score > threshold: tp += 1
            else: fn += 1
        else:
            if score > threshold: fp += 1
            else: tn += 1
    if not tp+fn or not fp+tn: raise ValueError('Both classes required')
    return {'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn, 'positive_photos': tp+fn, 'negative_photos': fp+tn,
            'fnr': fn/(tp+fn), 'fpr': fp/(fp+tn)}


def declare():
    study.validate(); assert not RECORD.exists()
    failures = list(study.RUN.glob('evaluate-failure-*.json'))
    assert len(failures) == 1
    assert 'Finite equally sized binary labels and scores required' in study.read(failures[0])['error']
    new(RECORD, {'declared_utc': datetime.now(timezone.utc).isoformat(), 'failure': failures[0].name, 'failure_sha256': sha(failures[0]),
        'source_sha256': {n: sha(ROOT/n) for n in SOURCES}, 'frozen_protocol_sha256': sha(study.PROTOCOL),
        'normal_bank_sha256': sha(study.RUN/'normal-banks.pt'), 'normal_CAL_sha256': sha(study.RUN/'normalCAL.json'),
        'repair': 'count missing normal as FP and missing defect as FN directly, same preregistered worst-case rule; finite-score helper rejects symbolic infinities',
        'old_frozen_source_edited': False, 'model_parameters_views_masks_weights_or_thresholds_changed': False,
        'first_partial_TEST_inference_already_executed_before_aggregation_failure': True,
        'partial_predictions_not_persisted_by_failed_process': True, 'same_fixed_inference_repeated_for_error_recovery_only': True,
        'actual_optimizer_updates': 0, 'new_unit_tests_passed': 2})


def evaluate():
    p = study.read(RECORD); study.validate(committed=True)
    for name, digest in p['source_sha256'].items(): assert sha(ROOT/name) == digest
    for name, digest in (('normal-banks.pt', p['normal_bank_sha256']), ('normalCAL.json', p['normal_CAL_sha256'])): assert sha(study.RUN/name) == digest
    assert sha(study.PROTOCOL) == p['frozen_protocol_sha256']
    repo = ROOT.parents[1]; git = ['C:/Program Files/Git/cmd/git.exe', '-c', 'safe.directory='+repo.as_posix(), '-C', str(repo)]
    for name in (*SOURCES, RECORD.relative_to(ROOT).as_posix()):
        assert hashlib.sha256(subprocess.check_output(git+['show', 'HEAD:safelog-cpp/ai-training/'+name])).hexdigest() == sha(ROOT/name)
    lock = study.RUN/'operation.lock'
    assert not lock.exists()
    with lock.open('x', encoding='utf-8') as out: out.write(str(os.getpid()))
    study.classification_rates = worst_case_rates
    try: study.evaluate()
    except BaseException as error:
        new(study.RUN/('recovery-failure-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'), {'error': repr(error)})
        raise
    finally: lock.unlink()


if __name__ == '__main__':
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['declare', 'evaluate']); args = parser.parse_args()
    globals()[args.mode]()
