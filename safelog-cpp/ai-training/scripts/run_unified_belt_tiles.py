"""Operational entry point with a read-only Windows lock probe; neural code frozen."""
import argparse
import os
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import torch
from scripts import study_unified_belt_tiles as study
from safelog_ai.process_probe import probe_only


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['prepare', 'freeze', 'train', 'evaluate', 'select', 'verify'])
    args = parser.parse_args(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    # Only the study's local os binding changes; global os and signal APIs stay intact.
    study.os = SimpleNamespace(getpid=os.getpid, kill=probe_only)
    getattr(study, args.mode)()
