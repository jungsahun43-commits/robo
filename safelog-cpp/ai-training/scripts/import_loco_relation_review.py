"""Validate exported observations into a pending queue, never training GT."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safelog_ai.loco_relation_review import import_review

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--review', required=True)
    p.add_argument('--package', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    print(import_review(args.review, args.package, args.output))
