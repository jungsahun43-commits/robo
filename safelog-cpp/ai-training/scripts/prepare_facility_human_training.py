"""Validate a real user submission and produce a separate TRAIN overlay."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import json

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from safelog_ai.human_training_feedback import build_overlay
from safelog_ai.review_feedback import read_json
from scripts.prepare_facility_human_review import write_new


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package",type=Path,required=True);parser.add_argument("--submission",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    root=ROOT.resolve();output=args.output.resolve()
    if not output.is_relative_to((root/"runs").resolve())or output.exists():raise ValueError("Choose a fresh local output directory in runs")
    package=read_json(args.package);submission=read_json(args.submission)
    core=read_json(root/"data/facility-spatial-training/train.json")
    overlay,stats=build_overlay(submission,package,core)
    # Bind and verify original photos before accepting new user labels.
    from scripts.train_facility_target import sha
    for case in package["cases"]:
        path=(root/case["image"]).resolve()
        if not path.is_relative_to(root/"data")or sha(path)!=case["image_sha256"]:raise ValueError("Original review photo changed")
    output.mkdir()
    write_new(output/"train-overlay.json",overlay);write_new(output/"overlay-summary.json",stats)
    print(json.dumps({"status":"human_overlay_prepared","stats":stats,"actual_new_training_epochs":0},ensure_ascii=False))


if __name__=="__main__":main()
