"""Explain validation errors by publisher damage size; never read test scores."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_facility_target import read, save, dacl_items, TARGETS
from scripts.prepare_facility_data import DACL_MAPPING


def fractions(record):
    source = ROOT / "data" / record["source"]
    doc = read(source.parent.parent.parent / "annotations/train" / f"{source.stem}.json")
    result = {}
    for label in TARGETS:
        mask = Image.new("1", (640, 640)); draw = ImageDraw.Draw(mask)
        for shape in doc["shapes"]:
            if DACL_MAPPING.get(shape["label"]) == label:
                draw.polygon([(x*640/doc['imageWidth'], y*640/doc['imageHeight']) for x,y in shape['points']], fill=1)
        result[label] = float(np.asarray(mask).mean())
    return record["stem"], result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--name", default="facility-presence-target-v2s")
    parser.add_argument("--aggregate-only", action="store_true", help="Keep individual source filenames in the ignored run; publish counts only")
    args = parser.parse_args(); run = ROOT / "runs" / args.name
    prediction, validation = read(run / "validation-dacl.json"), read(run / "VALIDATION.json")
    if prediction["split"] != "val": raise ValueError("Only validation errors can inform next training")
    items, classes = dacl_items("val", 384)
    target = np.array(prediction["targets"]); score = np.array(prediction["probabilities"])
    if not np.array_equal(target, [i['targets'] for i in items]): raise ValueError("Validation item order changed")
    records = [r for r in read(ROOT / "data/dacl10k-yolo/records.json") if r["split"] == "val"]
    with ThreadPoolExecutor(max_workers=4) as pool: areas = dict(pool.map(fractions, records))
    edges = (0., .001, .01, .05, 1.00001); results = {}
    for label in TARGETS:
        k = classes.index(label); threshold = validation["operating_points"][label]["threshold"]
        positive = target[:,k] == 1; found = score[:,k] >= threshold
        size = np.array([areas[Path(i['image']).stem][label] for i in items])
        bins = []
        for low, high in zip(edges, edges[1:]):
            mask = positive & (size >= low) & (size < high)
            count = int(mask.sum()); misses = int((mask & ~found).sum())
            bins.append({"polygon_fraction": [low, high], "positive_photos": count,
                         "false_negatives": misses, "fnr": misses/count if count else None})
        results[label] = {"threshold": threshold, "positive_size_bins": bins,
                         "fp_count": int((~positive & found).sum()),
                         "fn_count": int((positive & ~found).sum()),
                         "highest_confidence_false_positives": [items[i]['image'] for i in sorted(np.flatnonzero(~positive & found), key=lambda i: -score[i,k])[:10]],
                         "lowest_confidence_false_negatives": [items[i]['image'] for i in sorted(np.flatnonzero(positive & ~found), key=lambda i: score[i,k])[:10]]}
    result = {"run": args.name, "split": "val", "weights_sha256": prediction["weights_sha256"],
              "scope": "DACL validation only, 640x640 rasterized publisher polygons for approximate size analysis; gold labels unchanged", "per_class": results}
    save(run / "ERROR-SIZE-AUDIT.json", result)
    if args.aggregate_only:
        result["per_class"] = {label: {key: value for key, value in row.items()
                                      if key not in ("highest_confidence_false_positives", "lowest_confidence_false_negatives")}
                               for label, row in results.items()}
        result["individual_examples_published"] = False
    save(ROOT / "reports" / f"{args.name}-error-size-audit.json", result)
    print(__import__('json').dumps(result, indent=2))


if __name__ == "__main__": main()
