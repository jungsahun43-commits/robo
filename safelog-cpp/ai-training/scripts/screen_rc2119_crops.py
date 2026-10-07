"""Conservative SIFT crop/recompression candidate screen, without model scores.

Feature matches are potential shared views, not proof of identical scenes.
Detected groups are excluded; absence of matches cannot establish independence.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fetch_rc2119 import sha, write_new
from scripts.audit_rc2119 import existing_inventory, read

POLICY = {"feature": "OpenCV SIFT", "max_features": 96, "long_edge": 640,
          "global_flann_trees": 4, "global_flann_checks": 64, "nearest_neighbors": 2,
          "descriptor_ratio_max": .75, "minimum_tentative_matches": 8,
          "ransac_reprojection_pixels": 4.0, "minimum_inliers": 6,
          "minimum_inlier_fraction": .5, "all_matches_excluded_as_candidates": True,
          "heldout_labels_or_predictions_used": False, "new_training_epochs": 0}


def features(path):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("L")
        image.thumbnail((POLICY["long_edge"], POLICY["long_edge"]), Image.Resampling.LANCZOS)
        array = np.asarray(image)
    sift = cv2.SIFT_create(nfeatures=POLICY["max_features"])
    keys, values = sift.detectAndCompute(array, None)
    if values is None:
        return np.empty((0, 2), np.float32), np.empty((0, 128), np.float32)
    keys = keys[:POLICY["max_features"]]; values = values[:len(keys)]
    return np.asarray([k.pt for k in keys], np.float32), np.asarray(values, np.float32)


def geometric_candidate(query_points, train_points, matches):
    if len(matches) < POLICY["minimum_tentative_matches"]:
        return None
    a = np.asarray([query_points[i] for i, j in matches], np.float32)
    b = np.asarray([train_points[j] for i, j in matches], np.float32)
    if len(np.unique(a, axis=0)) < 8 or len(np.unique(b, axis=0)) < 8:
        return None
    matrix, mask = cv2.findHomography(a, b, cv2.RANSAC, POLICY["ransac_reprojection_pixels"], maxIters=2000, confidence=.995)
    if matrix is None or mask is None or not np.isfinite(matrix).all():
        return None
    count = int(mask.sum())
    if count < POLICY["minimum_inliers"] or count / len(matches) < POLICY["minimum_inlier_fraction"]:
        return None
    return {"tentative_matches": len(matches), "inliers": count, "inlier_fraction": count / len(matches)}


def main():
    base = ROOT / "data/rc2119-training"
    final = base / "crop-review.json"
    if final.exists(): raise ValueError("Preserve completed crop screen")
    cv2.setNumThreads(1); cv2.setRNGSeed(61)
    _, _, previous_proof = existing_inventory()
    cache = read(ROOT / "data/convid-training/existing-fingerprints.json")
    old = list(cache["image_sizes"])
    old += [f"data/convid-training/originals/{r['id']}.jpg" for r in read(ROOT / "data/convid-training/source-audit.json")["items"]]
    if len(old) != len(set(old)) or any(not (ROOT / p).is_file() for p in old):
        raise ValueError("Existing photo fingerprint path inventory incomplete")
    audited = read(base / "source-audit.json")
    new = [r["image"] for r in audited["items"]]
    points, descriptions, lengths = [], [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, (xy, desc) in enumerate(pool.map(features, (ROOT / p for p in old)), 1):
            points.append(xy); descriptions.append(desc); lengths.append(len(desc))
            if i % 1000 == 0: print(f"Old photo SIFT extraction {i}/{len(old)}", flush=True)
    matrix = np.concatenate(descriptions, axis=0)
    del descriptions
    offsets = np.r_[0, np.cumsum(lengths)]
    if matrix.dtype != np.float32 or not np.isfinite(matrix).all(): raise ValueError("Malformed SIFT descriptors")
    print(f"Indexing {len(matrix)} native feature descriptors", flush=True)
    index = cv2.FlannBasedMatcher(dict(algorithm=1, trees=POLICY["global_flann_trees"]), dict(checks=POLICY["global_flann_checks"]))
    index.add([matrix]); index.train()
    reviews = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, (xy, desc) in enumerate(pool.map(features, (ROOT / p for p in new)), 1):
            votes = defaultdict(list)
            if len(desc):
                nearest = index.knnMatch(desc, k=2)
                for pair in nearest:
                    if len(pair) != 2 or pair[0].distance >= POLICY["descriptor_ratio_max"] * pair[1].distance:
                        continue
                    match = pair[0]
                    owner = int(np.searchsorted(offsets, match.trainIdx, side="right") - 1)
                    votes[owner].append((match.queryIdx, int(match.trainIdx - offsets[owner])))
            candidates = []
            for owner, pairs in sorted(votes.items()):
                proof = geometric_candidate(xy, points[owner], pairs)
                if proof:
                    candidates.append({"existing_image": old[owner], **proof})
            reviews.append({"image": new[i - 1], "candidate_shared_views": candidates})
            if i % 100 == 0: print(f"New photo crop screen {i}/{len(new)}; detected {sum(bool(r['candidate_shared_views']) for r in reviews)}", flush=True)
    flagged = {r["image"] for r in reviews if r["candidate_shared_views"]}
    # Preserve conservative whole-photo group closure when a member matches.
    groups = {r["group"] for r in audited["items"] if r["image"] in flagged}
    excluded = {r["image"] for r in audited["items"] if r["group"] in groups}
    remaining = [r for r in audited["items"] if not r["excluded_reasons"] and r["image"] not in excluded]
    public = {"schema": "rc2119_crop_candidate_screen_v1", "policy": POLICY, "opencv_version": cv2.__version__,
        "existing_photo_records": len(old), "existing_descriptors": len(matrix), "new_photos_examined": len(new),
        "direct_feature_candidate_photos": len(flagged), "group_closed_candidate_photos": len(excluded),
        "additional_exclusions_after_whole_photo_screen": sum(not r["excluded_reasons"] and r["image"] in excluded for r in audited["items"]),
        "remaining_photos": len(remaining), "remaining_spalling_positive_photos": sum("Concrete spalling" in r["native_instances"] for r in remaining),
        "remaining_near_group_representatives": len({r["group"] for r in remaining}),
        "annotation_audit_sha256": sha(base / "source-audit.json"), "previous_inventory": previous_proof,
        "limitations": ["Approximate descriptor retrieval and limited features can miss shared views",
                        "A homography match is a review candidate, not established scene identity",
                        "No new source/scene independence or annotation completeness is established"],
        "train_manifest_created": False, "new_training_epochs": 0}
    write_new(final, {"summary": public, "items": reviews, "excluded_images": sorted(excluded)})
    public["private_crop_review_sha256"] = sha(final)
    write_new(ROOT / "reports/facility-rc2119-crop-audit.json", public)
    print(json.dumps({k: v for k, v in public.items() if k != "previous_inventory"}), flush=True)


if __name__ == "__main__":
    main()
