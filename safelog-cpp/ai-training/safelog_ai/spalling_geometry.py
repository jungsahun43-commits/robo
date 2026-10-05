"""Read-only source raster replay and descriptive image features.

The raster recipe follows ``scripts.prepare_facility_spatial.masks_for``:
DACL polygons are drawn at the processed parent image dimensions with PIL,
then resized to 640 with BOX and thresholded at >0. Dam publisher masks use
exact red/blue RGB matches before the same resize. An 80-cell target is the
any-positive reduction of each 8 by 8 fine block. These source rasters are
annotations, and their unmarked complement is not a healthy-background label.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from scripts.prepare_facility_data import DACL_MAPPING, NAMES


CLASSES = tuple(NAMES)
AMBIGUOUS_TAGS = ("Rockpocket", "WConccor", "Hollowareas", "Cavity")
FINE_SIZE = 640
GRID_SIZE = 80
BLOCK_SIZE = 8


def _bool_array(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.bool_ or value.shape != shape:
        raise ValueError(f"{name} must be a boolean array with shape {shape}")
    return value


def _rgb_array(value: np.ndarray, name: str, *, fine: bool = False) -> np.ndarray:
    if (not isinstance(value, np.ndarray) or value.ndim != 3 or value.shape[2] != 3
            or value.shape[0] <= 0 or value.shape[1] <= 0
            or value.dtype.kind not in "iu"
            or np.any(value < 0) or np.any(value > 255)
            or (fine and value.shape != (FINE_SIZE, FINE_SIZE, 3))):
        extent = "640 by 640 " if fine else "nonempty "
        raise ValueError(f"{name} must be a {extent}integer RGB array in [0,255]")
    return value.astype(np.uint8, copy=False)


def _size(value: Sequence[int], name: str) -> tuple[int, int]:
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) or v <= 0
                   for v in value)):
        raise ValueError(f"{name} must contain two positive integer dimensions")
    return int(value[0]), int(value[1])


def any80_mask(fine_mask: np.ndarray) -> np.ndarray:
    """Return the exact any-positive 8 by 8 reduction of a boolean 640 mask."""
    fine_mask = _bool_array(fine_mask, (FINE_SIZE, FINE_SIZE), "fine mask")
    return fine_mask.reshape(GRID_SIZE, BLOCK_SIZE, GRID_SIZE, BLOCK_SIZE).max((1, 3))


def _fine(mask: Image.Image) -> np.ndarray:
    return np.asarray(mask.resize((FINE_SIZE, FINE_SIZE), Image.Resampling.BOX)) > 0


@dataclass(frozen=True)
class SourceRaster:
    """Raster storage with availability kept separate from an empty mask.

    Unavailable classes have zero storage and ``class_available=False``; those
    zeros never assert an absent source label. A source with no pixel annotation
    has ``annotation_union=None`` and ``source_unmarked=None``.
    """

    source_kind: str
    fine_masks: np.ndarray
    any_masks: np.ndarray
    class_available: np.ndarray
    tag_masks: dict[str, np.ndarray]
    annotation_union: np.ndarray | None
    source_unmarked: np.ndarray | None

    def __post_init__(self) -> None:
        shape = (len(CLASSES), FINE_SIZE, FINE_SIZE)
        _bool_array(self.fine_masks, shape, "fine class masks")
        _bool_array(self.any_masks, (len(CLASSES), GRID_SIZE, GRID_SIZE), "any class masks")
        _bool_array(self.class_available, (len(CLASSES),), "class availability")
        expected = self.fine_masks.reshape(len(CLASSES), GRID_SIZE, BLOCK_SIZE,
                                           GRID_SIZE, BLOCK_SIZE).max((2, 4))
        if not np.array_equal(expected, self.any_masks):
            raise ValueError("Any masks must be the exact any-positive reduction of fine masks")
        if np.any(self.fine_masks[~self.class_available]):
            raise ValueError("Unavailable class masks cannot contain invented positive pixels")
        if not isinstance(self.tag_masks, dict) or any(tag not in AMBIGUOUS_TAGS for tag in self.tag_masks):
            raise ValueError("Unexpected ambiguous source tag")
        for tag, mask in self.tag_masks.items():
            _bool_array(mask, (FINE_SIZE, FINE_SIZE), f"{tag} mask")
        if (self.annotation_union is None) != (self.source_unmarked is None):
            raise ValueError("Source union and unmarked pixels must share availability")
        if self.annotation_union is not None:
            _bool_array(self.annotation_union, (FINE_SIZE, FINE_SIZE), "source annotation union")
            _bool_array(self.source_unmarked, (FINE_SIZE, FINE_SIZE), "source unmarked mask")
            if not np.array_equal(~self.annotation_union, self.source_unmarked):
                raise ValueError("Source unmarked pixels must be the annotation union complement")
            if np.any(self.fine_masks.any(axis=0) & ~self.annotation_union):
                raise ValueError("Class masks must be contained within the annotation union")
            if any(np.any(mask & ~self.annotation_union) for mask in self.tag_masks.values()):
                raise ValueError("Tag masks must be contained within the annotation union")
        elif self.class_available.any() or self.tag_masks:
            raise ValueError("An available source raster requires a source annotation union")


def _source(source_kind: str, fine_masks: np.ndarray, class_available: np.ndarray,
            tag_masks: dict[str, np.ndarray], union: np.ndarray | None) -> SourceRaster:
    coarse = fine_masks.reshape(len(CLASSES), GRID_SIZE, BLOCK_SIZE,
                                GRID_SIZE, BLOCK_SIZE).max((2, 4))
    return SourceRaster(source_kind, fine_masks, coarse, class_available, tag_masks,
                        union, None if union is None else ~union)


def dacl_source_raster(doc: Mapping[str, Any], processed_size: tuple[int, int]) -> SourceRaster:
    """Replay a DACL annotation document at the processed parent dimensions.

    Coordinate scaling, polygon filling, clipping, BOX resize and thresholding
    match the existing producer. Shapes with fewer than three points remain
    undrawn as in that producer. The source union also includes unmapped tags;
    its complement only means unmarked in this source annotation.
    """
    if not isinstance(doc, Mapping):
        raise ValueError("DACL annotation must be a mapping")
    width, height = _size(processed_size, "Processed parent size")
    source_width, source_height = _size((doc.get("imageWidth"), doc.get("imageHeight")),
                                       "DACL annotation size")
    shapes = doc.get("shapes")
    if not isinstance(shapes, list):
        raise ValueError("DACL shapes must be a list")
    masks = [Image.new("L", (width, height)) for _ in CLASSES]
    tag_images = {tag: Image.new("L", (width, height)) for tag in AMBIGUOUS_TAGS}
    union_image = Image.new("L", (width, height))
    for shape in shapes:
        if not isinstance(shape, Mapping) or not isinstance(shape.get("label"), str):
            raise ValueError("DACL shape must contain a string label")
        raw_points = shape.get("points")
        if not isinstance(raw_points, (list, tuple)):
            raise ValueError("DACL polygon points must be a sequence")
        points = []
        for point in raw_points:
            if (not isinstance(point, (list, tuple)) or len(point) != 2
                    or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real)
                           or not math.isfinite(float(v)) for v in point)):
                raise ValueError("DACL polygon coordinates must be finite numeric pairs")
            x, y = point
            scaled = (x * width / source_width, y * height / source_height)
            if not all(math.isfinite(float(v)) for v in scaled):
                raise ValueError("DACL scaled polygon coordinates must be finite")
            points.append(scaled)
        if len(points) < 3:
            continue
        label = shape["label"]
        ImageDraw.Draw(union_image).polygon(points, fill=255)
        mapped = DACL_MAPPING.get(label)
        if mapped is not None:
            ImageDraw.Draw(masks[CLASSES.index(mapped)]).polygon(points, fill=255)
        if label in tag_images:
            ImageDraw.Draw(tag_images[label]).polygon(points, fill=255)
    fine_masks = np.stack([_fine(mask) for mask in masks])
    tags = {tag: _fine(mask) for tag, mask in tag_images.items()}
    union = _fine(union_image)
    return _source("dacl", fine_masks, np.ones(len(CLASSES), dtype=bool), tags, union)


def dam_source_raster(rgb_mask: np.ndarray) -> SourceRaster:
    """Replay exact publisher red crack and blue spalling mask colors."""
    rgb = _rgb_array(rgb_mask, "Dam publisher mask")
    masks = np.zeros((len(CLASSES), FINE_SIZE, FINE_SIZE), dtype=bool)
    for index, color in ((CLASSES.index("concrete_crack"), (255, 0, 0)),
                         (CLASSES.index("concrete_spalling"), (0, 0, 255))):
        native = Image.fromarray(np.uint8(np.all(rgb == color, axis=-1)) * 255)
        masks[index] = _fine(native)
    available = np.zeros(len(CLASSES), dtype=bool)
    available[[CLASSES.index("concrete_crack"), CLASSES.index("concrete_spalling")]] = True
    # Other nonblack publisher colors are source marks without assigning them
    # a model class. They cannot become asserted healthy-background pixels.
    union = _fine(Image.fromarray(np.uint8(np.any(rgb != 0, axis=-1)) * 255))
    return _source("damsegment", masks, available, {}, union)


def unavailable_source_raster() -> SourceRaster:
    """Represent a source such as CODEBRIM that supplies no positive pixel mask."""
    return _source("unavailable", np.zeros((len(CLASSES), FINE_SIZE, FINE_SIZE), dtype=bool),
                   np.zeros(len(CLASSES), dtype=bool), {}, None)


def geometry_metrics(source: SourceRaster, target_index: int = 1) -> dict[str, Any]:
    """Describe source raster area, cell expansion, fill and actual tag overlap."""
    if not isinstance(source, SourceRaster):
        raise ValueError("Geometry requires a SourceRaster")
    if (isinstance(target_index, (bool, np.bool_)) or not isinstance(target_index, Integral)
            or not 0 <= target_index < len(CLASSES)):
        raise ValueError("Target index must identify one model class")
    result: dict[str, Any] = {
        "raster_available": bool(source.class_available[target_index]),
        "fine_area_fraction": None, "coarse_any_area_fraction": None,
        "occupied80cells": None, "expansion_ratio": None,
        "thin_cells_le_eighth": None, "thin_cells_le_eighth_fraction": None,
        "occupied80_cell_fill": None, "ambiguous_tag_overlap": {},
        "annotation_union_area_fraction": None if source.annotation_union is None
            else float(source.annotation_union.mean()),
        "source_unmarked_area_fraction": None if source.source_unmarked is None
            else float(source.source_unmarked.mean()),
    }
    if not result["raster_available"]:
        return result
    fine = source.fine_masks[target_index]
    count = int(fine.sum())
    fill = fine.reshape(GRID_SIZE, BLOCK_SIZE, GRID_SIZE, BLOCK_SIZE).sum((1, 3))
    occupied_fill = fill[fill > 0]
    occupied = int(occupied_fill.size)
    thin = int((occupied_fill <= BLOCK_SIZE * BLOCK_SIZE / 8).sum())
    result.update(
        fine_area_fraction=float(fine.mean()),
        coarse_any_area_fraction=float(source.any_masks[target_index].mean()),
        occupied80cells=occupied,
        expansion_ratio=None if count == 0 else float(occupied * BLOCK_SIZE * BLOCK_SIZE / count),
        thin_cells_le_eighth=thin,
        thin_cells_le_eighth_fraction=None if occupied == 0 else float(thin / occupied),
        occupied80_cell_fill={
            "min": None if occupied == 0 else float(occupied_fill.min() / 64),
            "mean": None if occupied == 0 else float(occupied_fill.mean() / 64),
            "median": None if occupied == 0 else float(np.median(occupied_fill) / 64),
            "max": None if occupied == 0 else float(occupied_fill.max() / 64),
            "histogram_fine_pixels": {str(int(n)): int(k) for n, k in
                zip(*np.unique(occupied_fill, return_counts=True))},
        },
    )
    for tag, mask in source.tag_masks.items():
        overlap = int((fine & mask).sum())
        result["ambiguous_tag_overlap"][tag] = {
            "fine_area_fraction": float(mask.mean()),
            "overlap_fine_pixels": overlap,
            "overlap_fraction_of_target": None if count == 0 else float(overlap / count),
        }
    return result


def texture_metrics(rgb640: np.ndarray, eligible: np.ndarray | None = None) -> dict[str, Any]:
    """Return descriptive image features on eligible source raster pixels.

    Rec.601 luminance is normalized to [0,1]. Finite-difference energy is the
    mean absolute difference pooled across horizontal and vertical neighbor
    pairs whose endpoints are eligible. The four-neighbor Laplacian variance
    uses interior pixels whose center and four neighbors are all eligible.
    These features do not diagnose defects or apply a sharpness threshold.
    """
    rgb = _rgb_array(rgb640, "Texture image", fine=True)
    if eligible is None:
        mask = np.ones((FINE_SIZE, FINE_SIZE), dtype=bool)
    else:
        mask = _bool_array(eligible, (FINE_SIZE, FINE_SIZE), "Eligible texture pixels")
    luminance = (rgb.astype(np.float64) @ np.array([0.299, 0.587, 0.114])) / 255.0
    values = luminance[mask]
    pixel_count = int(values.size)
    horizontal = mask[:, :-1] & mask[:, 1:]
    vertical = mask[:-1, :] & mask[1:, :]
    horizontal_values = np.abs(luminance[:, 1:] - luminance[:, :-1])[horizontal]
    vertical_values = np.abs(luminance[1:, :] - luminance[:-1, :])[vertical]
    pair_count = int(horizontal_values.size + vertical_values.size)
    centers = (mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1]
               & mask[1:-1, :-2] & mask[1:-1, 2:])
    laplacian = (luminance[:-2, 1:-1] + luminance[2:, 1:-1]
                 + luminance[1:-1, :-2] + luminance[1:-1, 2:]
                 - 4 * luminance[1:-1, 1:-1])[centers]
    return {
        "pixel_count": pixel_count,
        "luminance_mean": None if pixel_count == 0 else float(values.mean()),
        "luminance_std": None if pixel_count < 2 else float(values.std()),
        "finite_difference_pair_count": pair_count,
        "finite_difference_energy": None if pair_count == 0 else
            float((horizontal_values.sum() + vertical_values.sum()) / pair_count),
        "laplacian_pixel_count": int(laplacian.size),
        "laplacian_variance": None if laplacian.size < 2 else float(laplacian.var()),
    }
