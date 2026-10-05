import copy
import unittest

import numpy as np
from PIL import Image, ImageDraw

from scripts.prepare_facility_data import DACL_MAPPING
from safelog_ai.spalling_geometry import (AMBIGUOUS_TAGS, CLASSES, SourceRaster,
    any80_mask, dacl_source_raster, dam_source_raster, geometry_metrics,
    texture_metrics, unavailable_source_raster)


class SpallingGeometryTests(unittest.TestCase):
    def test_dacl_polygon_replay_matches_existing_box_and_any_recipe(self):
        # Non-square processed dimensions, fractional source coordinates and
        # out-of-frame coordinates exercise scaling and PIL's own clipping.
        doc = {"imageWidth": 320, "imageHeight": 200, "shapes": [
            {"label": "Spalling", "points": [[-2.5, 3.25], [70.5, 5.75], [65.0, 44.5]]},
            {"label": "Crack", "points": [[310, 170], [345, 170], [345, 220]]},
            {"label": "Cavity", "points": [[20, 15], [28, 15], [25, 22]]},
            {"label": "Spalling", "points": [[100, 100], [101, 101]]},
        ]}
        original = copy.deepcopy(doc)
        width, height = 1153, 777
        raster = dacl_source_raster(doc, (width, height))
        expected = [Image.new("L", (width, height)) for _ in CLASSES]
        # Independent replay of the unchanged producer's annotation recipe.
        for shape in doc["shapes"]:
            mapped = DACL_MAPPING.get(shape["label"])
            if mapped is None or len(shape["points"]) < 3:
                continue
            points = [(x * width / doc["imageWidth"], y * height / doc["imageHeight"])
                      for x, y in shape["points"]]
            ImageDraw.Draw(expected[CLASSES.index(mapped)]).polygon(points, fill=255)
        for index, mask in enumerate(expected):
            fine = np.asarray(mask.resize((640, 640), Image.Resampling.BOX)) > 0
            coarse = fine.reshape(80, 8, 80, 8).max((1, 3))
            np.testing.assert_array_equal(raster.fine_masks[index], fine)
            np.testing.assert_array_equal(raster.any_masks[index], coarse)
        self.assertEqual(doc, original)
        self.assertEqual(raster.fine_masks.dtype, np.bool_)
        self.assertTrue(raster.class_available.all())

    def test_dam_exact_colors_box_retention_and_unmapped_publisher_marks(self):
        rgb = np.zeros((1280, 1280, 3), dtype=np.uint8)
        rgb[0, 0] = (255, 0, 0)
        rgb[8, 8] = (0, 0, 255)
        rgb[24, 24] = (254, 0, 0)
        rgb[32, 32] = (0, 255, 0)
        before = rgb.copy()
        raster = dam_source_raster(rgb)
        for index, color in ((0, (255, 0, 0)), (1, (0, 0, 255))):
            native = Image.fromarray(np.uint8(np.all(rgb == color, axis=-1)) * 255)
            fine = np.asarray(native.resize((640, 640), Image.Resampling.BOX)) > 0
            np.testing.assert_array_equal(raster.fine_masks[index], fine)
            np.testing.assert_array_equal(raster.any_masks[index], any80_mask(fine))
            self.assertEqual(int(fine.sum()), 1)
        self.assertFalse(raster.fine_masks[:, 12, 12].any())
        self.assertFalse(raster.source_unmarked[12, 12])
        self.assertFalse(raster.source_unmarked[16, 16])
        np.testing.assert_array_equal(raster.class_available, [True, True, False, False, False, False, False])
        np.testing.assert_array_equal(rgb, before)

    def test_geometry_counts_exact_fill_and_any_cell_expansion(self):
        rgb = np.zeros((640, 640, 3), dtype=np.uint8)
        rgb[0, 0] = (0, 0, 255)              # One pixel in one cell.
        rgb[8, 8:16] = (0, 0, 255)           # Eight pixels: exactly 1/8.
        rgb[16:24, 16:24] = (0, 0, 255)      # One full cell.
        metrics = geometry_metrics(dam_source_raster(rgb))
        self.assertEqual(metrics["occupied80cells"], 3)
        self.assertEqual(metrics["thin_cells_le_eighth"], 2)
        self.assertAlmostEqual(metrics["fine_area_fraction"], 73 / (640 * 640))
        self.assertAlmostEqual(metrics["coarse_any_area_fraction"], 3 / (80 * 80))
        self.assertAlmostEqual(metrics["expansion_ratio"], 192 / 73)
        self.assertAlmostEqual(metrics["thin_cells_le_eighth_fraction"], 2 / 3)
        self.assertEqual(metrics["occupied80_cell_fill"]["histogram_fine_pixels"],
                         {"1": 1, "8": 1, "64": 1})

    def test_ambiguous_tag_overlap_uses_actual_pixels_and_union_includes_tags(self):
        def polygon(label, x1, y1, x2, y2):
            return {"label": label, "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]}
        doc = {"imageWidth": 640, "imageHeight": 640, "shapes": [
            polygon("Spalling", 0, 0, 10, 10),
            polygon("Rockpocket", 200, 200, 210, 210),
            polygon("WConccor", 0, 0, 4, 4),
            polygon("Hollowareas", 300, 300, 305, 305),
            polygon("Cavity", 3, 3, 8, 8),
            polygon("OtherPublisherTag", 400, 400, 401, 401),
        ]}
        source = dacl_source_raster(doc, (640, 640))
        metrics = geometry_metrics(source)
        overlap = metrics["ambiguous_tag_overlap"]
        self.assertEqual(set(overlap), set(AMBIGUOUS_TAGS))
        self.assertEqual(overlap["Rockpocket"]["overlap_fine_pixels"], 0)
        self.assertGreater(overlap["Rockpocket"]["fine_area_fraction"], 0)
        self.assertEqual(overlap["WConccor"]["overlap_fine_pixels"], 25)
        self.assertEqual(overlap["Cavity"]["overlap_fine_pixels"], 36)
        self.assertAlmostEqual(overlap["WConccor"]["overlap_fraction_of_target"], 25 / 121)
        self.assertFalse(source.fine_masks[:, 200, 200].any())
        self.assertFalse(source.source_unmarked[200, 200])
        self.assertFalse(source.source_unmarked[400, 400])
        self.assertTrue(source.source_unmarked[500, 500])

    def test_empty_available_raster_and_unavailable_source_remain_distinct(self):
        dam = dam_source_raster(np.zeros((640, 640, 3), dtype=np.uint8))
        empty = geometry_metrics(dam)
        unknown = geometry_metrics(unavailable_source_raster())
        self.assertTrue(empty["raster_available"])
        self.assertEqual(empty["fine_area_fraction"], 0.0)
        self.assertEqual(empty["occupied80cells"], 0)
        self.assertIsNone(empty["expansion_ratio"])
        self.assertEqual(empty["source_unmarked_area_fraction"], 1.0)
        self.assertFalse(unknown["raster_available"])
        self.assertIsNone(unknown["fine_area_fraction"])
        self.assertIsNone(unknown["occupied80cells"])
        self.assertIsNone(unknown["source_unmarked_area_fraction"])
        self.assertFalse(geometry_metrics(dam, target_index=2)["raster_available"])
        self.assertFalse(unavailable_source_raster().fine_masks.any())

    def test_texture_checkerboard_features_and_normalized_luminance(self):
        checker = ((np.indices((640, 640)).sum(axis=0) % 2) * 255).astype(np.uint8)
        rgb = np.repeat(checker[:, :, None], 3, axis=2)
        metrics = texture_metrics(rgb)
        self.assertEqual(metrics["pixel_count"], 640 * 640)
        self.assertAlmostEqual(metrics["luminance_mean"], 0.5)
        self.assertAlmostEqual(metrics["luminance_std"], 0.5)
        self.assertAlmostEqual(metrics["finite_difference_energy"], 1.0)
        self.assertEqual(metrics["finite_difference_pair_count"], 2 * 640 * 639)
        self.assertEqual(metrics["laplacian_pixel_count"], 638 * 638)
        self.assertAlmostEqual(metrics["laplacian_variance"], 16.0)
        red = np.zeros_like(rgb)
        red[:, :, 0] = 255
        self.assertAlmostEqual(texture_metrics(red)["luminance_mean"], 0.299)

    def test_texture_eligible_mask_excludes_neighbor_boundaries_and_handles_insufficient(self):
        rgb = np.full((640, 640, 3), 255, dtype=np.uint8)
        mask = np.zeros((640, 640), dtype=bool)
        empty = texture_metrics(rgb, mask)
        for key in ("luminance_mean", "luminance_std", "finite_difference_energy", "laplacian_variance"):
            self.assertIsNone(empty[key])
        mask[20, 20] = True
        rgb[20, 20] = 0
        singleton = texture_metrics(rgb, mask)
        self.assertEqual(singleton["luminance_mean"], 0.0)
        self.assertIsNone(singleton["luminance_std"])
        self.assertEqual(singleton["finite_difference_pair_count"], 0)
        self.assertIsNone(singleton["finite_difference_energy"])
        mask[40:43, 40:44] = True
        rgb[40:43, 40:44] = 0
        patch = texture_metrics(rgb, mask)
        self.assertEqual(patch["finite_difference_pair_count"], 17)
        self.assertEqual(patch["finite_difference_energy"], 0.0)
        self.assertEqual(patch["laplacian_pixel_count"], 2)
        self.assertEqual(patch["laplacian_variance"], 0.0)

    def test_invalid_dimensions_coordinates_arrays_and_unknown_positive_storage_are_rejected(self):
        doc = {"imageWidth": 640, "imageHeight": 640, "shapes": []}
        for size in ((True, 640), (640.5, 640), (0, 640), (640, float("nan"))):
            with self.subTest(size=size), self.assertRaises(ValueError):
                dacl_source_raster(doc, size)
        for value in (True, float("nan"), float("inf"), "3"):
            malformed = {**doc, "shapes": [{"label": "Spalling",
                "points": [[value, 1], [10, 1], [10, 10]]}]}
            with self.subTest(coordinate=value), self.assertRaises(ValueError):
                dacl_source_raster(malformed, (640, 640))
        for rgb in (np.zeros((640, 640, 3), dtype=bool),
                    np.full((640, 640, 3), 0.5),
                    np.full((640, 640, 3), np.nan),
                    np.full((640, 640, 3), 256, dtype=np.uint16),
                    np.zeros((640, 640), dtype=np.uint8)):
            with self.subTest(rgb_dtype=rgb.dtype, rgb_shape=rgb.shape), self.assertRaises(ValueError):
                dam_source_raster(rgb)
        for mask in (np.zeros((640, 640), dtype=np.uint8),
                     np.zeros((80, 80), dtype=bool),
                     np.full((640, 640), np.nan)):
            with self.subTest(mask_dtype=mask.dtype, mask_shape=mask.shape), self.assertRaises(ValueError):
                any80_mask(mask)
        with self.assertRaises(ValueError):
            texture_metrics(np.zeros((32, 32, 3), dtype=np.uint8))
        with self.assertRaises(ValueError):
            texture_metrics(np.zeros((640, 640, 3), dtype=np.uint8),
                            np.zeros((640, 640), dtype=np.uint8))
        with self.assertRaises(ValueError):
            geometry_metrics(unavailable_source_raster(), True)
        unavailable = unavailable_source_raster()
        fine = unavailable.fine_masks.copy()
        fine[1, 0, 0] = True
        coarse = unavailable.any_masks.copy()
        coarse[1, 0, 0] = True
        with self.assertRaises(ValueError):
            SourceRaster("unavailable", fine, coarse, unavailable.class_available,
                         {}, None, None)


if __name__ == "__main__":
    unittest.main()
