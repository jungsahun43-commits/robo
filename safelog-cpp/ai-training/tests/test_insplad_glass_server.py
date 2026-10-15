import unittest
from pydantic import ValidationError
from safelog_ai.insplad_glass_server import Context,CATEGORY


class GlassServerTests(unittest.TestCase):
    def test_explicit_glass_domain(self):
        Context(inspectionTask=CATEGORY,promptVersion='test')
        Context(inspectionTask='equipment_surface',equipmentCategory=CATEGORY,promptVersion='test')
        with self.assertRaises(ValidationError):Context(inspectionTask=CATEGORY,equipmentCategory='insplad_upper_shackle_rust',promptVersion='test')

    def test_previous_aliases_retained(self):
        Context(inspectionTask='ksdd2_surface',promptVersion='test')
        Context(inspectionTask='insplad_upper_shackle_rust',promptVersion='test')
        with self.assertRaises(ValidationError):Context(inspectionTask='facility_surface',equipmentCategory=CATEGORY,promptVersion='test')

    def test_unsupported_whole_health_rejected(self):
        for task in ('powerline_electrical_health','all_missing_parts','all_rust'):
            with self.assertRaises(ValidationError):Context(inspectionTask=task,promptVersion='test')
