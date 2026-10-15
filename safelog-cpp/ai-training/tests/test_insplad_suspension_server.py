import unittest
from pydantic import ValidationError
from safelog_ai.insplad_suspension_server import Context,CATEGORY,SuspensionEngine


class SuspensionServerTests(unittest.TestCase):
    def test_explicit_scope_and_previous_aliases(self):
        Context(inspectionTask=CATEGORY,promptVersion='test')
        Context(inspectionTask='equipment_surface',equipmentCategory=CATEGORY,promptVersion='test')
        for task in ('ksdd2_surface','insplad_upper_shackle_rust','insplad_glass_missing_cap'):
            Context(inspectionTask=task,promptVersion='test')
        with self.assertRaises(ValidationError):Context(inspectionTask=CATEGORY,equipmentCategory='insplad_upper_shackle_rust',promptVersion='test')

    def test_generic_health_and_mismatched_category_rejected(self):
        for task in ('whole_powerline_health','all_rust','facility_surface'):
            with self.assertRaises(ValidationError):Context(inspectionTask=task,equipmentCategory=CATEGORY,promptVersion='test')

    def test_unregistered_delta_rejected_before_model_load(self):
        with self.assertRaisesRegex(ValueError,'Registered suspension'):
            SuspensionEngine('unused','unused','unused','unused','unused','unused','unused','unused','not-opened.pt','bad','unused','unused')
