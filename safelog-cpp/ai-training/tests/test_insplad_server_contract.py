import unittest
from pydantic import ValidationError
from safelog_ai.insplad_server import Context,CATEGORY


class ContractTests(unittest.TestCase):
    def test_explicit_alias_and_category(self):
        Context(inspectionTask=CATEGORY,promptVersion='test')
        Context(inspectionTask='equipment_surface',equipmentCategory=CATEGORY,promptVersion='test')
        with self.assertRaises(ValidationError):Context(inspectionTask=CATEGORY,equipmentCategory='ad_screw',promptVersion='test')

    def test_unknown_or_generic_scope_rejected(self):
        for task in ('all_rust','all_equipment','whole_powerline'):
            with self.assertRaises(ValidationError):Context(inspectionTask=task,promptVersion='test')
        with self.assertRaises(ValidationError):Context(inspectionTask='equipment_surface',promptVersion='test')

    def test_old_task_constraints_retained(self):
        Context(inspectionTask='ksdd2_surface',promptVersion='test')
        with self.assertRaises(ValidationError):Context(inspectionTask='facility_surface',equipmentCategory=CATEGORY,promptVersion='test')
