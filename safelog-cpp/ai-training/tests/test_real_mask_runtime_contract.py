import inspect
import unittest
from safelog_ai.real_mask_server import RuntimeRealMaskMain
from safelog_ai.magnetic_surface import RealMaskMain
from safelog_ai.industrial_surface_reference import IndustrialSurfaceMain


class RuntimeContractTests(unittest.TestCase):
    def test_old_surface_method_is_preserved_with_original_signature(self):
        self.assertIs(RuntimeRealMaskMain.forward_surface,IndustrialSurfaceMain.forward_surface)
        inspect.signature(RuntimeRealMaskMain.forward_surface).bind(object(),object(),object(),'mpdd_bracket_brown')

    def test_new_surface_head_has_distinct_method_name(self):
        self.assertIs(RuntimeRealMaskMain.forward_real_mask,RealMaskMain.forward_surface)
        inspect.signature(RuntimeRealMaskMain.forward_real_mask).bind(object(),object())


if __name__=='__main__':unittest.main()
