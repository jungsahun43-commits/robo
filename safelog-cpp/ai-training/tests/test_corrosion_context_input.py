import unittest
import numpy as np
from safelog_ai.corrosion_context_input import whole_geometry,native_context,context_tile512,context_tile128,fixed_probe_tiles


class ContextGeometryTest(unittest.TestCase):
    def test_letterbox_padding_removed_before_native_probability_restore(self):
        size=(1348,132);g=whole_geometry(size);p=np.full((512,512),.99,np.float32)
        x,y=g['left'],g['top'];w,h=g['resized_size'];p[y:y+h,x:x+w]=.2
        field=native_context(p,size)
        np.testing.assert_allclose(field,np.log(.2/.8)/8,atol=1e-6)
        self.assertEqual(field.shape,(132,1348))

    def test_tile_feature_centers_follow_native_coordinates(self):
        yy,xx=np.mgrid[:800,:1200];field=((xx+yy)/4000).astype(np.float32)
        b={'left':384,'top':128,'width':512,'height':512};actual=context_tile128(field,b)[0]
        y,x=np.mgrid[:128,:128];expected=(384+4*x+1.5+128+4*y+1.5)/4000
        np.testing.assert_allclose(actual,expected,atol=1e-7)

    def test_thin_axis_padding_is_neutral_and_grid_uses_only_size(self):
        field=np.full((100,200),.3,np.float32);box=fixed_probe_tiles((200,100))[0]
        tile=context_tile512(field,box)
        self.assertTrue(np.all(tile[100:]==0));self.assertTrue(np.all(tile[:,200:]==0))
        np.testing.assert_array_equal(tile[:100,:200],field)
        self.assertEqual(len(fixed_probe_tiles((2048,2048))),3)
