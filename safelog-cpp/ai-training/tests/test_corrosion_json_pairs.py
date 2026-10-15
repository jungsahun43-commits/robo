import base64
import io
import unittest
import numpy as np
from PIL import Image
from safelog_ai.corrosion_json_pairs import embedded_image,polygon_labels,locked_phase


class CorrosionJsonPairTests(unittest.TestCase):
    def test_embedded_trust_boundary_and_no_path_fallback(self):
        f=io.BytesIO();Image.new('RGB',(8,6),(10,20,30)).save(f,format='PNG')
        a={'imageData':base64.b64encode(f.getvalue()).decode()+'\n','imageWidth':8,'imageHeight':6,'imagePath':'../../outside.png'}
        rgb,raw,meta=embedded_image(a);self.assertEqual(rgb.size,(8,6));self.assertEqual(raw,f.getvalue())
        for bad in (None,'%%%'):
            with self.assertRaises(ValueError):embedded_image({**a,'imageData':bad})
        with self.assertRaises(ValueError):embedded_image({**a,'imageWidth':9})

    def test_official_canvas_clipping_shape_order_and_unknown_label(self):
        a={'imageWidth':8,'imageHeight':6,'shapes':[
            {'label':'2_Fair_Steel_Corrosion','points':[[-5,-5],[5,-5],[5,4],[-5,4]]},
            {'label':'3_Poor_Steel_Corrosion','shape_type':'polygon','points':[[1,1],[3,1],[3,3],[1,3]]}]}
        m,report=polygon_labels(a,(8,6));self.assertEqual(int(m[0,0]),1);self.assertEqual(int(m[2,2]),2)
        self.assertEqual(int(m[5,7]),0);self.assertGreater(report['outside_canvas_vertices_clipped_by_rasterizer_not_moved'],0)
        a['shapes'][0]['label']='invented rust'
        with self.assertRaises(ValueError):polygon_labels(a,(8,6))

    def test_locked_roles_and_held_priority(self):
        self.assertEqual(locked_phase('Test',True,{'fit'}),'test')
        self.assertEqual(locked_phase('Train',True,{'fit','cal'}),'excluded_TEST_connected')
        self.assertEqual(locked_phase('Train',False,{'fit','cal'}),'excluded_prior_role_conflict')
        self.assertEqual(locked_phase('Train',False,{'validation'}),'validation')
        with self.assertRaises(ValueError):locked_phase('Train',False,set())


if __name__=='__main__':unittest.main()
