import unittest
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from scripts.prepare_s2ds import mask_targets
from scripts.audit_s2ds_crops import features,geometric_overlap


class S2DSTests(unittest.TestCase):
    def test_color_labels_and_unknown_categories_follow_publisher(self):
        classes=['concrete_crack','concrete_spalling','rust_stain','exposed_rebar','wet_surface','efflorescence','surface_cavity']
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'mask.png';image=np.zeros((64,64,3),dtype=np.uint8)
            image[:16,:16]=255;image[16:32,:16]=[255,0,0];image[32:48,:16]=[255,255,0];image[48:,:16]=[0,255,255]
            Image.fromarray(image).save(path);target,masks,known=mask_targets(path,classes)
            self.assertEqual(target,[1,1,1,-1,-1,1,-1]);self.assertEqual(known.tolist(),[1,1,1,0,0,1,0])
            self.assertFalse(masks[[3,4,6]].any())

    def test_transformed_crop_is_screened_as_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            random=np.random.default_rng(10)
            image=random.integers(0,255,(700,700),dtype=np.uint8)
            image=cv2.GaussianBlur(image,(5,5),0)
            left=Path(directory)/'whole.png';right=Path(directory)/'crop.png'
            cv2.imwrite(str(left),image);cv2.imwrite(str(right),image[100:600,150:650])
            self.assertGreaterEqual(geometric_overlap(features(right),features(left)),12)
