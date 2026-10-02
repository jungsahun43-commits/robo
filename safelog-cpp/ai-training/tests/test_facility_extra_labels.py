import xml.etree.ElementTree as ET
import numpy as np
import unittest
from scripts.prepare_codebrim import labels, parent_id
from scripts.prepare_facility_detail import tile_targets

CLASSES = ['concrete_crack', 'concrete_spalling', 'rust_stain', 'exposed_rebar',
           'wet_surface', 'efflorescence', 'surface_cavity']


def node(background=0, crack=1, spall=0):
    return ET.fromstring(f'<Defect><Background>{background}</Background><Crack>{crack}</Crack><Spallation>{spall}</Spallation><Efflorescence>0</Efflorescence><ExposedBars>0</ExposedBars><CorrosionStain>0</CorrosionStain></Defect>')


class ExtraLabelTests(unittest.TestCase):
    def test_background_cannot_claim_unknown_facility_labels_negative(self):
        self.assertEqual(labels(node(background=1, crack=0), CLASSES), [0, 0, 0, 0, -1, 0, -1])
        self.assertEqual(labels(node(crack=1, spall=1), CLASSES), [1, 1, 0, 0, -1, 0, -1])


    def test_bad_publisher_labels_fail_instead_of_silently_mapping(self):
        with self.assertRaises(ValueError): labels(node(background=1, crack=1), CLASSES)
        with self.assertRaises(ValueError): labels(node(background=0, crack=0), CLASSES)
        with self.assertRaises(ValueError): parent_id('arbitrary.png')
        self.assertEqual(parent_id('image_0000034_crop_0000001.png'), parent_id('image_0000034_crop_0000002.png'))


    def test_tiny_damage_crop_is_unknown_without_changing_zero_negative(self):
        a = np.zeros((5,5), dtype=bool); b = a.copy(); b.flat[:15] = True
        c = a.copy(); c.flat[:16] = True
        self.assertEqual(tile_targets([a,b,c,None], (0,0,5,5)), [0,-1,1,-1])
