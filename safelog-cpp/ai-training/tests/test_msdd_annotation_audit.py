import unittest
from scripts.audit_msdd_annotations import annotation_metadata


class AnnotationValidityTests(unittest.TestCase):
    def test_source_bad_coordinate_preserved(self):
        raw = b'<annotation><filename>a.jpg</filename><size><width>600</width><height>600</height></size><object><name>dent</name><bndbox><xmin>167</xmin><ymin>516</ymin><xmax>222</xmax><ymax>222</ymax></bndbox></object></annotation>'
        result = annotation_metadata(raw)
        self.assertEqual(result['boxes'][0]['xyxy'], [167, 516, 222, 222])
        self.assertFalse(result['all_boxes_coordinate_valid'])
        self.assertEqual(result['author_photo_label'], 1)

    def test_unannotated_surface_and_external_entities(self):
        result = annotation_metadata(b'<annotation><size><width>600</width><height>600</height></size></annotation>')
        self.assertEqual(result['author_photo_label'], 0)
        with self.assertRaises(ValueError):
            annotation_metadata(b'<!DOCTYPE a><annotation/>')


if __name__ == '__main__':
    unittest.main()
