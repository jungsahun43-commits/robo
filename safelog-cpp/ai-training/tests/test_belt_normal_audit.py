import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_belt_normal_coco import audit, normal_status


class NormalAuditTests(unittest.TestCase):
    def test_absent_or_mixed_labels_never_invent_normal_truth(self):
        self.assertEqual(normal_status([]),'unannotated_unknown')
        self.assertEqual(normal_status(['normal belt','longitudinal tear']),'mixed_labels_requires_review')
        self.assertEqual(normal_status(['belt splice']),'no_explicit_normal_roi')
        with self.assertRaises(ValueError):
            normal_status(['healthy machine'])

    def test_augmented_parent_leak_and_whole_photo_unknown(self):
        image = io.BytesIO()
        Image.new('RGB',(12,8),(80,80,80)).save(image,format='PNG')
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder)/'example.zip'
            with zipfile.ZipFile(archive,'w') as packed:
                for split,uuid in [('train','aaaa'),('valid','bbbb')]:
                    name = f'belt_jpg.rf.{uuid}.png'
                    coco = {'categories':[{'id':1,'name':'normal belt'}],
                        'images':[{'id':1,'file_name':name,'width':12,'height':8}],
                        'annotations':[{'id':1,'image_id':1,'category_id':1,'bbox':[1,1,5,4]}]}
                    packed.writestr(split+'/_annotations.coco.json',json.dumps(coco))
                    packed.writestr(split+'/'+name,image.getvalue())
            result = audit(archive)
        self.assertEqual(result['normal_roi_status_counts'],{'explicit_normal_roi_candidate':2})
        self.assertEqual(result['cross_split_parent_hint_groups'],['belt_jpg'])
        self.assertEqual(len(result['cross_split_exact_rgb_groups']),1)
        self.assertEqual(result['whole_photo_normal_labels_created'],0)
        self.assertFalse(result['normal_roi_is_whole_photo_normal'])

    def test_zip_path_traversal_is_rejected_without_extraction(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder)/'bad.zip'
            with zipfile.ZipFile(archive,'w') as packed:
                packed.writestr('../unexpected.txt','bad')
            with self.assertRaises(ValueError):
                audit(archive)


if __name__=='__main__':
    unittest.main()
