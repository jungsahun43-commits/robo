"""Meaningful ontology, geometry, range safety and exact source exposure tests."""
import io
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch
from xml.etree import ElementTree as E
import numpy as np
from scripts.fetch_idea_research import member_name,RangeFile,decode_member
from scripts.prepare_idea_research import author_targets,validate_geometry,inspect
from scripts.fetch_rc2119 import sha
from PIL import Image
from scripts.facility_idea_research import fixed_draws,valid_targets,EPOCHS,DRAW_COUNT,REPLACEMENTS

def document(presence='damage',structure='AE_BUILDING',objects=''):
    return E.fromstring(f'<annotation><structureType>{structure}</structureType><damagePresence>{presence}</damagePresence><size><width>100</width><height>100</height></size>{objects}</annotation>')
def obj(element='RC column',name='Spalling',detail='Concrete spalling',box='10,20,30,40'):
    values=box.split(',');geometry=''.join(f'<{k}>{v}</{k}>'for k,v in zip(('xmin','ymin','xmax','ymax'),values))
    return f'<object element="{element}" type="rect"><name damage="{detail}">{name}</name><bndbox>{geometry}</bndbox></object>'

class IdeaTests(unittest.TestCase):
    def test_explicit_normal_primary_only(self):
        self.assertEqual(author_targets(document('no damage'))[0],[0,0,-1,-1,-1,-1,-1])
    def test_contradictory_normal_rejected(self):
        with self.assertRaises(ValueError):author_targets(document('no damage',objects=obj()))
    def test_explicit_rc_spall_missing_crack_unknown(self):
        self.assertEqual(author_targets(document(objects=obj()))[0],[-1,1,-1,-1,-1,-1,-1])
    def test_crack_and_spall_cooccurrence(self):
        self.assertEqual(author_targets(document(objects=obj()+obj(name='Crack',detail='Shear crack')))[0],[1,1,-1,-1,-1,-1,-1])
    def test_nonconcrete_spall_rejected(self):
        with self.assertRaises(ValueError):author_targets(document(objects=obj('Plaster')))
    def test_crushing_not_spalling(self):
        with self.assertRaises(ValueError):author_targets(document(objects=obj(detail='Concrete crushing')))
    def test_bridge_excluded(self):
        with self.assertRaises(ValueError):author_targets(document(structure='BRIDGE',objects=obj()))
    def test_damage_absence_not_known_negative(self):
        with self.assertRaises(ValueError):valid_targets({'stratum':'rc_spalling','targets':[0,1,-1,-1,-1,-1,-1]})
    def test_out_of_frame_geometry_rejected(self):
        with self.assertRaises(ValueError):validate_geometry(document(objects=obj(box='10,20,101,40')),(100,100))
    def test_mismatched_geometry_size_rejected(self):
        with self.assertRaises(ValueError):validate_geometry(document(objects=obj()),(90,100))
    def test_valid_bbox_is_not_pixel_label(self):
        self.assertIsNone(validate_geometry(document(objects=obj()),(100,100)))
    def test_member_paths_cannot_escape(self):
        for path in('../x.jpg','/x.jpg','C:/x.jpg','a\\x.jpg'):
            with self.assertRaises(ValueError):member_name(path)
    def test_fixed_budget_exact_balance_untouched_core(self):
        original=np.arange(EPOCHS*DRAW_COUNT,dtype=np.int64).reshape(EPOCHS,DRAW_COUNT)%200
        items=[{'stratum':'rc_spalling'}for _ in range(3)]+[{'stratum':'author_no_damage'}for _ in range(4)]
        candidate=fixed_draws(original,200,items)
        self.assertTrue(np.array_equal(candidate,fixed_draws(original,200,items)))
        for old,new in zip(original,candidate):
            mask=new>=200;self.assertEqual(mask.sum(),REPLACEMENTS)
            self.assertEqual(((new>=200)&(new<203)).sum(),REPLACEMENTS//2)
            self.assertEqual((new>=203).sum(),REPLACEMENTS//2)
            self.assertTrue(np.array_equal(old[~mask],new[~mask]))
    def test_range_server_cannot_substitute_full_file(self):
        with tempfile.TemporaryDirectory(dir='runs')as folder:
            view=RangeFile('damage.zip');view.cache=Path(folder)
            response=type('Response',(),{'status_code':200,'headers':{},'content':b'invalid'})()
            with patch('scripts.fetch_idea_research.request',return_value=response):
                with self.assertRaises(ValueError):view.read(4)
    def test_member_crc_and_size_verified(self):
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED)as archive:archive.writestr('a.jpg',b'author-original-bytes'*30)
        with zipfile.ZipFile(io.BytesIO(stream.getvalue()))as archive:entry=archive.getinfo('a.jpg')
        self.assertEqual(decode_member(stream.getvalue(),entry),b'author-original-bytes'*30)
        entry.CRC^=1
        with self.assertRaises(ValueError):decode_member(stream.getvalue(),entry)
    def test_truncated_member_rejected(self):
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w')as archive:archive.writestr('a.jpg',b'1234567890')
        with zipfile.ZipFile(io.BytesIO(stream.getvalue()))as archive:entry=archive.getinfo('a.jpg')
        with self.assertRaises(ValueError):decode_member(stream.getvalue()[:36],entry)
    def test_same_size_wrong_photo_annotation_quarantined(self):
        with tempfile.TemporaryDirectory(dir='runs')as folder:
            base=Path(folder);image=base/'a.jpg';annotation=base/'a.xml';Image.new('RGB',(100,100)).save(image)
            root=document('no damage');E.SubElement(root,'filename').text='different.jpg';E.ElementTree(root).write(annotation)
            row={'image':str(image),'annotation':str(annotation),'source_image_sha256':sha(image),'annotation_sha256':sha(annotation),'image_member':'a.jpg'}
            self.assertIn('filename',inspect(row)['geometry_issue'])

if __name__=='__main__':unittest.main()
