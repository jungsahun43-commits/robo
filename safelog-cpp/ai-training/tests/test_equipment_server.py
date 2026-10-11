import base64
import io
import unittest

from fastapi.testclient import TestClient
from PIL import Image

from safelog_ai.belt_inference import decode_photo, letterbox
from safelog_ai.equipment_server import create_app


class FakeEngine:
    model_sha256 = 'test-only'
    def analyze(self,payload):
        image = decode_photo(payload)
        return {'inputWidth':image.width,'inputHeight':image.height,
                'mechanicalFailureState':'not_assessed','humanConfirmationRequired':True,
                'normalPhotoFprValidated':False}


class EquipmentServerTests(unittest.TestCase):
    def test_no_checkpoint_refuses_analysis(self):
        client = TestClient(create_app())
        self.assertFalse(client.get('/health').json()['ready'])
        self.assertEqual(client.post('/v1/equipment/belt/analyze',json={'image':'eA==','promptVersion':'test'}).status_code,503)

    def test_wrong_equipment_and_bad_images_rejected(self):
        client = TestClient(create_app(FakeEngine()))
        for body in [{'image':'eA==','promptVersion':'test','equipmentCategory':'motor'},
                     {'image':'%%%','promptVersion':'test'},{'image':'eA==','promptVersion':'test'}]:
            self.assertEqual(client.post('/v1/equipment/belt/analyze',json=body).status_code,422)

    def test_valid_request_keeps_unknown_health_and_native_geometry(self):
        buffer = io.BytesIO()
        Image.new('RGB',(120,60),(100,100,100)).save(buffer,format='PNG')
        client = TestClient(create_app(FakeEngine()))
        body = {'image':'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode(),
                'promptVersion':'belt-appearance-v1'}
        response = client.post('/v1/equipment/belt/analyze',json=body)
        self.assertEqual(response.status_code,200)
        value = response.json()
        self.assertEqual((value['inputWidth'],value['inputHeight']),(120,60))
        self.assertEqual(value['mechanicalFailureState'],'not_assessed')
        self.assertTrue(value['humanConfirmationRequired'])
        self.assertFalse(value['normalPhotoFprValidated'])
        self.assertEqual(value['promptVersion'],'belt-appearance-v1')

    def test_letterbox_keeps_aspect_ratio_and_extreme_geometry_nonzero(self):
        for size,expected in [((120,60),(0,160,640,320)),((1000,1),(0,319,640,1))]:
            tensor,geometry = letterbox(Image.new('RGB',size))
            self.assertEqual(tuple(tensor.shape),(1,3,640,640))
            self.assertEqual(geometry,expected)


if __name__=='__main__':
    unittest.main()
