import base64
import io
import unittest
from PIL import Image
from fastapi.testclient import TestClient
from safelog_ai.pcb_reverse_server import create_app,photo_tensor
from safelog_ai.belt_inference import decode_photo


class FakeEngine:
    records={}
    def analyze(self,category,payload):
        image=decode_photo(payload)
        return {'equipmentCategory':category,'inputWidth':image.width,'inputHeight':image.height,
                'mechanicalFailureState':'not_assessed','humanConfirmationRequired':True,'fieldPerformanceValidated':False}


class ServerTests(unittest.TestCase):
    def body(self,image='eA=='):
        return {'equipmentCategory':'pcb3','sceneType':'registered_pcb_closeup','promptVersion':'pcb-v2','image':image}

    def test_unset_registry_refuses_analysis(self):
        client=TestClient(create_app());self.assertFalse(client.get('/health').json()['ready'])
        self.assertEqual(client.post('/v1/equipment/pcb/analyze',json=self.body()).status_code,503)

    def test_other_equipment_scene_and_bad_image_rejected(self):
        client=TestClient(create_app(FakeEngine()))
        for body in [dict(self.body(),equipmentCategory='conveyor_belt'),dict(self.body(),sceneType='motor'),self.body('%%%'),self.body('eA==')]:
            self.assertEqual(client.post('/v1/equipment/pcb/analyze',json=body).status_code,422)

    def test_native_geometry_and_unknown_health_preserved(self):
        buf=io.BytesIO();Image.new('RGB',(300,120)).save(buf,format='PNG')
        client=TestClient(create_app(FakeEngine()))
        value=client.post('/v1/equipment/pcb/analyze',json=self.body(base64.b64encode(buf.getvalue()).decode())).json()
        self.assertEqual((value['inputWidth'],value['inputHeight']),(300,120))
        self.assertEqual(value['mechanicalFailureState'],'not_assessed')
        self.assertTrue(value['humanConfirmationRequired']);self.assertFalse(value['fieldPerformanceValidated'])

    def test_model_input_is_fixed_256_square(self):
        self.assertEqual(tuple(photo_tensor(Image.new('RGB',(300,120))).shape),(1,3,256,256))


if __name__=='__main__':
    unittest.main()
