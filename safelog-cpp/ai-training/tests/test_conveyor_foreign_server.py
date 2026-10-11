import base64
import io
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from safelog_ai.belt_inference import decode_photo
from safelog_ai.conveyor_foreign_inference import ForeignEngine
from safelog_ai.conveyor_foreign_server import create_app


class FakeEngine:
    model_sha256 = 'test-only'

    def analyze(self, payload):
        image = decode_photo(payload)
        return {'inputWidth':image.width, 'inputHeight':image.height, 'mechanicalFailureState':'not_assessed',
                'humanConfirmationRequired':True, 'localizationAvailable':False, 'fieldPerformanceValidated':False}


class ForeignServerTests(unittest.TestCase):
    def body(self, image='eA=='):
        return {'image':image, 'equipmentCategory':'conveyor_belt',
                'sceneType':'iron_ore_closeup', 'promptVersion':'conveyor-foreign-v1'}

    def test_no_checkpoint_refuses_analysis(self):
        client = TestClient(create_app())
        self.assertFalse(client.get('/health').json()['ready'])
        self.assertEqual(client.post('/v1/equipment/conveyor/foreign/analyze', json=self.body()).status_code, 503)

    def test_wrong_or_missing_scene_rejected(self):
        client = TestClient(create_app(FakeEngine()))
        bodies = [dict(self.body(), sceneType='parcel_conveyor'), dict(self.body(), equipmentCategory='motor')]
        missing = self.body(); missing.pop('sceneType'); bodies.append(missing)
        for body in bodies:
            self.assertEqual(client.post('/v1/equipment/conveyor/foreign/analyze', json=body).status_code, 422)

    def test_corrupt_image_and_wrong_data_url_rejected(self):
        client = TestClient(create_app(FakeEngine()))
        for value in ('%%%', 'eA==', 'data:text/plain;base64,eA=='):
            self.assertEqual(client.post('/v1/equipment/conveyor/foreign/analyze', json=self.body(value)).status_code, 422)

    def test_native_geometry_and_scope_preserved(self):
        buf = io.BytesIO(); Image.new('RGB', (180, 45)).save(buf, format='PNG')
        client = TestClient(create_app(FakeEngine()))
        response = client.post('/v1/equipment/conveyor/foreign/analyze',
            json=self.body('data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()))
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual((result['inputWidth'], result['inputHeight']), (180, 45))
        self.assertEqual(result['mechanicalFailureState'], 'not_assessed')
        self.assertTrue(result['humanConfirmationRequired'])
        self.assertFalse(result['localizationAvailable'])
        self.assertFalse(result['fieldPerformanceValidated'])

    def test_checkpoint_hash_mismatch_fails_before_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'bad.pt'; path.write_bytes(b'not a checkpoint')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                ForeignEngine(path, '0'*64)


if __name__=='__main__':
    unittest.main()
