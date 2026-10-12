import base64
import io
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from fastapi.testclient import TestClient
from safelog_ai.pcb_multisource_server import PooledEngine, create_app, decode


class SingleModelApiTests(unittest.TestCase):
    def test_unpinned_or_wrong_checkpoint_cannot_be_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'fake.pt'; path.write_bytes(b'not a trusted checkpoint')
            with self.assertRaises(ValueError):
                PooledEngine(path, '0'*64)
        request = {'image': 'eA==', 'equipmentCategory': 'robustad_pcb',
                   'sceneType': 'registered_pcb_closeup', 'promptVersion': 'test'}
        with TestClient(create_app()) as client:
            self.assertEqual(client.post('/v1/equipment/pcb/analyze', json=request).status_code, 503)
            self.assertEqual(client.post('/v1/equipment/pcb/analyze', json=dict(request, equipmentCategory='motor')).status_code, 422)

    def test_image_decoder_accepts_only_stated_formats_and_applies_native_geometry(self):
        for format in ['JPEG', 'PNG', 'GIF']:
            output = io.BytesIO(); Image.new('RGB', (300, 150)).save(output, format=format)
            if format == 'GIF':
                with self.assertRaises(ValueError): decode(output.getvalue())
            else:
                self.assertEqual(decode(output.getvalue()).size, (300, 150))


if __name__ == '__main__':
    unittest.main()
