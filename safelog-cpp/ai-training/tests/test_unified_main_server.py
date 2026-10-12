import unittest
from fastapi.testclient import TestClient
from safelog_ai.unified_main_server import create_app


class FakeEngine:
    digest='x'*64
    def analyze(self,task,image):
        return {'classResults':[{'className':'belt_crack','score':.8,'candidateDetected':image=='before'}],
            'candidateDetected':image=='before','inspectionTask':task,'humanConfirmationRequired':True}


class MainServerTests(unittest.TestCase):
    def test_missing_task_or_model_cannot_return_normal(self):
        with TestClient(create_app()) as client:
            body={'image':'x','promptVersion':'v1'}
            self.assertEqual(client.post('/v1/analyze-hazard',json=body).status_code,422)
            body['inspectionTask']='belt_damage';self.assertEqual(client.post('/v1/analyze-hazard',json=body).status_code,503)
            body['inspectionTask']='internal_motor_fault';self.assertEqual(client.post('/v1/analyze-hazard',json=body).status_code,422)
    def test_comparison_only_suggests_resolution_when_before_candidate_disappears(self):
        with TestClient(create_app(FakeEngine())) as client:
            body={'inspectionTask':'belt_damage','beforeImage':'before','afterImage':'after','promptVersion':'v1'}
            result=client.post('/v1/compare-action',json=body).json();self.assertTrue(result['likelyResolved']);self.assertTrue(result['humanConfirmationRequired'])
            body['beforeImage']='after';self.assertFalse(client.post('/v1/compare-action',json=body).json()['likelyResolved'])
            self.assertEqual(client.post('/v1/summarize',json={'inspectionContext':'점검 기록','promptVersion':'v1'}).json()['generationMethod'],'rule_template_not_trained_LLM')


if __name__=='__main__':unittest.main()
