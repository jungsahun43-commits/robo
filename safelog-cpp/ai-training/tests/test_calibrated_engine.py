import unittest
from unittest.mock import patch
from safelog_ai.foreign_calibrated_server import CalibratedEngine


class CalibratedEngineTests(unittest.TestCase):
    def test_fixed_checkpoint_threshold_replaces_v1_decision(self):
        engine = object.__new__(CalibratedEngine)
        engine.threshold = .02;engine.calibration_protocol_sha256 = 'test-only'
        with patch('safelog_ai.conveyor_foreign_inference.ForeignEngine.analyze',return_value={
            'presenceScore':.03,'threshold':.5,'candidateDetected':False,
            'mechanicalFailureState':'not_assessed','humanConfirmationRequired':True,'fieldPerformanceValidated':False}):
            value = engine.analyze(b'test-only')
        self.assertTrue(value['candidateDetected'])
        self.assertEqual(value['threshold'],.02)
        self.assertEqual(value['schema'],'safelog.equipment.conveyor-foreign.v2')
        self.assertEqual(value['mechanicalFailureState'],'not_assessed')
        self.assertFalse(value['fieldPerformanceValidated'])

    def test_threshold_boundary_remains_strict(self):
        engine = object.__new__(CalibratedEngine)
        engine.threshold = .02;engine.calibration_protocol_sha256 = 'test-only'
        with patch('safelog_ai.conveyor_foreign_inference.ForeignEngine.analyze',return_value={'presenceScore':.02}):
            self.assertFalse(engine.analyze(b'test-only')['candidateDetected'])


if __name__=='__main__':
    unittest.main()
