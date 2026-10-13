"""Same API and ONE checkpoint with an explicit belt inference-policy identity."""
import argparse
import base64
import hashlib
import io
from PIL import Image
import torch
from .unified_main_server import MainEngine, create_app, decode
from .belt_tiles import MODES, predict, policy


class TilesEngine(MainEngine):
    def __init__(self, checkpoint, expected, mode):
        self.inference_policy = policy(mode); self.mode = mode
        super().__init__(checkpoint, expected)

    def analyze(self, task, encoded):
        if task != 'belt_damage' or self.mode == 'global_only':
            result = super().analyze(task, encoded); views = 0
        else:
            rgb, raw = decode(encoded)
            with self.lock: probability, box, views = predict(self.model, rgb, self.mode, 'cpu')
            left, top, width, height = box; valid = probability[0, 0, top:top+height, left:left+width]
            score = float(valid.max()); threshold = self.thresholds[task]
            heat = Image.fromarray((valid.numpy()*255).astype('uint8')).resize(rgb.size, Image.Resampling.BILINEAR)
            buffer = io.BytesIO(); heat.save(buffer, format='PNG')
            result = {'schema': 'safelog.unified-main.v1', 'inspectionTask': task, 'neuralModelCount': 1, 'sharedBackboneCount': 1,
                'modelName': 'unified-main-v1', 'modelSha256': self.digest,
                'classResults': [{'className': 'belt_crack', 'score': score, 'threshold': threshold, 'candidateDetected': score > threshold}],
                'candidateDetected': score > threshold, 'inputSha256': hashlib.sha256(raw).hexdigest(), 'inputWidth': rgb.width, 'inputHeight': rgb.height,
                'heatmapPngBase64': base64.b64encode(buffer.getvalue()).decode(), 'heatmapIsConfirmedDefectLocalization': False,
                'scoreIsCalibratedProbability': False, 'otherTasksEvaluated': False, 'mechanicalFailureState': 'not_assessed',
                'humanConfirmationRequired': True, 'fieldPerformanceValidated': False, 'beltPhotoFprValidated': False,
                'evidenceScope': 'belt_author_mask_review', 'trainedDatasetSources': ['DACL10k', 'DamSegment', 'CODEBRIM', 'BeltCrack', 'Mendeley-s25x2bnshz']}
        result.update(inferencePolicyVersion=self.inference_policy['version'], inferencePolicySha256=self.inference_policy['sha256'],
                      configuredBeltInferenceMode=self.mode, nativeDetailViewsEvaluated=views)
        return result


def tiles_app(engine):
    app = create_app(engine)
    @app.get('/v1/inference-policy')
    def inference_policy(): return engine.inference_policy
    return app


if __name__ == '__main__':
    import uvicorn
    parser = argparse.ArgumentParser(); parser.add_argument('--checkpoint', required=True); parser.add_argument('--sha256', required=True)
    parser.add_argument('--mode', choices=MODES, required=True); parser.add_argument('--port', type=int, default=8775)
    args = parser.parse_args(); torch.set_num_threads(4)
    uvicorn.run(tiles_app(TilesEngine(args.checkpoint, args.sha256, args.mode)), host='127.0.0.1', port=args.port)
