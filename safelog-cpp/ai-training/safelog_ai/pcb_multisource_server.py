"""One SHA-pinned CPU neural model trained on two official PCB sources."""
import argparse
import base64
import binascii
import hashlib
import io
import math
from pathlib import Path
from threading import RLock
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from fastapi import FastAPI, HTTPException
from .pcb_multisource import ARCH, CATEGORIES, create_teacher, ReverseStudent, anomaly_maps
from .pcb_reverse_server import photo_tensor, Request
from .retention_distillation import state_sha256


def decode(payload):
    if not payload or len(payload) > 8*1024*1024:
        raise ValueError('Photo must be nonempty and at most8MB')
    try:
        with Image.open(io.BytesIO(payload)) as source:
            if source.format not in ['JPEG', 'PNG'] or source.width*source.height > 40_000_000:
                raise ValueError('JPEG/PNG up to40 megapixels required')
            return ImageOps.exif_transpose(source).convert('RGB')
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError('Readable JPEG/PNG photo required') from error


class PooledEngine:
    def __init__(self, checkpoint, expected_sha):
        raw = Path(checkpoint).read_bytes(); digest = hashlib.sha256(raw).hexdigest()
        if digest != expected_sha:
            raise ValueError('Selected single PCB model SHA256 mismatch')
        state = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
        if (state.get('architecture') != ARCH or state.get('input_size') != 256
                or state.get('categories') != list(CATEGORIES) or not state.get('one_shared_student_network')
                or state.get('trained_source_datasets') != ['VisA', 'RobustAD']):
            raise ValueError('Two-source single PCB network contract required')
        self.teacher = create_teacher(False); self.student = ReverseStudent().eval()
        self.teacher.load_state_dict(state['teacher'], strict=True); self.student.load_state_dict(state['student'], strict=True)
        if state_sha256(self.teacher) != state['teacher_sha256'] or state_sha256(self.student) != state['final_student_sha256']:
            raise ValueError('Teacher/student state hashes mismatch')
        if set(state['operating_points']) != set(CATEGORIES):
            raise ValueError('All registered PCB operating points required')
        self.records = {}; self.lock = RLock(); self.digest = digest
        for category, point in state['operating_points'].items():
            if any(not math.isfinite(point[k]) or point[k] <= 0 for k in ['threshold', 'pixel_threshold']):
                raise ValueError('Positive finite normal-CAL operating points required')
            self.records[category] = {'modelSha256': digest, 'threshold': point['threshold'], 'pixelThreshold': point['pixel_threshold']}

    def analyze(self, category, raw):
        rgb = decode(raw); tensor = photo_tensor(rgb); point = self.records[category]
        with self.lock, torch.inference_mode():
            target = self.teacher(tensor); maps = anomaly_maps(self.student(target), target)[0]
        if not torch.isfinite(maps).all():
            raise RuntimeError('Nonfinite PCB output')
        score = float(maps.max()); display = (maps/point['threshold']).clamp(0, 1).numpy()
        heat = Image.fromarray((display*255).astype('uint8')).resize(rgb.size, Image.Resampling.BILINEAR)
        buffer = io.BytesIO(); heat.save(buffer, format='PNG')
        return {'schema': 'safelog.equipment.pcb-visual.v3', 'equipmentCategory': category,
            'sceneType': 'registered_pcb_closeup', 'className': 'pcb_appearance_anomaly', 'analysisType': 'appearance_anomaly',
            'candidateDetected': score > point['threshold'], 'anomalyScore': score, 'threshold': point['threshold'],
            'pixelThreshold': point['pixelThreshold'], 'scoreIsProbability': False,
            'modelName': 'pcb-multisource-v3', 'modelSha256': self.digest,
            'trainedSourceDatasets': ['VisA', 'RobustAD'], 'neuralModelCount': 1,
            'categorySpecificNeuralWeights': False, 'categorySpecificNormalCalibration': True,
            'inputSha256': hashlib.sha256(raw).hexdigest(), 'inputWidth': rgb.width, 'inputHeight': rgb.height,
            'heatmapPngBase64': base64.b64encode(buffer.getvalue()).decode(),
            'heatmapScale': 'Feature distance / fixed category normalCAL photo threshold, clipped0..255',
            'heatmapIsConfirmedDefectLocalization': False, 'mechanicalFailureState': 'not_assessed',
            'localizationReadyForApp': False, 'heatmapUse': 'feature_difference_review_only',
            'humanConfirmationRequired': True, 'fieldPerformanceValidated': False,
            'scope': 'Only the five specified source PCB types in registered close-up views; no arbitrary PCB, internal fault or operation verdict'}


def create_app(engine=None):
    app = FastAPI(title='SafeLog single two-source PCB model', version='3.0.0')
    @app.get('/health')
    def health():
        return {'ready': engine is not None, 'neuralModelCount': 1 if engine else 0,
                'categories': list(CATEGORIES), 'models': engine.records if engine else {},
                'trainedSourceDatasets': ['VisA', 'RobustAD'], 'mechanicalFailureState': 'not_assessed'}
    @app.post('/v1/equipment/pcb/analyze')
    def analyze(body: Request):
        if body.equipmentCategory not in CATEGORIES or body.sceneType != 'registered_pcb_closeup':
            raise HTTPException(422, 'Explicitly select a supported source PCB type and close-up scene')
        if engine is None:
            raise HTTPException(503, 'Explicit single model path and SHA256 required')
        encoded = body.image
        if encoded.startswith('data:'):
            prefix, sep, encoded = encoded.partition(',')
            if not sep or prefix not in ['data:image/jpeg;base64', 'data:image/png;base64']:
                raise HTTPException(422, 'JPEG/PNG data URL required')
        try:
            raw = base64.b64decode(encoded, validate=True)
            if len(raw) > 8*1024*1024:
                raise HTTPException(413, 'Photo exceeds8MB')
            value = engine.analyze(body.equipmentCategory, raw)
        except (ValueError, binascii.Error) as error:
            raise HTTPException(422, str(error)) from error
        value['promptVersion'] = body.promptVersion
        return value
    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--sha256', required=True); parser.add_argument('--port', type=int, default=8774)
    args = parser.parse_args(); torch.set_num_threads(4)
    import uvicorn
    uvicorn.run(create_app(PooledEngine(args.checkpoint, args.sha256)), host='127.0.0.1', port=args.port)
