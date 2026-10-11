"""Pinned local CPU photo classifier for the iron-ore conveyor research scope."""
import hashlib
import io
from pathlib import Path
from threading import RLock

import torch

from .belt_inference import decode_photo
from .conveyor_foreign import ARCH, CLASS, SIZE, THRESHOLD, create_model, letterbox, normalize


class ForeignEngine:
    def __init__(self, checkpoint, expected_sha256):
        path = Path(checkpoint)
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest!=expected_sha256:
            raise ValueError('Checkpoint SHA256 does not match the selected model')
        state = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
        if (state.get('architecture')!=ARCH or state.get('class')!=CLASS
                or state.get('input_size')!=SIZE or state.get('threshold')!=THRESHOLD):
            raise ValueError('Unsupported foreign-object checkpoint contract')
        self.model = create_model(False).eval()
        self.model.load_state_dict(state['state_dict'], strict=True)
        self.model_sha256, self.model_name = digest, path.parent.name
        self.lock = RLock()

    def analyze(self, payload):
        image = decode_photo(payload)
        tensor = normalize(letterbox(image).unsqueeze(0))
        with self.lock, torch.inference_mode():
            score = self.model(tensor).flatten().sigmoid()
        if score.numel()!=1 or not torch.isfinite(score).all():
            raise RuntimeError('Nonfinite or invalid classifier output')
        value = float(score[0])
        return {'schema':'safelog.equipment.conveyor-foreign.v1', 'equipmentCategory':'conveyor_belt',
            'sceneType':'iron_ore_closeup', 'analysisType':'foreign_object_presence', 'className':CLASS,
            'candidateDetected':value>THRESHOLD, 'presenceScore':value, 'threshold':THRESHOLD,
            'scoreIsCalibratedProbability':False, 'scoreIsMechanicalFailureProbability':False,
            'mechanicalFailureState':'not_assessed', 'fieldPerformanceValidated':False,
            'humanConfirmationRequired':True, 'localizationAvailable':False,
            'modelName':self.model_name, 'modelSha256':self.model_sha256,
            'inputSha256':hashlib.sha256(payload).hexdigest(), 'inputWidth':image.width, 'inputHeight':image.height,
            'scope':'Close-up iron-ore conveyor flow; foreign-object photo presence only, no health verdict'}
