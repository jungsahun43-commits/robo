"""Explicitly selected calibrated decision rule; existing v1 service stays immutable."""
import argparse
import hashlib
import io
import math
from pathlib import Path
from threading import RLock

import torch
from .conveyor_foreign import ARCH, CLASS, SIZE, create_model
from .conveyor_foreign_inference import ForeignEngine
from .conveyor_foreign_server import create_app


class CalibratedEngine(ForeignEngine):
    def __init__(self,checkpoint,expected_sha256):
        path = Path(checkpoint);raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest!=expected_sha256:
            raise ValueError('Checkpoint SHA256 does not match the selected model')
        state = torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
        threshold = state.get('threshold',float('nan'))
        if (state.get('architecture')!=ARCH or state.get('class')!=CLASS or state.get('input_size')!=SIZE
                or state.get('decision_schema')!='safelog.foreign-calibrated.v3'
                or state.get('threshold_fitted_using_test',True) or not math.isfinite(threshold) or not 0<threshold<1):
            raise ValueError('Unsupported calibrated checkpoint contract')
        self.model = create_model(False).eval()
        self.model.load_state_dict(state['state_dict'],strict=True)
        self.model_sha256,self.model_name = digest,path.parent.name
        self.threshold = threshold
        self.calibration_protocol_sha256 = state['calibration_protocol_sha256']
        self.lock = RLock()

    def analyze(self,payload):
        result = super().analyze(payload)
        result.update(schema='safelog.equipment.conveyor-foreign.v2',threshold=self.threshold,
            candidateDetected=result['presenceScore']>self.threshold,
            decisionThresholdSource='Model calibration checkpoint, not per-photo or request override',
            calibrationProtocolSha256=self.calibration_protocol_sha256)
        return result


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--sha256',required=True);parser.add_argument('--port',type=int,default=8772)
    args = parser.parse_args();torch.set_num_threads(4)
    import uvicorn
    uvicorn.run(create_app(CalibratedEngine(args.checkpoint,args.sha256)),host='127.0.0.1',port=args.port)


if __name__=='__main__':
    main()
