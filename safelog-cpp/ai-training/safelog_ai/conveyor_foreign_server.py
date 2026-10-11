"""Separate opt-in endpoint; scene selection is explicit in each request."""
import argparse
import base64
import binascii
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch

from .belt_inference import MAX_IMAGE_BYTES
from .conveyor_foreign_inference import ForeignEngine


class ForeignRequest(BaseModel):
    image: str = Field(min_length=1, max_length=12_000_000)
    equipmentCategory: str
    sceneType: str
    promptVersion: str = Field(min_length=1, max_length=80)


def create_app(engine=None):
    app = FastAPI(title='SafeLog conveyor foreign-object presence AI', version='1.0.0')

    @app.get('/health')
    def health():
        return {'ready':engine is not None, 'className':'conveyor_foreign_object',
            'sceneType':'iron_ore_closeup', 'modelSha256':engine.model_sha256 if engine is not None else None,
            'mechanicalFailureState':'not_assessed', 'fieldPerformanceValidated':False}

    @app.post('/v1/equipment/conveyor/foreign/analyze')
    def analyze(body:ForeignRequest):
        if body.equipmentCategory!='conveyor_belt' or body.sceneType!='iron_ore_closeup':
            raise HTTPException(status_code=422, detail='Explicit conveyor_belt / iron_ore_closeup scene required')
        if engine is None:
            raise HTTPException(status_code=503, detail='Select the foreign-object checkpoint and SHA at startup')
        encoded = body.image
        if encoded.startswith('data:'):
            prefix, separator, encoded = encoded.partition(',')
            if not separator or prefix not in ('data:image/jpeg;base64', 'data:image/png;base64'):
                raise HTTPException(status_code=422, detail='JPEG/PNG base64 data URL required')
        try:
            payload = base64.b64decode(encoded, validate=True)
            if len(payload)>MAX_IMAGE_BYTES:
                raise HTTPException(status_code=413, detail='Image exceeds8MB')
            result = engine.analyze(payload)
        except (binascii.Error, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        result['promptVersion'] = body.promptVersion
        return result
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--port', type=int, default=8772)
    args = parser.parse_args()
    torch.set_num_threads(4)
    import uvicorn
    uvicorn.run(create_app(ForeignEngine(args.checkpoint, args.sha256)), host='127.0.0.1', port=args.port)


if __name__=='__main__':
    main()
