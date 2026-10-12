"""Pinned four-category PCB CPU server, separate from belt and internal-fault diagnosis."""
import argparse
import base64
import binascii
import hashlib
import io
import json
import math
from pathlib import Path
from threading import RLock

import numpy as np
from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor,normalize
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field

from .belt_inference import decode_photo,MAX_IMAGE_BYTES
from .pcb_reverse import ARCH,SIZE,CATEGORIES,create_teacher,ReverseStudent,anomaly_maps


def photo_tensor(rgb):
    resized=rgb.resize((SIZE,SIZE),Image.Resampling.BILINEAR)
    return normalize(to_tensor(resized),[.485,.456,.406],[.229,.224,.225]).unsqueeze(0)


class PcbEngine:
    def __init__(self,registry):
        path=Path(registry).resolve();records=json.loads(path.read_text(encoding='utf-8'))
        if set(records)!=set(CATEGORIES):
            raise ValueError('Registry must explicitly pin exactly PCB1-4')
        self.teacher=None;self.students={};self.records={};self.lock=RLock()
        teacher_digest=None
        for category,record in records.items():
            checkpoint=(path.parent/record['path']).resolve()
            if not checkpoint.is_relative_to(path.parent):
                raise ValueError('Checkpoint path must stay inside registry directory')
            raw=checkpoint.read_bytes();digest=hashlib.sha256(raw).hexdigest()
            if digest!=record['sha256']:
                raise ValueError('Checkpoint SHA256 mismatch')
            state=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
            if state.get('architecture')!=ARCH or state.get('equipment_category')!=category or state.get('input_size')!=SIZE:
                raise ValueError('Unsupported PCB checkpoint contract')
            threshold=state['threshold']
            if not math.isfinite(threshold) or threshold<=0:
                raise ValueError('Positive finite normal-calibration threshold required')
            if self.teacher is None:
                self.teacher=create_teacher(False);self.teacher.load_state_dict(state['teacher'],strict=True)
                teacher_digest=state['teacher_sha256']
            elif state['teacher_sha256']!=teacher_digest or any(not torch.equal(v,state['teacher'][k]) for k,v in self.teacher.state_dict().items()):
                raise ValueError('Category teachers must match exactly')
            student=ReverseStudent().eval();student.load_state_dict(state['student'],strict=True)
            self.students[category]=student
            self.records[category]={'modelSha256':digest,'threshold':threshold,'pixelThreshold':state['pixel_threshold']}

    def analyze(self,category,payload):
        if category not in self.students:
            raise ValueError('Registered PCB1-4 only')
        rgb=decode_photo(payload);tensor=photo_tensor(rgb);record=self.records[category]
        with self.lock,torch.inference_mode():
            target=self.teacher(tensor);maps=anomaly_maps(self.students[category](target),target)[0]
        if not torch.isfinite(maps).all():
            raise RuntimeError('Nonfinite model output')
        score=float(maps.max());scaled=(maps/record['threshold']).clamp(0,1).numpy()
        heatmap=Image.fromarray((scaled*255).astype('uint8')).resize(rgb.size,Image.Resampling.BILINEAR)
        buffer=io.BytesIO();heatmap.save(buffer,format='PNG')
        return {'schema':'safelog.equipment.pcb-visual.v2','equipmentCategory':category,
            'sceneType':'registered_pcb_closeup','className':'pcb_appearance_anomaly','analysisType':'appearance_anomaly',
            'candidateDetected':score>record['threshold'],'anomalyScore':score,'threshold':record['threshold'],
            'pixelThreshold':record['pixelThreshold'],'scoreIsProbability':False,'mechanicalFailureState':'not_assessed',
            'humanConfirmationRequired':True,'fieldPerformanceValidated':False,
            'modelName':'pcb-reverse-v2/'+category,'modelSha256':record['modelSha256'],
            'inputSha256':hashlib.sha256(payload).hexdigest(),'inputWidth':rgb.width,'inputHeight':rgb.height,
            'heatmapPngBase64':base64.b64encode(buffer.getvalue()).decode(),
            'heatmapScale':'Fixed feature distance / category normalCAL photo threshold, clipped0..255',
            'heatmapIsConfirmedDefectLocalization':False,
            'scope':'Specified registered PCB type close-up; no general cabinet, electrical operation or internal-fault verdict'}


class Request(BaseModel):
    image:str=Field(min_length=1,max_length=12_000_000)
    equipmentCategory:str
    sceneType:str
    promptVersion:str=Field(min_length=1,max_length=80)


def create_app(engine=None):
    app=FastAPI(title='SafeLog registered PCB appearance AI',version='2.0.0')
    @app.get('/health')
    def health():
        return {'ready':engine is not None,'categories':list(CATEGORIES),
                'models':engine.records if engine is not None else {},'mechanicalFailureState':'not_assessed'}
    @app.post('/v1/equipment/pcb/analyze')
    def analyze(body:Request):
        if body.equipmentCategory not in CATEGORIES or body.sceneType!='registered_pcb_closeup':
            raise HTTPException(422,'Select a registered PCB1-4 close-up scene')
        if engine is None:
            raise HTTPException(503,'Explicit model registry and hashes required')
        encoded=body.image
        if encoded.startswith('data:'):
            prefix,separator,encoded=encoded.partition(',')
            if not separator or prefix not in ('data:image/png;base64','data:image/jpeg;base64'):
                raise HTTPException(422,'JPEG/PNG base64 data URL required')
        try:
            raw=base64.b64decode(encoded,validate=True)
            if len(raw)>MAX_IMAGE_BYTES:
                raise HTTPException(413,'Image exceeds8MB')
            result=engine.analyze(body.equipmentCategory,raw)
        except (binascii.Error,ValueError) as error:
            raise HTTPException(422,str(error)) from error
        result['promptVersion']=body.promptVersion
        return result
    return app


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--registry',type=Path,required=True)
    parser.add_argument('--port',type=int,default=8773);args=parser.parse_args();torch.set_num_threads(4)
    import uvicorn
    uvicorn.run(create_app(PcbEngine(args.registry)),host='127.0.0.1',port=args.port)


if __name__=='__main__':
    main()
