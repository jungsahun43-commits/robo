"""One main checkpoint for the app's existing three HTTP operations."""
import argparse
import base64
import binascii
import hashlib
import io
from pathlib import Path
from threading import RLock
from typing import Literal
import numpy as np
from PIL import Image,ImageOps,UnidentifiedImageError
import torch
from torch.nn import functional as F
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
from .unified_main import ARCH,TASKS,FACILITY_CLASSES,UnifiedMain

LABELS={'concrete_crack':'표면 균열','concrete_spalling':'표면 박락','rust_stain':'녹 얼룩','exposed_rebar':'철근 노출',
        'wet_surface':'젖은 표면','efflorescence':'백화','surface_cavity':'표면 공동','belt_crack':'벨트 손상','conveyor_foreign_object':'이물질'}


def decode(encoded):
    if encoded.startswith('data:'):
        prefix,sep,encoded=encoded.partition(',')
        if not sep or prefix not in ['data:image/jpeg;base64','data:image/png;base64']:raise ValueError('JPEG/PNG data URL required')
    raw=base64.b64decode(encoded,validate=True)
    if not raw or len(raw)>8*1024*1024:raise ValueError('Photo must be nonempty and at most8MB')
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.format not in ['JPEG','PNG'] or image.width*image.height>12_000_000:raise ValueError('JPEG/PNG up to12MP required')
            return ImageOps.exif_transpose(image).convert('RGB'),raw
    except (UnidentifiedImageError,OSError,Image.DecompressionBombError) as error:raise ValueError('Readable JPEG/PNG required') from error


def tensor(rgb,task):
    w,h=rgb.size;box=None
    if task=='facility_surface':prepared=rgb.resize((640,640),Image.Resampling.BICUBIC)
    else:
        scale=640/max(w,h);rw,rh=max(1,round(w*scale)),max(1,round(h*scale));left,top=(640-rw)//2,(640-rh)//2
        prepared=Image.new('RGB',(640,640),(123,116,103));prepared.paste(rgb.resize((rw,rh),Image.Resampling.BILINEAR),(left,top));box=(left,top,rw,rh)
    values=torch.from_numpy(np.array(prepared)).permute(2,0,1).float().unsqueeze(0)/255
    return (values-values.new_tensor([.485,.456,.406]).view(1,3,1,1))/values.new_tensor([.229,.224,.225]).view(1,3,1,1),box


class MainEngine:
    def __init__(self,path,expected):
        raw=Path(path).read_bytes();digest=hashlib.sha256(raw).hexdigest()
        if digest!=expected:raise ValueError('Main checkpoint SHA256 mismatch')
        state=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
        if state.get('architecture')!=ARCH or state.get('tasks')!=list(TASKS) or not state.get('one_shared_backbone'):raise ValueError('One shared main model contract required')
        self.model=UnifiedMain().eval();self.model.load_state_dict(state['state_dict'],strict=True)
        self.thresholds=state['thresholds'];self.digest=digest;self.lock=RLock()
    def analyze(self,task,encoded):
        rgb,raw=decode(encoded);images,box=tensor(rgb,task)
        with self.lock,torch.inference_mode():output=self.model.forward_task(images,task)
        heat=None;classes=[]
        if task=='facility_surface':
            scores=output[0].sigmoid()[0].tolist()
            classes=[{'className':c,'score':s,'threshold':self.thresholds[task][c],'candidateDetected':s>self.thresholds[task][c]} for c,s in zip(FACILITY_CLASSES,scores)]
        elif task=='conveyor_foreign':
            score=float(output.sigmoid()[0]);threshold=self.thresholds[task]
            classes=[{'className':'conveyor_foreign_object','score':score,'threshold':threshold,'candidateDetected':score>threshold}]
        else:
            left,top,w,h=box;prob=output.sigmoid()[0,0,top:top+h,left:left+w];score=float(prob.max());threshold=self.thresholds[task]
            classes=[{'className':'belt_crack','score':score,'threshold':threshold,'candidateDetected':score>threshold}]
            image=Image.fromarray((prob.numpy()*255).astype('uint8')).resize(rgb.size,Image.Resampling.BILINEAR);buffer=io.BytesIO();image.save(buffer,format='PNG');heat=base64.b64encode(buffer.getvalue()).decode()
        return {'schema':'safelog.unified-main.v1','inspectionTask':task,'neuralModelCount':1,'sharedBackboneCount':1,
            'modelName':'unified-main-v1','modelSha256':self.digest,'classResults':classes,
            'candidateDetected':any(c['candidateDetected'] for c in classes),'inputSha256':hashlib.sha256(raw).hexdigest(),
            'inputWidth':rgb.width,'inputHeight':rgb.height,'heatmapPngBase64':heat,'heatmapIsConfirmedDefectLocalization':False,
            'scoreIsCalibratedProbability':False,'otherTasksEvaluated':False,'mechanicalFailureState':'not_assessed',
            'humanConfirmationRequired':True,'fieldPerformanceValidated':False,'beltPhotoFprValidated':False,
            'evidenceScope':'belt_author_mask_review' if task=='belt_damage' else 'photo_presence',
            'trainedDatasetSources':['DACL10k','DamSegment','CODEBRIM','BeltCrack','Mendeley-s25x2bnshz']}


class HazardRequest(BaseModel):
    image:str=Field(min_length=1,max_length=12_000_000)
    inspectionTask:Literal['facility_surface','belt_damage','conveyor_foreign']
    userMemo:str=Field(default='',max_length=10000)
    promptVersion:str=Field(min_length=1,max_length=80)
class CompareRequest(BaseModel):
    beforeImage:str=Field(min_length=1,max_length=12_000_000)
    afterImage:str=Field(min_length=1,max_length=12_000_000)
    inspectionTask:Literal['facility_surface','belt_damage','conveyor_foreign']
    actionNote:str=Field(default='',max_length=10000)
    promptVersion:str=Field(min_length=1,max_length=80)
class SummaryRequest(BaseModel):
    inspectionContext:str=Field(min_length=1,max_length=20000)
    promptVersion:str=Field(min_length=1,max_length=80)


def create_app(engine=None):
    app=FastAPI(title='SafeLog unified main',version='1.0.0')
    def analysis(task,image):
        if engine is None:raise HTTPException(503,'Explicit main checkpoint and SHA required')
        try:return engine.analyze(task,image)
        except (ValueError,binascii.Error) as error:raise HTTPException(422,str(error)) from error
    @app.get('/health')
    def health():return {'ready':engine is not None,'neuralModelCount':1 if engine else 0,'sharedBackboneCount':1 if engine else 0,
        'inspectionTasks':list(TASKS),'modelSha256':engine.digest if engine else None,'humanConfirmationRequired':True}
    @app.post('/v1/main/analyze')
    @app.post('/v1/analyze-hazard')
    def hazard(body:HazardRequest):
        result=analysis(body.inspectionTask,body.image);candidates=[c for c in result['classResults'] if c['candidateDetected']]
        names=[LABELS[c['className']]+' 후보' for c in candidates];score=max(c['score'] for c in result['classResults'])
        result.update(hazardCategory=' / '.join(names) if names else '외관 이상 후보 미검출',riskLevel=3 if names else 1,
            detectedHazards=names,suggestedDescription='사진에서 '+', '.join(names)+'가 제안되었습니다.' if names else '해당 점검 항목에서 AI 외관 이상 후보를 찾지 못했습니다.',
            suggestedAction='점검자가 사진과 현장을 확인한 뒤 담당자의 조치와 재점검 내용을 기록하세요.',
            confidence=score,riskLevelMethod='rule_template_not_learned_severity',promptVersion=body.promptVersion)
        return result
    @app.post('/v1/compare-action')
    def compare(body:CompareRequest):
        before=analysis(body.inspectionTask,body.beforeImage);after=analysis(body.inspectionTask,body.afterImage)
        resolved=before['candidateDetected'] and not after['candidateDetected']
        remain=[LABELS[c['className']]+' 후보' for c in after['classResults'] if c['candidateDetected']]
        return {'schema':'safelog.unified-main.comparison.v1','likelyResolved':resolved,'remainingRisks':remain,
            'assessment':'조치 후 사진에서 이전 점검 항목의 AI 후보가 미검출되었습니다. 점검자의 최종 확인이 필요합니다.' if resolved else '조치 전후 사진의 AI 후보 결과를 확인하고 점검자가 조치 상태를 판단해야 합니다.',
            'confidence':max(c['score'] for c in after['classResults']),'scoreIsCalibratedProbability':False,
            'modelName':'unified-main-v1','modelSha256':engine.digest,'inspectionTask':body.inspectionTask,
            'beforeCandidateDetected':before['candidateDetected'],'afterCandidateDetected':after['candidateDetected'],
            'humanConfirmationRequired':True,'mechanicalFailureState':'not_assessed','promptVersion':body.promptVersion,
            'comparisonMethod':'Same fixed model inspection scores, not a separately trained change detector'}
    @app.post('/v1/summarize')
    def summary(body:SummaryRequest):
        return {'summary':'점검·조치·재점검 기록 요약\n'+body.inspectionContext,'keyRisks':[],
            'generationMethod':'rule_template_not_trained_LLM','modelName':'unified-main-v1/report-template',
            'promptVersion':body.promptVersion,'humanConfirmationRequired':True}
    return app


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--sha256',required=True);parser.add_argument('--port',type=int,default=8775)
    args=parser.parse_args();torch.set_num_threads(4)
    import uvicorn
    uvicorn.run(create_app(MainEngine(args.checkpoint,args.sha256)),host='127.0.0.1',port=args.port)
