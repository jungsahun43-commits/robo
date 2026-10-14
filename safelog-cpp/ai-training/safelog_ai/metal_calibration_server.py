"""Same one-model prototype with explicit20% normalCAL budget for metal plate."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from .equipment_expansion_server import ExpansionEngine,create_expansion_app

CATEGORY='mpdd_metal_plate'


class MetalCalibrationEngine(ExpansionEngine):
    def __init__(self,checkpoint,expected,evaluation=None):
        super().__init__(checkpoint,expected,evaluation)
        state=torch.load(Path(checkpoint),map_location='cpu',weights_only=True)
        if state.get('metal_refinement_variant')!='calibration_only20' or state.get('normalCAL_rank')!=17:
            raise ValueError('Explicit normal-only calibration_only20 refinement required')
        setting={'version':'safelog-metal-plate-calibration-v1','category':CATEGORY,'parentEquipmentPolicySha256':self.equipment_policy['sha256'],
            'neuralModelCount':1,'sharedBackboneCount':1,'neuralParametersChanged':False,
            'photoScore':'same mean top1percent reconstruction residual','normalCALCount':20,'normalCALAlpha':.20,'normalCALRank':17,
            'pixelThresholdChanged':False,'heatmapChanged':False,'TESTThresholdTuning':False}
        self.metal_policy={**setting,'sha256':hashlib.sha256(json.dumps(setting,sort_keys=True,separators=(',',':')).encode()).hexdigest()}

    def analyze_equipment(self,category,encoded):
        result=super().analyze_equipment(category,encoded)
        if category==CATEGORY:
            result.update(inferencePolicyVersion=self.metal_policy['version'],inferencePolicySha256=self.metal_policy['sha256'],
                          normalCALAlpha=.20,normalCALRank=17)
        return result


def create_metal_app(engine=None):
    app=create_expansion_app(engine)
    # Replace just the policy-discovery endpoint, keeping all existing requests.
    app.router.routes=[r for r in app.router.routes if getattr(r,'path',None)!='/v1/inference-policy']
    @app.get('/v1/inference-policy')
    def policies():
        if engine is None:
            from fastapi import HTTPException
            raise HTTPException(503,'Model not loaded')
        return {'legacyBelt':engine.inference_policy,'equipment':engine.equipment_policy,'metalPlate':engine.metal_policy}
    return app


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--sha256',required=True)
    parser.add_argument('--evaluation');parser.add_argument('--port',type=int,default=8775);args=parser.parse_args();torch.set_num_threads(4)
    import uvicorn
    uvicorn.run(create_metal_app(MetalCalibrationEngine(args.checkpoint,args.sha256,args.evaluation)),host='127.0.0.1',port=args.port)
