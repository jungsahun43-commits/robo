"""Inspect new-model scores independently of the application's default model."""
from pathlib import Path
import argparse
import json
import sys
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from safelog_ai.convnext_facility import ConvnextResearchPresence,ARCH,require
from scripts.facility_convnext import NAME,read,sha
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu');args=parser.parse_args();torch.set_num_threads(4)
    run=ROOT/'runs'/NAME;checkpoint=run/'best.pt';proof=read(ROOT/'reports/facility-convnext-study-verification.json')
    require(proof['status']=='passed'and proof['weights_sha256']==sha(checkpoint),'Verified research checkpoint required')
    adapter=ConvnextResearchPresence(checkpoint,args.device)
    with Image.open(args.image)as image:inputs=adapter.transform(image.convert('RGB'))[None].to(args.device)
    with torch.inference_mode():probability=adapter.model(inputs).sigmoid()[0].cpu().tolist()
    require(all(0<=p<=1 for p in probability),'Invalid probability')
    print(json.dumps({'schema':'facility_research_scores_v1','architecture':ARCH,'weights_sha256':sha(checkpoint),
        'scores':dict(zip(adapter.classes,probability)),'human_review_required':True,'application_profile_updated':False,
        'probabilities_are_verified_industrial_accuracy':False},ensure_ascii=True),flush=True)
if __name__=='__main__':main()
