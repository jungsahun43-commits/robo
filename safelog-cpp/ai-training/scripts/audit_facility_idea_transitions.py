"""Read-only VAL transition audit; no relabels, TRAIN selection or TEST scores."""
from pathlib import Path
from collections import Counter
import json
import sys
import numpy as np
from PIL import Image,ImageDraw
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.facility_idea_research import CONTROL,NAME as IDEA,read,sha,require
from scripts.train_facility_target import dacl_items,TARGETS
from scripts.prepare_facility_data import DACL_MAPPING
from scripts.fetch_rc2119 import write_new
OUTPUT='reports/facility-idea-transition-audit.json'
PRIVATE='runs/facility-idea-transition-audit'

def transitions(truth,old,new):
    truth=np.asarray(truth);old=np.asarray(old);new=np.asarray(new)
    require(truth.shape==old.shape==new.shape and truth.ndim==1 and truth.dtype==old.dtype==new.dtype==np.dtype(bool),'Boolean same-size vectors required')
    return {'photos':len(truth),'old_wrong':int((old!=truth).sum()),'new_wrong':int((new!=truth).sum()),
        'newly_wrong':int(((old==truth)&(new!=truth)).sum()),'corrected':int(((old!=truth)&(new==truth)).sum()),
        'both_wrong':int(((old!=truth)&(new!=truth)).sum()),'both_correct':int(((old==truth)&(new==truth)).sum()),
        'new_fp':int((~truth&new&~old).sum()),'resolved_fp':int((~truth&old&~new).sum()),
        'new_fn':int((truth&old&~new).sum()),'resolved_fn':int((truth&~old&new).sum())}

def main():
    torch.set_num_threads(4);destination=ROOT/PRIVATE
    require(not(ROOT/OUTPUT).exists()and not destination.exists(),'Preserve completed transition audit')
    destination.mkdir(parents=True)
    old_selection=read(ROOT/f'runs/{CONTROL}/TARGET-SELECTION.json')['selected']['per_class']
    new_selection=read(ROOT/f'runs/{IDEA}/TARGET-SELECTION.json')['selected']['per_class']
    items,_=dacl_items('val',640);records={r['stem']:r for r in read(ROOT/'data/dacl10k-yolo/records.json')if r['split']=='val'}
    result=[];private=[];inputs={};review=[]
    for domain in('dacl','damsegment','codebrim'):
        paths=[f'runs/{name}/validation-{domain}.json'for name in(CONTROL,IDEA)]
        inputs.update({p:sha(ROOT/p)for p in paths});documents=[read(ROOT/p)for p in paths]
        truths=np.asarray(documents[0]['targets']);scores=[np.asarray(d['probabilities'])for d in documents]
        require(np.array_equal(truths,np.asarray(documents[1]['targets']))and truths.shape==scores[0].shape==scores[1].shape and np.isfinite(scores).all(),'Original VAL/cache shape/order differs')
        if domain=='dacl':require(np.array_equal(truths,[r['targets']for r in items]),'DACL image cache order differs')
        for k,label in enumerate(TARGETS):
            require(np.isin(truths[:,k],(0,1)).all(),'Primary truth must be asserted')
            truth=truths[:,k].astype(bool);a=old_selection[label]['threshold'];b=new_selection[label]['threshold']
            old=scores[0][:,k]>=a;new=scores[1][:,k]>=b;common=scores[1][:,k]>=a
            result.append({'domain':domain,'class':label,'selected_thresholds':{'old':a,'new':b},
                'selected':transitions(truth,old,new),'same_old_threshold':transitions(truth,old,common),
                'threshold_only_changes_on_new_scores':transitions(truth,common,new)})
            for index in np.flatnonzero(old!=new):
                row={'domain':domain,'class':label,'index':int(index),'truth':int(truths[index,k]),'old_probability':float(scores[0][index,k]),
                    'new_probability':float(scores[1][index,k]),'old_found':bool(old[index]),'new_found':bool(new[index]),
                    'new_found_at_old_threshold':bool(common[index]),'newly_wrong':bool(old[index]==truth[index]and new[index]!=truth[index])}
                if domain=='dacl':
                    record=records[Path(items[index]['image']).stem];source=ROOT/'data'/record['source'].replace('\\','/')
                    annotation=source.parent.parent.parent/'annotations/train'/f'{source.stem}.json'
                    doc=read(annotation);row.update(image=items[index]['image'],native_image=source.relative_to(ROOT).as_posix(),
                        native_annotation=annotation.relative_to(ROOT).as_posix(),native_tags=dict(Counter(s['label']for s in doc['shapes'])),
                        native_image_sha256=sha(source),native_annotation_sha256=sha(annotation))
                    mapped=[name for name in row['native_tags']if DACL_MAPPING.get(name)==label]
                    require(bool(mapped)==bool(truth[index]),'Native author class disagrees with stored photo truth')
                    if label=='concrete_spalling':review.append(row)
                private.append(row)
    # Deterministic balanced interpretation examples. VAL never becomes TRAIN.
    selected=[]
    for changed_to_wrong in(True,False):
        group=sorted([r for r in review if r['newly_wrong']==changed_to_wrong],key=lambda r:r['native_image'])
        selected+=group[:4]
    panels=[]
    for row in selected:
        image_path=ROOT/row['native_image'];doc=read(ROOT/row['native_annotation'])
        with Image.open(image_path)as source:image=source.convert('RGB');image.thumbnail((700,520),Image.Resampling.LANCZOS)
        draw=ImageDraw.Draw(image)
        for shape in doc['shapes']:
            if DACL_MAPPING.get(shape['label'])==row['class']:
                points=[(x*image.width/doc['imageWidth'],y*image.height/doc['imageHeight'])for x,y in shape['points']]
                draw.line(points+[points[0]],fill=(255,40,40),width=3)
        panel=Image.new('RGB',(720,570),'white');panel.paste(image,(10,40))
        ImageDraw.Draw(panel).text((10,8),f"case {len(panels)+1}: {'NEW ERROR'if row['newly_wrong']else'CORRECTED'} | truth={row['truth']} | p {row['old_probability']:.3f}->{row['new_probability']:.3f}",fill='black')
        panels.append(panel)
    sheet=Image.new('RGB',(1440,570*((len(panels)+1)//2)),'white')
    for i,panel in enumerate(panels):sheet.paste(panel,((i%2)*720,(i//2)*570))
    sheet.save(destination/'native-review.png')
    write_new(destination/'changed-cases.json',{'items':private,'visual_review_cases':selected,'human_labels_created':0,'val_photos_used_in_training':0})
    public={'schema':'facility_idea_transition_audit_v1','split':'val','control':CONTROL,'candidate':IDEA,'input_sha256':inputs,
        'transitions':result,'native_dacl_changed_case_count':sum(r['domain']=='dacl'for r in private),
        'native_annotation_mapping_agrees':True,'local_visual_review_cases':len(selected),'private_case_sha256':sha(destination/'changed-cases.json'),
        'native_overlay_sha256':sha(destination/'native-review.png'),'source_test_used':False,'new_training_epochs':0,
        'new_labels_created':0,'val_cases_used_in_training_or_sampling':False,
        'interpretation':'Score/threshold transition description only; native tags do not establish cause or annotation completeness. No primary forgetting causal claim.'}
    write_new(ROOT/OUTPUT,public);print(json.dumps({'transitions':result,'review_cases':len(selected)}),flush=True)

if __name__=='__main__':main()
