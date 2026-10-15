"""FIT-only feasibility preview. No gradients, CAL/TEST photos or model inference."""
from collections import Counter
from datetime import datetime,timezone
import hashlib
from pathlib import Path
import sys
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from safelog_ai.target_synthetic import VERSION,synthesize
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read
DATA=ROOT/'data/industrial-target-synthetic-v19'
NORMAL=ROOT/'data/industrial-neighbor-surfaces-v15/manifest.json'
SOURCE=ROOT/'data/industrial-ksdd1-v18/manifest-r1.json'
REPORT=ROOT/'reports/industrial-target-synthetic-v19-preview.json'
CATEGORIES=('mpdd_bracket_black','mpdd_tubes')


def main():
    assert not DATA.exists() and not REPORT.exists();DATA.mkdir()
    normal=[]
    for c in CATEGORIES:
        pool=[r for r in read(NORMAL)['rows'] if r['category']==c and r['phase']=='fit' and r['label']==0]
        normal.extend(sorted(pool,key=lambda r:hashlib.sha256(('preview19:'+r['id']).encode()).hexdigest())[:2])
    shapes=sorted([r for r in read(SOURCE)['rows'] if r['phase']=='fit' and r['label']],key=lambda r:r['id'])[:3]
    new(DATA/'fixed-preview-inputs.json',{'schema':VERSION,'UTC':datetime.now(timezone.utc).isoformat(),'normal_manifest_sha256':sha(NORMAL),'source_manifest_sha256':sha(SOURCE),
        'source_code_sha256':{n:sha(ROOT/n) for n in ('safelog_ai/target_synthetic.py','scripts/preview_target_synthetic.py')},
        'normal_FIT_ids':[r['id'] for r in normal],'actual_shape_FIT_ids':[r['id'] for r in shapes],
        'procedure':'single declared residual extraction/inpaint estimate, extent24to72, angle0/45/90/135, foreground98percent placement, amplitude16to48, observed support>=16pixels, no TEST-derived settings',
        'seed_rule':'SHA256(preview19:normalID:sourceID) first8hex','sample_count_requested':len(normal)*len(shapes),'model_optimizer_updates':0})
    samples=[];skipped=[];visuals={c:[] for c in CATEGORIES}
    for nr in normal:
        assert nr['phase']=='fit' and nr['publisher_split']=='train' and not nr['label'];assert sha(ROOT/nr['image'])==nr['image_sha256']
        with Image.open(ROOT/nr['image']) as im:normal_rgb=im.convert('RGB')
        for sr in shapes:
            assert sr['phase']=='fit' and sr['label'];assert sha(ROOT/sr['image'])==sr['image_sha256'] and sha(ROOT/sr['mask'])==sr['mask_sha256']
            with Image.open(ROOT/sr['image']) as im:rgb=im.convert('RGB')
            with Image.open(ROOT/sr['mask']) as im:mask=im.convert('L')
            seed=int(hashlib.sha256(('preview19:'+nr['id']+':'+sr['id']).encode()).hexdigest()[:8],16)
            try:base,augmented,label,metadata=synthesize(normal_rgb,rgb,mask,seed)
            except ValueError as error:skipped.append({'normal_id':nr['id'],'source_shape_id':sr['id'],'reason':str(error)});continue
            ident=f'{len(samples):03d}';paths={}
            for name,im in [('normal',base),('synthetic',augmented),('procedural_mask',label)]:
                path=DATA/(ident+'-'+name+'.png');im.save(path);paths[name]={'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path)}
            row={'id':ident,'category':nr['category'],'normal_FIT_id':nr['id'],'source_real_shape_FIT_id':sr['id'],'paths':paths,'metadata':metadata,'phase':'FIT_preview_only','synthetic_not_natural_ground_truth':True}
            samples.append(row)
            x,y,w,h=metadata['target_rect_xywh'];box=(max(0,x-24),max(0,y-24),min(512,x+w+24),min(512,y+h+24))
            zoom=Image.new('RGB',(200,200),'white');zoom.paste(base.crop(box).resize((100,200)),(0,0));zoom.paste(augmented.crop(box).resize((100,200)),(100,0))
            rendered=[]
            for im in (base,augmented,label.convert('RGB')):
                thumb=im.resize((200,200));draw=ImageDraw.Draw(thumb);draw.rectangle((x*200/512,y*200/512,(x+w)*200/512,(y+h)*200/512),outline='red',width=1);rendered.append(thumb)
            visuals[nr['category']].append((rendered+[zoom],ident))
    contacts=[]
    for c,rows in visuals.items():
        if not rows:continue
        canvas=Image.new('RGB',(800,len(rows)*225),'#eeeeee');draw=ImageDraw.Draw(canvas)
        for i,(images,ident) in enumerate(rows):
            draw.text((2,i*225+2),f'{c} #{ident}: base | synthetic | procedural mask | zoom base/synthetic',fill='black')
            for j,im in enumerate(images):canvas.paste(im,(j*200,i*225+20))
        path=DATA/(c+'-FIT-preview.png');canvas.save(path);contacts.append(path.relative_to(ROOT).as_posix())
    result={'schema':VERSION,'UTC':datetime.now(timezone.utc).isoformat(),'requested':12,'generated':len(samples),'categories':dict(Counter(r['category'] for r in samples)),
        'skipped':skipped,'samples':samples,'contacts':contacts,'only_target_and_shape_FIT_RGB_read':True,'VAL_CAL_TEST_photos_or_masks_used':False,
        'new_encoder_features_or_model_scores':0,'actual_optimizer_updates':0,'new_natural_defect_labels':0,'source_shapes_used':len(shapes),
        'synthetic_support_is_not_natural_defect_GT':True,'input_foreground_is_not_defect_or_part_GT':True,'clinical_or_mechanical_realism_validated':False,
        'not_the_DRAEM_architecture_or_paper_reproduction':True,'not_a_full_training_protocol':True,'agent_visual_review_pending':True}
    new(DATA/'preview.json',result);new(REPORT,result)
    print({k:v for k,v in result.items() if k not in ('samples','skipped')},flush=True)


if __name__=='__main__':main()
