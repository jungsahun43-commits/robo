"""Fixed first-three FIT contact sheet, display-only coordinates and context."""
from io import BytesIO
import hashlib,json,zipfile
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/industrial-corrosion-subpixel-data-v78'
OUT=ROOT/'data/industrial-corrosion-subpixel-v78'
manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'));assert manifest['audit']['ready']
rows=sorted((r for r in manifest['rows'] if r['phase']=='fit'),key=lambda r:r['original']['member'])[:3]
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
canvas=Image.new('RGB',(1080,630),'white');draw=ImageDraw.Draw(canvas);spec=[]
for i,r in enumerate(rows):
    with zipfile.ZipFile(ROOT/r['archive']) as archive:
        raw=archive.read(r['original']['member']);assert hashlib.sha256(raw).hexdigest()==r['original']['encoded_SHA256']
        rgb=Image.open(BytesIO(raw)).convert('RGB')
    assert hashlib.sha256(rgb.tobytes()).hexdigest()==r['rgb_sha256']
    left=i*360;draw.text((left+10,8),r['id']+' FIT only',fill='black',font=font)
    marked=rgb.copy();pen=ImageDraw.Draw(marked)
    for key,color in (('positive_ROIs','red'),('negative_ROIs','lime')):
        for b in r[key]:pen.rectangle((b['left'],b['top'],b['right']-1,b['bottom']-1),outline=color,width=3)
    marked.thumbnail((340,260));canvas.paste(marked,(left+10,35));chosen={}
    for j,(key,label,color) in enumerate((('positive_ROIs','provided corrosion','red'),('negative_ROIs','provided background','green'))):
        b=sorted(r[key],key=lambda b:b['member'])[0];cx=(b['left']+b['right'])//2;cy=(b['top']+b['bottom'])//2
        crop=rgb.crop((cx-32,cy-32,cx+32,cy+32));p=ImageDraw.Draw(crop);p.rectangle((b['left']-cx+32,b['top']-cy+32,b['right']-cx+31,b['bottom']-cy+31),outline=color,width=1)
        y=315+j*150;canvas.paste(crop.resize((128,128),Image.Resampling.NEAREST),(left+10,y));draw.text((left+150,y+10),label,fill=color,font=font);draw.text((left+150,y+32),'native context64',fill='black',font=font)
        chosen[label]=b['member']
    spec.append({'id':r['id'],'first_sorted_ROI_per_class':chosen,'raw_RGB_redecode':1})
draw.text((10,615),'Supplied ROI markers/context for FIT QA only; not training input or modified ground truth.',fill='black',font=font)
path=RUN/'first3FIT-contact.png';assert not path.exists();canvas.save(path)
record={'fixed_first3_FIT':spec,'policy':'existing v78 first3FIT QA, firstsortedROI perclass/context64/displayonly','new_original_RGB_redecodes':3,'new_natural_photos':0,'VAL_CAL_TEST_content_views':0,'new_features_scores_optimizer':0,'contact_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
(ROOT/'reports/industrial-corrosion-subpixel-v78-FIT-QA.json').write_text(json.dumps(record,indent=2),encoding='utf-8');print(path)
