"""Publisher examples from FIT only; not independent human acceptance labels."""
from collections import defaultdict
from pathlib import Path
import math
import sys
from PIL import Image, ImageDraw
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.prepare_gearbox_stratified import MANIFEST, RUN
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read


def main():
    groups = defaultdict(list)
    for row in read(MANIFEST)['rows']:
        if row['phase']=='fit': groups[(row['functional_component_category'],row['label'])].append(row)
    selected = [min(v,key=lambda r:r['idx']) for _,v in sorted(groups.items())]
    canvas = Image.new('RGB',(4*256,math.ceil(len(selected)/4)*290),'#eeeeee'); draw = ImageDraw.Draw(canvas)
    for i,row in enumerate(selected):
        assert sha(ROOT/row['image'])==row['image_sha256']
        with Image.open(ROOT/row['image']) as image: rgb=image.convert('RGB').resize((256,256),Image.Resampling.NEAREST)
        x,y=(i%4)*256,(i//4)*290; canvas.paste(rgb,(x,y)); draw.text((x+3,y+259),f"FIT idx{row['idx']} {row['label_name']}",fill='black')
        draw.text((x+3,y+273),row['functional_component_category'],fill='black')
    path=RUN/'FIT-examples.png'
    assert not path.exists();canvas.save(path)
    new(RUN/'FIT-example-reservation.json',{'manifest_sha256':sha(MANIFEST),'ids':[r['id'] for r in selected],
        'all_FIT':True,'image_sha256':sha(path),'scope':'AI visual technical QA, not new human ground truth or all source audit'})
    print({'path':str(path),'FIT_examples':len(selected)},flush=True)


if __name__=='__main__': main()
