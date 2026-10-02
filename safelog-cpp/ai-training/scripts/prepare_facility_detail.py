"""Create TRAIN-only detail crops using publisher polygon/mask ground truth."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_facility_target import read, save


def tile_targets(masks, box, minimum=16):
    x1,y1,x2,y2=box
    result=[]
    for mask in masks:
        pixels=int(mask[y1:y2,x1:x2].sum()) if mask is not None else None
        result.append(-1 if pixels is None or (0 < pixels < minimum) else int(pixels >= minimum))
    return result


def process(task):
    record, classes, output, kind = task
    if kind == "dacl":
        from scripts.prepare_facility_data import DACL_MAPPING
        path = ROOT / "data/dacl10k-yolo/images/train" / f"{record['stem']}.jpg"
        source = ROOT / "data" / record["source"]
        annotation = read(source.parent.parent.parent / "annotations/train" / f"{source.stem}.json")
        with Image.open(path) as handle: image=handle.convert("RGB")
        arrays=[]
        for label in classes:
            mask=Image.new("1",image.size); draw=ImageDraw.Draw(mask)
            for shape in annotation["shapes"]:
                if DACL_MAPPING.get(shape["label"])==label:
                    points=[(float(x)*image.width/annotation['imageWidth'],float(y)*image.height/annotation['imageHeight']) for x,y in shape['points']]
                    draw.polygon(points,fill=1)
            arrays.append(np.asarray(mask,dtype=bool))
        parent=record["stem"]; domain="dacl"
    else:
        if record["target_split"]!="train": raise ValueError("Dam validation photo used as crop parent")
        path=ROOT/record["image"]
        with Image.open(path) as handle: image=handle.convert("RGB")
        original=ROOT/record["source"]
        if "Damage Detection" not in str(original): return []
        level={"E":"Easy","M":"Medium","H":"Hard"}[original.stem[0]]
        mask_path=ROOT/"data/damsegment/source/Damage Segmentaion"/level/"Labels/Mask"/f"{original.stem}_mask.png"
        with Image.open(mask_path) as handle: mask=np.asarray(handle.convert("RGB"))
        arrays=[np.all(mask==(255,0,0),axis=-1),np.all(mask==(0,0,255),axis=-1)]+[None]*5
        parent=path.stem;domain="damsegment"
    grid=3 if kind=="dacl" else 2
    boxes=[(x*image.width//grid,y*image.height//grid,(x+1)*image.width//grid,(y+1)*image.height//grid) for y in range(grid) for x in range(grid)]
    labels=[tile_targets(arrays,box) for box in boxes]
    selected=set()
    for i in (0,1):
        valid=[j for j,t in enumerate(labels) if t[i]==1]
        if valid: selected.add(max(valid,key=lambda j:int(arrays[i][boxes[j][1]:boxes[j][3],boxes[j][0]:boxes[j][2]].sum())))
    normal=[j for j,t in enumerate(labels) if t[0]==0 and t[1]==0]
    if normal: selected.add(normal[0])
    items=[]
    for j in sorted(selected):
        crop=image.crop(boxes[j]);crop.thumbnail((512,512),Image.Resampling.LANCZOS)
        target=output/f"detail-{domain}-{parent}-{j}.jpg";crop.save(target,quality=95)
        items.append({"image":target.relative_to(ROOT).as_posix(),"targets":labels[j],"domain":domain,
                      "parent_id":parent,"parent_split":"train","box_in_parent":list(boxes[j]),
                      "parent_image":path.relative_to(ROOT).as_posix(),"augmentation":"publisher mask/polygon derived crop"})
    return items


def main():
    split=read(ROOT/"runs/facility-presence-target-v2s/SPLIT.json")
    records=read(ROOT/"data/dacl10k-yolo/records.json")
    original=[r for r in records if r["split"]=="train"]
    output=ROOT/"data/facility-detail-training/images";output.mkdir(parents=True,exist_ok=True)
    tasks=[(r,split['classes'],output,"dacl") for r in original]+[(r,split['classes'],output,"dam") for r in split['train']]
    items=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,result in enumerate(pool.map(process,tasks),1):
            items.extend(result)
            if i%1000==0:print(f"Detail parents audited {i}/{len(tasks)}",flush=True)
    if not items:raise ValueError("No valid training crops")
    audit={"split":"train","augmented_crops":len(items),"domains":dict(Counter(i['domain'] for i in items)),
           "parent_photos":len({i['parent_image'] for i in items}),"new_independent_photos":0,
           "per_label":{label:{"positive":sum(i['targets'][j]==1 for i in items),"negative":sum(i['targets'][j]==0 for i in items),"unknown":sum(i['targets'][j]==-1 for i in items)} for j,label in enumerate(split['classes'])},
           "policy":"DACL train parents and new dam TRAIN only; up to two largest target-positive tiles + one negative tile; <=15 mask pixels unknown; original validation/test never cropped into train"}
    save(output.parent/"manifest.json",{"split":"train","classes":split['classes'],"items":items,"audit":audit})
    save(ROOT/"reports/facility-target-detail-data-audit.json",audit)
    print(json.dumps(audit,indent=2),flush=True)


if __name__=="__main__":main()
