"""Audit an explicitly downloaded COCO belt ZIP without guessing normal photos.

Reads only archive contents, does not log in, download, train, or infer. Normal
bboxes establish candidate normal ROIs, never whole-photo mechanical health.
"""
import argparse
import hashlib
import io
import json
import math
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'https://universe.roboflow.com/gofaones-workspace/belt-defects/dataset/2'
EXPECTED = {'belt splice','deep scratch','longitudinal tear','normal belt','objects','slight scratch'}
DAMAGE = {'deep scratch','longitudinal tear','slight scratch'}


def parent_name(name):
    # Roboflow UUID names group augmented siblings; this is not a session ID.
    base = PurePosixPath(name).name
    return re.split(r'\.rf\.[0-9a-f]+',base,flags=re.IGNORECASE)[0]


def checked_bbox(box,width,height):
    if not isinstance(box,list) or len(box)!=4 or not all(math.isfinite(float(v)) for v in box):
        raise ValueError('Nonfinite or malformed bbox')
    x,y,w,h = map(float,box)
    if x<0 or y<0 or w<=0 or h<=0 or x+w>width+1 or y+h>height+1:
        raise ValueError('Bbox outside declared image bounds')
    return [x,y,w,h]


def normal_status(names):
    unknown = set(names)-EXPECTED
    if unknown:
        raise ValueError('Unknown belt labels: '+str(sorted(unknown)))
    if not names:
        return 'unannotated_unknown'
    if 'normal belt' not in names:
        return 'no_explicit_normal_roi'
    if set(names)=={'normal belt'}:
        return 'explicit_normal_roi_candidate'
    return 'mixed_labels_requires_review'


def audit(archive):
    rows, files, seen_rgb, parent_splits = [], {}, defaultdict(set), defaultdict(set)
    with zipfile.ZipFile(archive) as packed:
        members = {}
        for item in packed.infolist():
            name = item.filename.replace('\\','/')
            parts = PurePosixPath(name)
            if parts.is_absolute() or '..' in parts.parts or ':' in name:
                raise ValueError('Unsafe ZIP member')
            if name in members:
                raise ValueError('Duplicate ZIP path')
            members[name] = item
        if sum(i.file_size for i in members.values())>2_000_000_000:
            raise ValueError('Audit limit2GB exceeded; inspect the archive before expanding')
        annotations = sorted(n for n in members if n.endswith('/_annotations.coco.json') or n=='_annotations.coco.json')
        if not annotations:
            raise ValueError('COCO _annotations.coco.json required')
        for label_path in annotations:
            raw = packed.read(members[label_path])
            files[label_path] = hashlib.sha256(raw).hexdigest()
            coco = json.loads(raw)
            categories = {c['id']:c['name'].strip().lower() for c in coco['categories']}
            actual = set(categories.values())
            # Some Roboflow COCO exports include one unused root superclass.
            used_ids = {a['category_id'] for a in coco['annotations']}
            if not used_ids<=set(categories) or set(categories[i] for i in used_ids)-EXPECTED:
                raise ValueError('Unexpected annotated categories')
            by_image = defaultdict(list)
            for annotation in coco['annotations']:
                by_image[annotation['image_id']].append(annotation)
            image_ids = [i['id'] for i in coco['images']]
            if len(image_ids)!=len(set(image_ids)) or not set(by_image)<=set(image_ids):
                raise ValueError('Duplicate or unlinked COCO image IDs')
            directory = str(PurePosixPath(label_path).parent)
            split = PurePosixPath(directory).name if directory!='.' else 'unspecified'
            if split not in ('train','valid','val','test','unspecified'):
                raise ValueError('Unexpected export split')
            for image in coco['images']:
                name = str(PurePosixPath(directory)/image['file_name']) if directory!='.' else image['file_name']
                if name not in members:
                    raise ValueError('Missing source image: '+name)
                payload = packed.read(members[name])
                with Image.open(io.BytesIO(payload)) as source:
                    rgb = np.asarray(source.convert('RGB'))
                    width,height = source.size
                if width!=image['width'] or height!=image['height']:
                    raise ValueError('Image geometry differs from COCO')
                rgb_sha = hashlib.sha256(str(rgb.shape).encode()+rgb.tobytes()).hexdigest()
                seen_rgb[rgb_sha].add(split)
                parent = parent_name(image['file_name'])
                parent_splits[parent].add(split)
                labels = [categories[a['category_id']] for a in by_image[image['id']]]
                boxes = [{'label':categories[a['category_id']],
                          'bbox':checked_bbox(a['bbox'],width,height)} for a in by_image[image['id']]]
                rows.append({'id':name,'split':split,'source_parent_hint':parent,'decoded_rgb_sha256':rgb_sha,
                    'source_bytes_sha256':hashlib.sha256(payload).hexdigest(),'width':width,'height':height,
                    'labels':labels,'normal_roi_status':normal_status(labels),'boxes':boxes,
                    'damage_annotation_present':bool(set(labels)&DAMAGE),
                    'whole_photo_normal_ground_truth':'unknown','mechanical_health_ground_truth':'unknown'})
    return {'status':'audited_not_training_ready','source':SOURCE,'license_declared_on_source':'CC-BY-4.0',
        'source_archive_sha256':hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
        'exported_images':len(rows),'source_parent_hints':len(parent_splits),
        'normal_roi_status_counts':dict(Counter(r['normal_roi_status'] for r in rows)),
        'cross_split_exact_rgb_groups':[k for k,v in seen_rgb.items() if len(v)>1],
        'cross_split_parent_hint_groups':[k for k,v in parent_splits.items() if len(v)>1],
        'annotation_sha256':files,'rows':rows,
        'whole_photo_normal_labels_created':0,'ai_labels_created':0,'source_session_groups_verified':False,
        'source_author_identity_and_original_provenance_verified':False,
        'next_gate':'Review normal ROI coverage, original provenance and sequence/parent grouping before split/train',
        'normal_roi_is_whole_photo_normal':False,'model_inference_performed':False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = audit(args.archive)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as handle:
        json.dump(result,handle,indent=2,ensure_ascii=False)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False,indent=2))
