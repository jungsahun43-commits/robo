"""Audit task meaning and video-name overlap before any new foreign-object fit."""
import json
import re
import zipfile
from collections import Counter,defaultdict
from pathlib import Path

from fetch_conveyor_foreign import DATA, ROOT, EXPECTED_SHA, sha


def main():
    source = json.loads((DATA/'archive-index.json').read_text(encoding='utf-8'))
    assert source['publisher_sha256_verified'] and source['archive_sha256']==EXPECTED_SHA
    rows = [r for r in source['members'] if r['name'].startswith('split-ds-normal-filtered/')]
    groups,extensions = defaultdict(lambda:Counter()),Counter()
    parsed = []
    for row in rows:
        path = row['name'].split('/')
        split = path[1]
        extensions[(split,Path(path[-1]).suffix.lower())]+=1
        match = re.fullmatch(r'(\d+)-(.*?)(?:_detected_\d+)?\.(?:jpeg|jpg|png)',path[-1],re.I)
        assert match,'Audit unexpected curated filename: '+row['name']
        frame,group = match.groups()
        groups[group][split]+=1
        parsed.append({'name':row['name'],'source_split':split,'frame_number_hint':int(frame),
            'recording_name_hint':group,'foreign_object_photo_label':int(split=='anomalous'),
            'label_source':'Publisher normal-op/anomaly folder; no new AI or human labels',
            'belt_crack_photo_label':'unknown','mechanical_health_label':'unknown',
            'crc32':row['crc32'],'source_bytes':row['bytes']})
    assert len(parsed)==18933 and len(groups)==19
    overlap = sorted(g for g,splits in groups.items() if splits.get('train') and splits.get('test'))
    assert len(overlap)==19
    # The YOLO export is belt-region localization, not defect-object GT.
    with zipfile.ZipFile(DATA/'image-data.zip') as packed:
        yaml_text = packed.read('yolo-dataset-center/data.yaml').decode('utf-8')
    assert "names: ['conveyor']" in yaml_text and 'license: Private' in yaml_text
    duplicate_candidates = defaultdict(list)
    for row in parsed:
        duplicate_candidates[row['crc32'],row['source_bytes']].append(row['name'])
    repeated = [names for names in duplicate_candidates.values() if len(names)>1]
    result = {'status':'source_acquired_task_and_overlap_audited_not_trained',
        'source':source['source'],'license_declared_on_dataset_page':'CC-BY-4.0',
        'archive_sha256':EXPECTED_SHA,'publisher_whole_archive_sha256_verified':True,
        'archive_bytes':source['archive_bytes'],'all_archive_members':len(source['members']),
        'curated_photos':len(parsed),'normal_train':12951,'normal_publisher_test':2991,
        'foreign_object_anomalous':2991,'recording_name_hints':19,
        'cross_source_train_test_recording_name_overlap':overlap,
        'physical_recording_equivalence_verified':False,'raw_video_to_processed_photo_equivalence_verified':False,
        'crc_and_size_duplicate_candidates':repeated,'decoded_rgb_and_near_frame_duplicates_verified':False,
        'yolo_classes':['conveyor'],'yolo_export_license_field':'Private','yolo_subset_used':False,
        'normal_means_no_foreign_objects_not_healthy_belt':True,'belt_crack_labels_created':0,
        'new_ai_or_human_labels':0,'model_training_epochs':0,'model_inference_performed':False,
        'source_preview_seen':'split-ds-normal-filtered/train/normal-op/10745-boardcamera_iron_ore_jaspilito_detected_1.jpg',
        'source_test_image_or_annotation_payloads_decoded':0,
        'source_test_normal_folder_label_metadata_counted':2991,
        'source_test_model_fit_calibration_or_scoring_performed':False,
        'next_required_work':'Resolve source video aliases, group splits without sourceTEST fitting, decoded duplicates, and image quality before a separate foreign-object experiment',
        'extensions':{'/'.join(key):count for key,count in extensions.items()},'per_recording_name_counts':dict(groups),
        'acquisition_source_sha256':{name:sha(ROOT/name) for name in
            ('scripts/fetch_conveyor_foreign.py','scripts/fetch_conveyor_foreign_parallel.py','scripts/audit_conveyor_foreign.py')}}
    target = ROOT/'reports/conveyor-foreign-acquisition.json'
    with target.open('x',encoding='utf-8') as handle:
        json.dump(result,handle,indent=2)
    with (DATA/'curated-label-inventory.json').open('x',encoding='utf-8') as handle:
        json.dump({'rows':parsed,'meaning':'Foreign-object photo presence only; belt mechanical health unknown'},handle,indent=2)
    summary = f'''# 새 컨베이어 이물질 원본 확보 · 2026-10-12

[작성자 공개 Mendeley V1](https://data.mendeley.com/datasets/s25x2bnshz/1), DOI10.17632/s25x2bnshz.1, 페이지 라이선스 CC BY4.0. [작성자·라이선스 기록](CONVEYOR_DATA_ATTRIBUTION_KO.md)을 함께 유지한다.

## 실제 확보

공개 원본 ZIP **6,228,464,374바이트(6.23GB)**를 내려받고 작성자가 공개한 전체 SHA256 **{EXPECTED_SHA}**과 일치함을 확인했다. 부분 전송으로 받은 바이트를 보존하고 남은 구간을 최대4개씩 받았다. 압축 전체69,636개 파일은 여러 복사본·동영상·사진·보조 파일을 포함하므로 독립 사진69,636장이라고 세지 않는다.

이번 점검은 작성자가 선별한 `split-ds-normal-filtered`만 대상으로 했다. 정상 TRAIN **12,951장**, 정상 publisherTEST **2,991장**, 이물질 이상 **2,991장**, 총 **18,933장**의 사진 파일을 확인했다. 이 자료의 정상은 **광석만 흐르는 장면**이며 벨트 균열 없음·모터/롤러 정상·안전 가동의 정답은 아니다. 기존 벨트 균열 모델의 정상 사진이나 오탐 분모로 사용하지 않았다.

## 학습 전에 해결할 항목

1. 이름에서 추정한 촬영명19개가 작성자 정상 TRAIN과 TEST에서 **모두 겹친다**. 같은 촬영 구간의 이웃 프레임이 섞인 채 성능을 높게 측정하지 않도록, 촬영명/원본 동영상의 관계를 확인하고 구간 단위로 분리해야 한다. 실제 물리적 촬영·기계의 독립성까지 확인된 것은 아니다.
2. `yolo-dataset-center/data.yaml`의 유일한 클래스는 `conveyor`다. 이 주석을 이물질·벨트 손상의 bbox 정답으로 사용하지 않는다. YAML의 `license: Private` 표시도 기록하고 해당 YOLO 하위자료는 사용하지 않았다. 공개 페이지의 전체 CC BY4.0과 하위 export 표기가 동일하다고 가정하지 않는다.
3. 영상 이름에서 추정한 그룹, 프레임 번호, 정상/이상 의미를 고정한 뒤 decoded RGB·근접 프레임 중복과 흐림·조명을 조사한다. 전체 원본 동영상과 처리 사진의 완전 동등성은 검증하지 않았다.
4. 새 클래스는 `conveyor_foreign_object`로 분리한다. sourceTEST를 학습·보정에 넣지 않는다. 이번에 본 샘플은 작성자 TRAIN 한 장이며 모델 점수는 계산하지 않았다.

**이 새 데이터의 모델 학습은 아직0epoch다.** 현재 실제 완료된 학습은 별도 [벨트 원본 확대40epoch](BELTCRACK_EXPANSION_RESULTS_KO.md)다. 새 정상/이물질 사진을 확보했다고 기존 벨트 균열 오탐률이 검증된 것은 아니다.

[체크섬·파일 수·촬영명 분리 점검](conveyor-foreign-acquisition.json). 원본은 `data/equipment-conveyor-foreign/image-data.zip`, 사진별 작업용 목록은 같은 폴더의 `curated-label-inventory.json`에 있다.
'''
    with (ROOT/'reports/CONVEYOR_FOREIGN_ACQUISITION_KO.md').open('x',encoding='utf-8') as handle:
        handle.write(summary)
    print(json.dumps({k:v for k,v in result.items() if k not in ('per_recording_name_counts','crc_and_size_duplicate_candidates')},indent=2))


if __name__=='__main__':
    main()
