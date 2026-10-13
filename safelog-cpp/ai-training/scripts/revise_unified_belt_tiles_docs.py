"""Docs-only v1.1: explicit reused weights, original resolution and fused-score scope."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.unified_main_data import read, sha
from scripts.release_unified_main import local_links
from scripts.study_unified_belt_tiles import BASE
REPO = ROOT.parents[1]; OUTPUTS = REPO.parents[1]/'outputs'; REPORTS = ROOT/'reports'
OLD_NAME = 'unified-main-belt-tiles-role4-handoff-2026-10-13-v1'
NAME = OLD_NAME+'.1'; STAGE = BASE/'handoff-stage-v1.1'; ARCHIVE = OUTPUTS/(NAME+'.zip')


def new(path, value):
    path.parent.mkdir(exist_ok=True, parents=True)
    with path.open('x', encoding='utf-8', newline='\n') as out: out.write(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    original = read(REPORTS/'unified-belt-tiles-release.json'); choice = read(REPORTS/'unified-belt-tiles-selection.json')
    assert original['status'] == 'passed' and choice['selected'] == 'accepted_tiles' and choice['selected_new_neural_updates'] == 0
    assert sha(Path(original['archive'])) == original['archive_sha256'] and not STAGE.exists() and not ARCHIVE.exists()
    scope = {'status': 'descriptive_metadata_clarified_no_metric_change', 'all_original_counts_scores_and_thresholds_preserved': True,
        'common_summary_photo_score_definition': 'Inherited single-view shorthand; use explicit per-config policy/formula below',
        'accepted_tiles_and_trained_tiles_actual_photo_score': 'Maximum over original640 valid pixels of0.5global sigmoid probability +0.5native Hann-mean tile sigmoid probability, mapped bilinearly; not calibrated',
        'trained_global_actual_photo_score': 'Maximum global sigmoid probability over original640 valid pixels; not calibrated',
        'selected_checkpoint_additional_uniform_training_updates': 0, 'performed_research_candidate_updates_not_selected': 1280,
        'selected_policy_sha256': choice['inference_policy']['sha256'], 'selected_model_sha256': choice['checkpoint_sha256']}
    new(REPORTS/'unified-belt-tiles-score-scope.json', scope)
    guide = REPO/'safelog-cpp/docs/UNIFIED_BELT_TILES_HANDOFF_KO.md'
    text = guide.read_text(encoding='utf-8').replace('# 역할4 · 벨트 출력 추가 학습 통합 모델', '# 역할4 · 기존 가중치와 새 원본 구역 분석')
    text = text.replace(OLD_NAME+'.zip', NAME+'.zip')
    paragraph = '''
이번 전달의 가중치는 직전 native_mix와 동일하다. 추가10회 학습 후보는 미탐 유지 조건을 통과하지 못해 적용하지 않았고, 기존 가중치에 새 구역 분석을 연결했다. 구역 입력도 같은 모델 하나로 처리한다.

## 사진 입력과 분석 방식 확인

- 카메라 원본 JPEG/PNG를 보낸다. 서버 제한은12MP이하·8MB이하다. 보내기 전에640으로 줄이면 원본 세부 정보가 사라지고 작은 사진은 전체 분석으로 fallback한다.
- 기존 C++ 어댑터는 파일 바이트를 그대로 읽어 전송한다. 촬영·저장 화면에서도 원본 파일을 유지한다.
- GET /v1/inference-policy의beltMode는global_tile_mean이고 정책SHA는`'''+choice['inference_policy']['sha256']+'''`다. 가중치SHA와 정책SHA를 함께 저장한다.
- 같은 모델이 여러 구역을 처리해서 전체 사진만 분석할 때보다 계산량이 늘어난다. 정상 사진 오탐·현장 일반화는 아직 미검증이다.
- 구역 모드의 사진 점수는 전체sigmoid50%+구역sigmoid평균50%로 합친 pixel 확률의최댓값이다. 공통 JSON요약의single-view문구보다[정확한점수범위](../ai-training/reports/unified-belt-tiles-score-scope.json)를사용한다. 실제점수·계수·검출기준은바꾸지않았다.
'''
    first, rest = text.split('\n\n', 1); guide.write_bytes((first+'\n\n'+paragraph.strip()+'\n\n'+rest).encode('utf-8'))
    result_doc = REPORTS/'UNIFIED_BELT_TILES_RESULTS_KO.md'
    with result_doc.open('a', encoding='utf-8', newline='\n') as out:
        out.write('\n## 선택 버전과 점수 설명\n\n최종 연결 버전은 기존 native_mix 가중치에 새 구역 분석을 적용한 accepted_tiles다. 이번에 실제 학습한1,280회 후보 가중치는 적용하지 않았다. 원본 사진을 서버에 보내며, 구역 분석은 전체 분석보다 계산량이 늘어난다.\n\n구역 모드의 사진 점수는 원래640유효영역에서 최종융합확률의최댓값이다. 공통요약의single-view sigmoid문구는[설정별실제점수설명](unified-belt-tiles-score-scope.json)으로보완한다. 수치·기준·선택·가중치는변경하지않았다.\n')
    shutil.copytree(BASE/'handoff-stage', STAGE, ignore=shutil.ignore_patterns('package-sha256.json'))
    for source in [guide, result_doc, REPORTS/'unified-belt-tiles-score-scope.json']:
        target = STAGE/source.relative_to(REPO); target.parent.mkdir(exist_ok=True, parents=True); shutil.copy2(source, target)
    runtime = [p for p in (BASE/'handoff-stage').rglob('*') if p.is_file() and (p.suffix in ('.pt', '.py', '.cpp', '.hpp') or p.name in ('model.json', 'requirements-api.txt'))]
    assert all(sha(p) == sha(STAGE/p.relative_to(BASE/'handoff-stage')) for p in runtime)
    for doc in STAGE.rglob('*.md'):
        for link in local_links(doc): assert link.is_relative_to(STAGE) and link.is_file(), link
    new(STAGE/'package-sha256.json', {p.relative_to(STAGE).as_posix(): sha(p) for p in STAGE.rglob('*') if p.is_file()})
    with zipfile.ZipFile(ARCHIVE, 'x', zipfile.ZIP_DEFLATED, compresslevel=5) as archive:
        for p in sorted(STAGE.rglob('*')):
            if p.is_file(): archive.write(p, p.relative_to(STAGE).as_posix())
    with zipfile.ZipFile(ARCHIVE) as archive:
        assert archive.testzip() is None and sum(n.endswith('.pt') for n in archive.namelist()) == 1
        members = json.loads(archive.read('package-sha256.json')); assert all(hashlib.sha256(archive.read(n)).hexdigest() == h for n, h in members.items())
        for name in original['hazard_responses']:
            assert name['inferencePolicySha256'] == choice['inference_policy']['sha256']
    new(REPORTS/'unified-belt-tiles-release-v1.1.json', {'status': 'passed_docs_only_revision', 'archive': str(ARCHIVE), 'archive_sha256': sha(ARCHIVE),
        'archive_bytes': ARCHIVE.stat().st_size, 'model_sha256': choice['checkpoint_sha256'], 'mode': choice['mode'],
        'inference_policy_sha256': choice['inference_policy']['sha256'], 'runtime_model_manifest_API_and_CPP_byte_identical_to_tested_v1': True,
        'runtime_files_verified': len(runtime), 'isolated_v1_CPU_CPP_operations_inherited': 9, 'new_neural_updates': 0,
        'zip_crc_member_SHA_local_links_passed': True, 'original_v1_zip_sha256_preserved': original['archive_sha256'],
        'fresh_v1_1_API_calls_repeated': False, 'only_docs_and_descriptive_score_scope_changed': True, 'app_default_promoted': False})
    print(json.dumps({'revision': 'v1.1 docs only', 'archive': str(ARCHIVE), 'sha256': sha(ARCHIVE), 'runtime_files_identical': len(runtime)}), flush=True)


if __name__ == '__main__': main()
