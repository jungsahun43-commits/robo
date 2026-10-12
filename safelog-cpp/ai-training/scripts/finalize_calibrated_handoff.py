"""Close local documentation links in a new ZIP; retain the HTTP-tested v5 runtime."""
import hashlib
import posixpath
import re
import zipfile
from pathlib import Path
from prepare_foreign_calibration_check import read,new,sha
from study_foreign_calibration import ROOT


def main():
    release = read(ROOT/'reports/calibrated-conveyor-release-v5.json')
    tested = read(ROOT/'reports/calibrated-conveyor-standalone-v5.json')
    original = Path(release['package'])
    assert tested['status']=='passed' and sha(original)==release['package_sha256']==tested['package_sha256']
    with zipfile.ZipFile(original) as archive:
        entries = {name:archive.read(name) for name in archive.namelist() if name!='file-sha256.json'}
    added = []
    for iteration in range(12):
        missing = set()
        for name,content in list(entries.items()):
            if name.endswith('.md'):
                for url in re.findall(r'\]\(([^)]+)\)',content.decode('utf-8')):
                    if url.startswith(('http:','https:','#')):
                        continue
                    target = posixpath.normpath(posixpath.join(posixpath.dirname(name),url))
                    if target not in entries:
                        missing.add(target)
        if not missing:
            break
        for name in sorted(missing):
            assert not name.startswith(('../','/'))
            source = ROOT.parent/name if name.startswith('docs/') else ROOT/name
            assert source.resolve().is_relative_to(ROOT.parents[1].resolve()) and source.is_file(),name
            assert source.stat().st_size<5_000_000 and source.suffix in ('.md','.json','.png','.py'),name
            content = source.read_bytes()
            if source.suffix=='.md':
                text = content.decode('utf-8').replace('../../docs/','../docs/').replace('../ai-training/reports/','../reports/')
                if name=='docs/CONVEYOR_FOREIGN_API_HANDOFF_KO.md':
                    text = '이전 v1 API의 기록이다. 해당 모델은 v3/v4 ZIP에 보존한다. 이번 실행은 FOREIGN_CALIBRATED_HANDOFF_KO.md를 사용한다.\n\n'+text
                content = text.encode('utf-8')
            entries[name]=content;added.append(name)
    else:
        raise RuntimeError('Documentation link closure did not finish')
    for name,digest in release['file_sha256'].items():
        if name.startswith('safelog_ai/') or name.endswith('.pt'):
            assert hashlib.sha256(entries[name]).hexdigest()==digest
    entries['reports/calibrated-conveyor-standalone-v5.json'] = (ROOT/'reports/calibrated-conveyor-standalone-v5.json').read_bytes()
    manifest = {name:hashlib.sha256(content).hexdigest() for name,content in entries.items()}
    import json
    package = original.with_name('equipment-ai-role4-handoff-2026-10-12-v5.1.zip')
    assert not package.exists()
    with zipfile.ZipFile(package,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as archive:
        for name,content in entries.items():
            archive.writestr(name,content)
        archive.writestr('file-sha256.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(package) as archive:
        assert archive.testzip() is None
        assert all(hashlib.sha256(archive.read(name)).hexdigest()==digest for name,digest in manifest.items())
    new(ROOT/'reports/calibrated-conveyor-release-v5_1.json',{'status':'created_and_verified','package':str(package.resolve()),
        'package_sha256':sha(package),'bytes':package.stat().st_size,'parent_v5_sha256_preserved':sha(original),
        'models':release['models'],'file_sha256':manifest,'additional_documentation_files':added,
        'local_markdown_links_resolved':True,'runtime_and_model_bytes_identical_to_actual_http_tested_v5':True,
        'actual_http_test_inherited_from':'calibrated-conveyor-standalone-v5.json','source_sha256':sha(Path(__file__))})
    print({'package':str(package),'bytes':package.stat().st_size,'sha256':sha(package),'added_docs':len(added)},flush=True)


if __name__=='__main__':
    main()
