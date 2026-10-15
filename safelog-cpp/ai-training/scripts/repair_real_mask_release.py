"""Repair only runtime dispatch; preserve initial failed ZIP, weights and cuts."""
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import release_real_mask_transfer as old
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new


def main():
    run=old.RUN;source=old.STAGE;stage=run/'handoff-stage-r1';assert not stage.exists()
    shutil.copytree(source,stage)
    shutil.copyfile(ROOT/'safelog_ai/real_mask_server.py',stage/'safelog_ai/real_mask_server.py')
    assert sha(source/'model.pt')==sha(stage/'model.pt') and sha(source/'evaluation.json')==sha(stage/'evaluation.json')
    old_hashes=json.loads((source/'package-sha256.json').read_text(encoding='utf-8'))
    hashes={p.relative_to(stage).as_posix():sha(p) for p in stage.rglob('*') if p.is_file() and p.name!='package-sha256.json'}
    changes=[n for n,h in hashes.items() if old_hashes.get(n)!=h]
    assert changes==['safelog_ai/real_mask_server.py'],changes
    (stage/'package-sha256.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    archive=old.OUTPUTS/'unified-real-mask-transfer-role4-handoff-2026-10-15-v16-r1.zip';assert not archive.exists()
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for p in sorted(stage.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(stage).as_posix())
    record=old.study.read(run/'package.json');record.update(archive=str(archive),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,
        runtime_revision='r1',weights_thresholds_evaluation_preserved=True,initial_failed_archive_preserved=str(old.ARCHIVE),changed_runtime_files=changes)
    new(run/'package-r1.json',record)
    smoke=inspect.getsource(old.smoke)
    header=Path(inspect.getfile(old)).read_text(encoding='utf-8').split('def checked():',1)[0]
    header=header.replace("STAGE=RUN/'handoff-stage';ISOLATED=RUN/'isolated-zip'","STAGE=RUN/'handoff-stage-r1';ISOLATED=RUN/'isolated-zip-r1'")
    header=header.replace('2026-10-15-v16.zip','2026-10-15-v16-r1.zip')
    checked=inspect.getsource(old.checked).replace("RUN/'package.json'","RUN/'package-r1.json'")
    footer=Path(inspect.getfile(old)).read_text(encoding='utf-8').split("if __name__=='__main__':",1)[1]
    footer=footer.replace("'release.lock'","'release-r1.lock'").replace("'release-failure-'","'release-r1-failure-'")
    target=ROOT/'scripts/release_real_mask_transfer_r1.py'
    with target.open('x',encoding='utf-8',newline='\n') as f:f.write(header+checked+'\n'+smoke+"\nif __name__=='__main__':"+footer)
    print(record,flush=True)


if __name__=='__main__':main()
