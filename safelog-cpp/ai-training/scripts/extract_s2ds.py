"""Extract only the audited S2DS snapshot into a fresh confined directory."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.train_facility_target import read,save,sha
from scripts.extract_codebrim import extract


if __name__=='__main__':
    archive=ROOT/'data/facility-archives/s2ds.zip';pinned=read(ROOT/'datasets/s2ds_source.json')
    if archive.stat().st_size!=pinned['bytes'] or sha(archive)!=pinned['sha256']:
        raise ValueError('S2DS archive differs from audited snapshot')
    destination=ROOT/'data/s2ds/source'
    if destination.exists():raise ValueError('Keep existing extraction; this script expects a fresh destination')
    result=extract(archive,destination)
    result.update(archive_sha256=pinned['sha256'],source=pinned['source'])
    save(ROOT/'reports/facility-target-s2ds-extraction.json',result)
    print('S2DS snapshot extracted, all member CRC/size/path checks retained')
