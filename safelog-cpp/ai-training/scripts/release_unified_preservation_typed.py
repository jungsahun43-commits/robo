"""Public release entry point for the valid typed-v3 comparison only."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import release_unified_preservation as release

release.NAME=release.NAME.removesuffix('-v2')+'-v3'
release.ARCHIVE=release.OUTPUTS/(release.NAME+'.zip')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['report','package','smoke'])
    getattr(release,parser.parse_args().mode)()
