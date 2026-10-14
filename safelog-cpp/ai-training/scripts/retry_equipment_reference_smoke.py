"""Same immutable ZIP, corrected optional legacy heatmap checker, fresh namespace."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import release_equipment_reference as release

if __name__ == '__main__':
    release.ISOLATED = release.RUN/'isolated-zip-v2'
    release.smoke()
