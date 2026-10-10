"""Same fixed grid1 evaluator with a scoped new-architecture adapter."""
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from safelog_ai.convnext_facility import ConvnextResearchPresence
from scripts.facility_convnext import NAME,PROTOCOL,validate_protocol,read,require
import scripts.evaluate_facility_target as existing
def main():
    torch.set_num_threads(4);validate_protocol(read(ROOT/PROTOCOL))
    require(not(ROOT/'runs'/NAME/'TARGET-SELECTION.json').exists(),'Preserve completed selection')
    original=existing.PresenceClassifier
    try:
        existing.PresenceClassifier=ConvnextResearchPresence
        existing.select(NAME,'cuda',(1,))
    finally:existing.PresenceClassifier=original
if __name__=='__main__':main()
