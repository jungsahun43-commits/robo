"""Plot actual reviewed-selection outcome and verified validation comparison."""
from pathlib import Path
import os
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".config/matplotlib"))
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from scripts.train_facility_target import read,save,sha

def main():
    source=ROOT/"reports/facility-ai-agreement-study-comparison.json";report=read(source)
    entries=report["experiments"];names=["Original","Low-LR control","AI agreement"]
    fig,axes=plt.subplots(1,3,figsize=(13.6,4.8),layout="constrained")
    audit=report["visual_review_audit"]
    counts=[audit["both_definite_agreement_photo"],audit["definite_disagreement_photo"],audit["uncertain_photo"]]
    bars=axes[0].bar(["Agreed","Disagreed","Uncertain"],counts,color=["#237a57","#bf673c","#88909c"])
    axes[0].bar_label(bars);axes[0].set_title("200 TRAIN photos visually reviewed by AI");axes[0].set_ylabel("Photos");axes[0].set_ylim(0,max(counts)*1.16)
    worst=[e["worst_error"]*100 for e in entries]
    bars=axes[1].bar(names,worst,color=["#8a96a8","#e0a240","#237a57"])
    axes[1].bar_label(bars,fmt="%.2f%%");axes[1].axhline(5,linestyle="--",color="#6b6b6b",label="Every rate <5% target")
    axes[1].set_title("Largest primary VAL FNR/FPR (12 rates)");axes[1].set_ylabel("Percent");axes[1].set_ylim(0,max(worst)*1.22);axes[1].legend(fontsize=8)
    x=list(range(3));crack=[e["small_dacl_polygon_area_below_one_percent"]["concrete_crack"]["false_negatives"]for e in entries]
    spall=[e["small_dacl_polygon_area_below_one_percent"]["concrete_spalling"]["false_negatives"]for e in entries]
    a=axes[2].bar(x,crack,color="#437fb0",label="Crack: 93 positives")
    b=axes[2].bar(x,spall,bottom=crack,color="#bc7551",label="Spalling: 105 positives")
    axes[2].bar_label(a,label_type="center");axes[2].bar_label(b,label_type="center")
    axes[2].set_xticks(x,names);axes[2].set_ylabel("False-negative task/photo cases");axes[2].set_title("Small DACL defects: annotated area <1%");axes[2].legend(fontsize=8)
    fig.suptitle("AI visual selection + bounded exposure: actual training results",fontsize=13)
    fig.supxlabel("AI judgements are unconfirmed. Original labels retained. Repeated public VAL; independent factory accuracy not measured.",fontsize=9)
    output=ROOT/"reports/facility-ai-agreement-study-comparison.png";fig.savefig(output,dpi=150);plt.close(fig)
    save(output.with_suffix(".plot.json"),{"plot_sha256":sha(output),"script_sha256":sha(Path(__file__)),"comparison_sha256":sha(source),"actual_new_epochs":report["actual_new_training_epochs"]})
    print(output,flush=True)

if __name__=="__main__":main()
