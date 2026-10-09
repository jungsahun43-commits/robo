"""Plot verified primary-ASL history and source-validation measurements."""
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
    source=ROOT/"reports/facility-primary-asymmetric-study-comparison.json";report=read(source)
    entries=report["experiments"];names=["Original","Low-LR control","Primary ASL"]
    fig,axes=plt.subplots(1,3,figsize=(13.6,4.8),layout="constrained")
    for entry,color,label in zip(entries[1:],("#d69930","#32688e"),("Original focal control","Primary asymmetric loss")):
        history=read(ROOT/"runs"/entry["run"]/"history.json")
        axes[0].plot([h["epoch"]for h in history],[100*h["worst_target_error"]for h in history],marker="o",color=color,label=label)
    axes[0].set_title("Actual six-epoch full-photo VAL history");axes[0].set_ylabel("Largest primary FNR/FPR (%)");axes[0].set_xlabel("Epoch");axes[0].grid(alpha=.2);axes[0].legend(fontsize=8)
    worst=[e["worst_error"]*100 for e in entries]
    bars=axes[1].bar(names,worst,color=["#8a96a8","#d69930","#32688e"])
    axes[1].bar_label(bars,fmt="%.2f%%");axes[1].axhline(5,linestyle="--",color="#6b6b6b",label="Every rate <5% target")
    axes[1].set_title("Largest selected VAL FNR/FPR (12 rates)");axes[1].set_ylabel("Percent");axes[1].set_ylim(0,max(worst)*1.22);axes[1].legend(fontsize=8)
    x=list(range(3));crack=[e["small_dacl_polygon_area_below_one_percent"]["concrete_crack"]["false_negatives"]for e in entries]
    spall=[e["small_dacl_polygon_area_below_one_percent"]["concrete_spalling"]["false_negatives"]for e in entries]
    a=axes[2].bar(x,crack,color="#437fb0",label="Crack: 93 positives");b=axes[2].bar(x,spall,bottom=crack,color="#bc7551",label="Spalling: 105 positives")
    axes[2].bar_label(a,label_type="center");axes[2].bar_label(b,label_type="center");axes[2].set_xticks(x,names)
    axes[2].set_ylabel("False-negative task/photo cases");axes[2].set_title("Small DACL defects: annotated area <1%");axes[2].legend(fontsize=8)
    fig.suptitle("Primary asymmetric-loss study: actual training results",fontsize=13)
    fig.supxlabel("Original labels, draw order, architecture and other losses retained. Repeated public VAL; independent factory accuracy not measured.",fontsize=8)
    output=ROOT/"reports/facility-primary-asymmetric-study-comparison.png";fig.savefig(output,dpi=150);plt.close(fig)
    save(output.with_suffix(".plot.json"),{"plot_sha256":sha(output),"script_sha256":sha(Path(__file__)),"comparison_sha256":sha(source),"actual_new_epochs":report["actual_new_training_epochs"]})
    print(output,flush=True)

if __name__=="__main__":main()
