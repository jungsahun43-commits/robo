"""Plot actual full-photo validation history, separating evaluation scopes."""
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.config/matplotlib'))
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scripts.train_facility_target import read,save,sha

EXPERIMENTS=[('facility-presence-target-v2s','Larger photo model',0),
             ('facility-presence-target-detail','Detail augmentation',0),
             ('facility-presence-target-highres','640px photo model',0),
             ('facility-presence-target-codebrim','More real concrete data',1),
             ('facility-presence-target-spatial','Photo + position training',1),
             ('facility-presence-target-s2ds','Extra S2DS position labels',1),
             ('facility-presence-target-hard','Bounded hard TRAIN sampling',1),
             ('facility-presence-target-auxiliary','Original-tag auxiliary task',1),
             ('facility-presence-target-roi-control','Fixed crop control',1),
             ('facility-presence-target-small-region','Small-region context crops',1),
             ('facility-presence-target-building-control','Concrete study control',1),
             ('facility-presence-target-building-convid','ConViD positive photos',1),
             ('facility-presence-target-discrimination-control','Discrimination study control',1),
             ('facility-presence-target-discrimination-ranking','Within-source pos/neg ranking',1),
             ('facility-presence-target-detail-control','Architecture study control',1),
             ('facility-presence-target-detail-s4','Stride4 feature residual',1),
             ('facility-presence-target-context-control','Pool study control',1),
             ('facility-presence-target-context-pool','Narrow/broad pool contrast',1)]


def main():
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),sharey=True)
    records=[];maximum=35
    colors=[plt.get_cmap('tab10'),plt.get_cmap('tab20')]
    color_counts=[0,0]
    for name,label,index in EXPERIMENTS:
        path=ROOT/'runs'/name/'history.json'
        if not path.exists():continue
        history=read(path)
        values=[100*r['worst_target_error'] for r in history]
        if not values:continue
        epochs=[r['epoch'] for r in history]
        state=read(path.with_name('TRAINING.json'))['status']
        color=colors[index](color_counts[index])
        color_counts[index]+=1
        axes[index].plot(epochs,values,marker='.',color=color,label=label+(' (running)' if state!='complete' else ''))
        best=min(range(len(values)),key=lambda i:values[i])
        axes[index].scatter(epochs[best],values[best],facecolors='none',edgecolors='black',s=75,zorder=3)
        maximum=max(maximum,max(values)+3)
        records.append({'run':name,'history_sha256':sha(path),'status':state,'actual_epochs':len(history)})
    for ax,title in zip(axes,('DACL + Dam validation','DACL + Dam + CODEBRIM validation')):
        ax.axhline(5,color='#216e39',linestyle='--',label='Target: strictly below 5%')
        ax.set_title(title,fontsize=11);ax.set_xlabel('Actual training epoch')
        ax.set_ylim(0,maximum);ax.set_xlim(.5,18.5);ax.grid(alpha=.2)
        ax.legend(loc='lower right',fontsize=7,ncol=2,framealpha=1.)
    axes[0].set_ylabel('Largest crack/spalling FNR or FPR (%)')
    fig.suptitle('Facility target: full-photo validation during training',fontsize=13)
    fig.text(.5,.025,'Source validation only. Panels use different domains. Circles mark best epochs; tests are not selection data.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.065,1,.94))
    output=ROOT/'reports/facility-five-percent-curve.png'
    fig.savefig(output,dpi=160);plt.close(fig)
    save(output.with_suffix('.json'),{'plot_sha256':sha(output),'script_sha256':sha(Path(__file__)),
                                     'histories':records,'metric':'Full-photo validation worst FNR/FPR, not aggregate wrong-photo fraction or field error'})
    print(output)


if __name__=='__main__':main()
