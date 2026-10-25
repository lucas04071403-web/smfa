"""Export aggregate development trade-off curves, with measured knots visible."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from wp1_supplement.common import OUT, summary, frozen
from wp1_supplement.analyze import OLD_C1, c1_config, points


def make_plot():
    protocol,_=frozen()
    selected=json.loads((OUT/'c1_selection.json').read_text())
    final=(OUT/'B1_operator_selection.json').exists()
    fig,axes=plt.subplots(1,2 if final else 1,figsize=(13 if final else 7,5.1),squeeze=False)
    ax=axes[0,0]
    old=points([c1_config(n) for n in OLD_C1]);new=selected['c1_curve']
    ax.plot([p['forget_combined'] for p in old],[p['retain_image'] for p in old],
            '--',color='#89939c',lw=1.6,label='Original four-point C1')
    ax.plot([p['forget_combined'] for p in new],[p['retain_image'] for p in new],
            '-o',color='#226c9a',ms=4,lw=1.8,label='Enriched C1 (measured knots)')
    for p in new:
        if .75<=p['parameter']<=1:
            ax.annotate(f"c={p['parameter']:g}",(p['forget_combined'],p['retain_image']),
                        xytext=(4,-13),textcoords='offset points',fontsize=8)
    c0=[summary(n) for n in ('C0_k10','SMFA','C0_k1')]
    ax.plot([p['forget_combined'] for p in c0],[p['retain_image'] for p in c0],
            ':s',color='#6c7460',ms=4,label='C0 mask (k=10,5,1)')
    smfa=summary('SMFA');ax.scatter(smfa['forget_combined'],smfa['retain_image'],s=80,marker='*',color='#2e6b31',zorder=6)
    ax.annotate('SMFA',(smfa['forget_combined'],smfa['retain_image']),xytext=(-40,9),textcoords='offset points',fontsize=9)
    for lam,col in zip((10,100,1000),('#ac432e','#b76718','#8e53a2')):
        p=summary(f'C2_l{lam}');ax.scatter(p['forget_combined'],p['retain_image'],s=48,marker='D',color=col,zorder=5,label=f'C2 lambda={lam}, alpha=1')
    ax.set_title(f'C1 enrichment; selected lambda = {selected["selected_lambda"]}')
    if final:
        ax=axes[0,1]
        for family,col,label in [('C2','#ac432e','Soft'),('C3','#8155aa','Hard')]:
            match=json.loads((OUT/'matching'/f'{family}.json').read_text())
            curve=points(match['configs'])
            ax.plot([p['forget_combined'] for p in curve],[p['retain_image'] for p in curve],
                    '-o',color=col,ms=4,lw=1.3,label=f'{label} measured curve')
            if match['state']=='matched':
                p=match['summary'];ax.scatter(p['forget_combined'],p['retain_image'],s=110,marker='*',color=col,zorder=7)
        for family,col in [('C0','#6c7460'),('C1','#226c9a')]:
            match=json.loads((OUT/'matching'/f'{family}.json').read_text())
            if match['state']=='matched':
                p=match['summary'];ax.scatter(p['forget_combined'],p['retain_image'],s=55,marker='s',color=col,label=f'{family} actual target check',zorder=7)
        ax.set_title('B1: actual strength validation')
    for ax in axes.flat:
        ax.axvspan(*protocol['hit_range'],color='#8ab391',alpha=.16,label='Final hit interval')
        ax.axvline(protocol['target'],color='#42664c',ls='--',lw=.9)
        ax.set_xlabel('Forget combined-memory ROUGE-L (lower = more forgetting)')
        ax.set_ylabel('Retain image-memory ROUGE-L (higher = better)')
        ax.grid(alpha=.18);ax.legend(fontsize=7.3,loc='lower right')
    fig.suptitle('Development only; connecting lines are interpolation, not observed outcomes',fontsize=10)
    fig.tight_layout()
    destination=OUT/'figures';destination.mkdir(exist_ok=True)
    name='supplement_tradeoffs' if final else 'c1_enrichment'
    for ext in ('png','svg'):fig.savefig(destination/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    return destination/f'{name}.png'


if __name__=='__main__':print(make_plot())
