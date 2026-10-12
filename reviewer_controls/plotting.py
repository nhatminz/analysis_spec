"""Reproducible ACL-column figures from saved JSON, without model imports."""
from pathlib import Path
import json
import numpy as np


def pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'savefig.bbox':'tight'})
    return plt


def curve(ax, x, stats, label, color, marker='o', linestyle='-'):
    values = np.array([s['estimate'] if s['estimate'] is not None else np.nan for s in stats])
    lower = np.array([s['ci95'][0] if s['ci95'][0] is not None else np.nan for s in stats])
    upper = np.array([s['ci95'][1] if s['ci95'][1] is not None else np.nan for s in stats])
    ax.plot(x,values,marker=marker,markersize=3,label=label,color=color,linestyle=linestyle,linewidth=1.)
    good = np.isfinite(values)&np.isfinite(lower)&np.isfinite(upper)
    # Percentile intervals need not contain the plug-in estimate.
    ax.vlines(np.asarray(x)[good],lower[good],upper[good],color=color,linewidth=1.)


def finish(fig, ax, destination, name, png=False):
    axes = np.ravel(ax) if isinstance(ax,np.ndarray) else [ax]
    for a in axes:
        a.grid(axis='y',alpha=.2);a.set_ylim(-.02,1.02)
        if not any(np.isfinite(np.asarray(line.get_ydata(),dtype=float)).any() for line in a.lines):
            a.text(.5,.5,'Insufficient support',transform=a.transAxes,ha='center',fontsize=8)
    fig.tight_layout(pad=.5)
    fig.savefig(Path(destination)/(name+'.pdf'))
    if png:fig.savefig(Path(destination)/(name+'.png'),dpi=250)
    import matplotlib.pyplot as plt
    plt.close(fig)


def plot_reviewer(destination, summary=None):
    destination = Path(destination)
    if summary is None:summary = json.loads((destination/'summary.json').read_text())
    plt = pyplot()
    for step,result in summary.items():
        fig,ax = plt.subplots(figsize=(3.35,2.45))
        rows = [r for r in result['E2'] if r['distance']!='all'];x = np.arange(len(rows))
        for offset,(cell,label,color) in enumerate((('current_response','Current response','#0072B2'),
            ('independent_only','Independent only','#D55E00'),('no_prior','No prior miss','#777777'))):
            curve(ax,x+(offset-1)*.07,[r['risks'][cell] for r in rows],label,color)
        ax.set_xticks(x,[r['distance'] for r in rows]);ax.set_xlabel('Prior-miss round distance')
        ax.set_ylabel('Matched miss probability');ax.legend(frameon=False,fontsize=7)
        finish(fig,ax,destination,f'E2_step_{step}')
        fig,ax = plt.subplots(figsize=(3.35,2.5));rows=result['E3'];x=np.arange(len(rows))
        curve(ax,x-.04,[r['risks']['exposed'] for r in rows],'Prior miss','#0072B2')
        curve(ax,x+.04,[r['risks']['unexposed'] for r in rows],'No prior miss','#777777',linestyle='--')
        ax.set_xticks(x,[('Token' if r['definition']=='token' else 'Context4')+'\n'+('Sibling' if r['source']=='sibling_response' else 'Other prompt') for r in rows])
        ax.set_ylabel('Matched miss probability');ax.legend(frameon=False,fontsize=7)
        finish(fig,ax,destination,f'E3_step_{step}')
        fig,axs = plt.subplots(1,2,figsize=(6.8,2.45));rows=result['E6_frequency'];x=np.arange(3)
        curve(axs[0],x-.05,[r['observed'] for r in rows],'Observed','#0072B2')
        curve(axs[0],x+.05,[dict(estimate=r['null_mean'],ci95=r['null_interval95']) for r in rows],
              'Frequency null','#D55E00',linestyle='--')
        axs[0].set_xticks(x,['Response','Prompt','Batch']);axs[0].set_ylabel('Availability at misses')
        axs[0].legend(frameon=False,fontsize=7)
        rows=result['E6_donor']
        curve(axs[1],x,[r['availability'] for r in rows],'Recipient population','#0072B2',linestyle='')
        axs[1].set_xticks(x,['Within batch','Donor','Donor only']);axs[1].set_ylabel('Availability at recipient misses')
        finish(fig,axs,destination,f'E6_step_{step}')


def plot_rq2(destination, summary=None):
    destination = Path(destination)
    if summary is None:summary = json.loads((destination/'rq2_repeated_misses.json').read_text())
    plt = pyplot();steps=sorted(summary['checkpoints'],key=int)
    fig,axes = plt.subplots(1,max(1,len(steps)),figsize=(6.8 if len(steps)>1 else 3.35,2.6),squeeze=False,sharey=True)
    for ax,step in zip(axes[0],steps):
        rows=summary['checkpoints'][step]['distances'];x=np.arange(len(rows))
        for arm,color,offset in (('off','#777777',-.06),('on','#0072B2',.06)):
            for exposure,style,marker in (('prior','-','o'),('no_prior','--','s')):
                curve(ax,x+offset,[r['matched']['risks'][arm+'_'+exposure] for r in rows],
                      f'{arm.upper()} / '+('prior miss' if exposure=='prior' else 'control'),color,marker,style)
        ax.set_title(f'Step {step}');ax.set_xticks(x,[r['distance'] for r in rows])
        ax.set_xlabel('Prior-miss round distance');ax.tick_params(axis='x',labelsize=8)
    axes[0,0].set_ylabel('Matched miss probability')
    if steps:axes[0,0].legend(frameon=False,fontsize=6.5,ncol=2)
    finish(fig,axes,destination,'rq2_repeated_misses',png=True)
