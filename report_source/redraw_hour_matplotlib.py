"""Redraw Figure 3 with Matplotlib and validate annotation clearance.

python redraw_figure3.py --project-root ADY_Group25 --output-dir figures
"""
from pathlib import Path
import argparse,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

def main(project,out):
    h=pd.read_csv(project/'analysis_outputs/revision/hour_mean_intervals.csv')
    assert h.sessions.sum()==69570
    h23=int(h.loc[h.arrival_hour.eq(23),'sessions'].item())
    assert h23==1669
    out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,
                        'axes.titlesize':8,'axes.labelsize':8,
                        'xtick.labelsize':7.5,'ytick.labelsize':7.5,
                        'pdf.fonttype':42})
    fig,ax=plt.subplots(1,2,figsize=(4.82,2.25))
    fig.subplots_adjust(left=.11,right=.98,bottom=.23,top=.84,wspace=.86)
    blue='#27648c'
    bars=ax[0].bar(h.arrival_hour,h.sessions,width=.82,color=blue)
    ax[0].set(xlim=(-.6,23.6),ylim=(0,7200),
              title='(a) Arrival frequency',xlabel='Arrival hour',ylabel='Sessions')
    peak=int(h.loc[h.arrival_hour.eq(18),'sessions'].item())
    ax[0].annotate(f'{peak:,}',xy=(18,peak),xytext=(0,4),
                   textcoords='offset points',ha='center',va='bottom',fontsize=7.5)
    # A 3-point offset still lets the wide label intersect the taller 21/22h bars.
    # Keep its centre exactly at x=23, raise it into empty space and link it to 23h.
    label=ax[0].annotate('1,669',xy=(23,h23),xytext=(0,32),
                         textcoords='offset points',ha='center',va='bottom',
                         fontsize=7.5,annotation_clip=False,
                         arrowprops={'arrowstyle':'-','color':'#53606a',
                                     'lw':.6,'shrinkA':2,'shrinkB':2})
    ax[1].fill_between(h.arrival_hour,h.lower_95_kWh,h.upper_95_kWh,
                       color=blue,alpha=.18,linewidth=0)
    ax[1].plot(h.arrival_hour,h.mean_kWh,color=blue,lw=1)
    low=h.sessions.lt(500)
    ax[1].scatter(h.loc[low,'arrival_hour'],h.loc[low,'mean_kWh'],
                   marker='^',facecolors='white',edgecolors=blue,s=19,lw=.7,zorder=4)
    ax[1].set(xlim=(-.6,23.6),ylim=(0,35),
              title='(b) Eventual energy',xlabel='Arrival hour',ylabel='Mean energy (kWh)')
    for a in ax:
        a.set_xticks([0,6,12,18,23]);a.spines[['top','right']].set_visible(False)
        a.tick_params(length=3,width=.6)
        a.spines[['bottom','left']].set_linewidth(.6)
    ax[0].set_yticks([0,2000,4000,6000]);ax[1].set_yticks([0,10,20,30])
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    # get_window_extent includes the leader; use the text's own bounding box.
    from matplotlib.text import Text
    bounds=Text.get_window_extent(label,renderer=renderer)
    overlapping=[i for i,b in enumerate(bars) if bounds.overlaps(b.get_window_extent(renderer))]
    assert not overlapping, f'Label intersects bars: {overlapping}'
    assert bounds.x1 < ax[1].get_window_extent(renderer).x0-15
    fig.savefig(out/'figure3.pdf',bbox_inches='tight',pad_inches=.06)
    fig.savefig(out/'figure3.png',dpi=240,bbox_inches='tight',pad_inches=.06)
    (out/'validation.json').write_text(json.dumps({
        'library':'matplotlib','version':matplotlib.__version__,
        'annotation_xy':[23,h23],'annotation_offset_points':[0,32],
        'xlim':[-.6,23.6],'bars_overlapping_label':overlapping,
        'total_sessions':int(h.sessions.sum()),'export':'direct savefig PDF',
        'report_page_imported':False},indent=2),encoding='utf-8')
    plt.close(fig)
    print('PASS: Matplotlib PDF exported; label bounding box overlaps zero bars.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,default=Path('figures'))
    a=p.parse_args();main(a.project_root,a.output_dir)
