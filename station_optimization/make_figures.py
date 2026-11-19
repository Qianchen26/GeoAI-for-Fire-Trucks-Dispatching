"""Render standalone research figures from saved evidence; run with /opt/anaconda3/bin/python.

No training or simulations. Final E1/E2 panels require execution_summary.json status=complete.
"""
import os
from pathlib import Path
HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / "cache" / "matplotlib"))
import json
import pickle
import warnings
warnings.filterwarnings("ignore", message="Pandas requires version.*numexpr")
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.patheffects as pe

ROOT, RES, FIG = HERE.parent, HERE / "results", HERE / "figures"
FIG.mkdir(exist_ok=True)
SOURCE = ROOT / "data" / "models" / "planning" / "final"
TEAL, GRAY, INK, GOLD, BROWN = "#007f86", "#65747e", "#18344b", "#e8a42a", "#995414"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.4,
    "axes.titlesize": 8.8, "axes.labelsize": 8.4, "axes.edgecolor": "#78828a",
    "xtick.labelsize": 7.4, "ytick.labelsize": 7.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white"})
METRICS = ("D_F", "M_F", "M_O")
NAMES = {"D_F": r"$D_F$: between-tract dispersion", "M_F": r"$M_F$: mean fire travel",
         "M_O": r"$M_O$: mean other-call travel"}
DRAW_CASES = [("Medium", 1.0), ("High", 1.0), ("High", 0.0)]
MANIFEST = []


def save(fig, stem):
    for suffix in ("pdf", "png"):
        # Use the final publication width: downstream scaling must not shrink type.
        fig.savefig(FIG / f"{stem}.{suffix}", dpi=300)
    plt.close(fig)
    print(stem, flush=True)


def scalarbar(fig, ax, norm, cmap, label, ticks=None):
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                     orientation="horizontal", fraction=0.045, pad=0.035, aspect=30, ticks=ticks)
    cb.set_label(label, labelpad=3)
    cb.ax.tick_params(labelsize=7.4)
    return cb


def map_base(ax, title, scale=False):
    tracts.boundary.plot(ax=ax, color="#777e82", linewidth=0.25, zorder=2)
    county.boundary.plot(ax=ax, color=INK, linewidth=0.75, zorder=3)
    ax.set(xlim=bounds[[0, 2]], ylim=bounds[[1, 3]], title=title)
    ax.set_aspect("equal")
    ax.set_axis_off()
    if scale:
        x, y = bounds[0] + 2200, bounds[1] + 2200
        ax.plot([x, x + 10000], [y, y], color=INK, linewidth=2, zorder=20)
        ax.text(x + 5000, y + 850, "10 km", ha="center", fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.5), zorder=21)
        ax.annotate("N", xy=(0.96, 0.10), xytext=(0.96, 0.02), xycoords="axes fraction",
                    ha="center", fontsize=8, arrowprops=dict(arrowstyle="-|>", color=INK))


def choropleth(ax, values, cmap, norm, title, scale=False):
    data = tracts.assign(value=np.asarray(values, dtype=float))
    data.plot(column="value", ax=ax, cmap=cmap, norm=norm, linewidth=0,
              missing_kwds={"color": "#e3e3e3", "hatch": "////", "edgecolor": "#aaa", "linewidth": 0.25})
    map_base(ax, title, scale)


def station_points(ax, data= None, muted=False):
    data = stations if data is None else data
    ax.scatter(data.x, data.y, s=15 if muted else 20, marker="^",
               color=INK, edgecolor="white", linewidth=0.4, zorder=8)


def pending(ax):
    ax.set_axis_off()
    ax.text(0.5, 0.5, "Paired event-simulation results\npending final CSV bundle", ha="center",
            va="center", color=GRAY, transform=ax.transAxes)


def paired_point(ax, value, low, high, y, color=TEAL, marker="o"):
    assert np.isfinite([value, low, high]).all()
    assert low <= high
    ax.hlines(y, low, high, color=color, linewidth=1.5)
    ax.plot(value, y, marker, color=color, markersize=3.7)


def resource_counts(policy, path, w, year=2045):
    h = next(h for h in plan["fast_histories"][(policy, (path, w))]["history"] if h["year"] == year)
    cfg = h["config"]
    return (int((np.asarray(cfg["site"]) >= 0).sum()), int(np.sum(cfg["engines"])), int(np.sum(cfg["quints"])))


def forest_style(ax, labels, title, xlabel):
    ax.axvline(0, color="#9aa4ac", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels)-0.5, -0.5)
    ax.set_title(title, loc="left", pad=8)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color="#e8ebed", linewidth=0.6)
    ax.tick_params(axis="y", length=0, labelsize=8.1)
    ax.spines["left"].set_visible(False)


METRIC_NOTE = ('D_F is the fixed-fire-rate-weighted standard deviation of tract mean fire travel; '
               'M_F and M_O are incident-level regional means for study-fire and other calls. '
               'Travel is en route to arrival.')
STREAM_NOTE = ('Circles and 95% intervals use 2,000 paired bootstrap samples of 15 whole event streams '
               '(seeds 101–115); diamonds show the saved fast-evaluator differences.')


def record(fig, stem, role, number, label, title, caption):
    save(fig, stem)
    MANIFEST.append(dict(filename=f'figures/{stem}.pdf', png=f'figures/{stem}.png',
        block=stem, label=label, role=role, order=int(str(number).replace('S', '')),
        caption_number=str(number), caption=f'**Figure {number}. {title}** {caption}'))


def map_canvas():
    fig = plt.figure(figsize=(7, 7.1))
    ax = fig.add_axes([0.08, 0.21, 0.84, 0.76])
    return fig, ax


def map_bar(fig, norm, cmap, label, ticks=None):
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=fig.add_axes([0.14, 0.125, 0.72, 0.018]), orientation='horizontal', ticks=ticks)
    cb.set_label(label, labelpad=4, fontsize=8.8)
    cb.ax.tick_params(labelsize=8.1)
    return cb


def forest_canvas(height=5, left=0.28):
    fig, ax = plt.subplots(figsize=(7, height))
    fig.subplots_adjust(left=left, right=0.96, bottom=0.20, top=0.97)
    return fig, ax


def event_legend(fig):
    fig.legend(handles=[Line2D([], [], color=TEAL, marker='o', markersize=4,
        label='Event simulation + 95% paired CI'), Line2D([], [], color=GRAY,
        marker='D', linestyle='none', markersize=4, label='Fast-evaluator point')],
        loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=2, frameon=False, fontsize=8.1)


def study_maps():
    for kind, stem, role, number, label in [
        ('population', 'fig01_study_area', 'main', 1, 'fig:study_area'),
        ('fire', 'supp04_fire_weights', 'supplement', 'S4', 'fig:fire_weights'),
        ('travel', 'supp05_baseline_travel', 'supplement', 'S5', 'fig:baseline_travel')]:
        fig, ax = map_canvas()
        if kind == 'population':
            values = tracts.pop2025 / (tracts.area / 1e6)
            norm, cmap = LogNorm(min(10, values.min()), max(5000, values.max())), 'YlGnBu'
            bar = 'Estimated 2025 residents per km² (log scale)'
            title = 'Population and modeled station locations in Alachua County.'
            caption = ('Estimated 2025 population is divided by the total area of each 2020 census tract. '
                'Triangles show the 25 modeled 2025 station/base locations, with 23 engines and five quints. '
                'The dashed 2026 Gainesville boundary provides geographic context. '
                'The study covers 58 tracts, including 57 with positive baseline study-fire rates. '
                'The population-density map uses a logarithmic scale and UTM zone 17N.')
        elif kind == 'fire':
            values, cmap = tracts.fire_weight*100, 'YlOrBr'
            norm = Normalize(0, values.max())
            bar = 'Share of county baseline study-fire rate (%)'
            title = 'Spatial weights for study-fire demand.'
            caption = ('Each tract contributes its share of the fixed study-fire rate estimated from '
                'October 2024–March 2025 incidents. Study-fire calls comprise outside, structure, vehicle, '
                'and brush fires; other calls form the remaining modeled call group. These shares are '
                'county demand weights, rather than per-resident risk rates. One of the 58 tracts has '
                'zero baseline fire rate. The scenarios retain these fire rates while non-fire workload '
                'changes with population. Triangles show the initial modeled station/base locations.')
        else:
            h = plan['fast_histories'][('do nothing', ('Medium', 1.0))]['history'][0]
            values = np.where(tracts.fire_weight > 0, h['tract_fire_mean'], np.nan)
            norm, cmap = Normalize(np.nanmin(values), np.nanmax(values)), 'magma_r'
            bar = 'Tract mean first-arriving fire travel (min)'
            title = 'Baseline service under the 2030 Medium trend scenario.'
            caption = ('The fast evaluator estimates first-arriving eligible engine/quint travel with '
                'the unchanged 2025 configuration and 2030 Medium historical-trend growth (w = 1). '
                'Travel is en route to arrival. The hatched tract has zero baseline fire rate and an '
                'undefined fire-travel mean. Triangles show the initial modeled station/base locations. '
                'The scenario keeps fire rates fixed and changes non-fire workload with population.')
        choropleth(ax, values, cmap, norm, '', True)
        city.boundary.plot(ax=ax, color='#151515', linewidth=0.8, linestyle='--', zorder=6)
        station_points(ax)
        center = city.union_all().centroid
        ax.annotate('Gainesville', (center.x, center.y), xytext=(25, -40),
            textcoords='offset points', color=INK, fontsize=9,
            path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])
        cb = map_bar(fig, norm, cmap, bar, [10,100,1000,5000] if kind=='population' else None)
        if kind == 'population': cb.set_ticklabels(['10','100','1,000','5,000'])
        handles=[Line2D([], [], marker='^', color='none', markerfacecolor=INK,
            label='Modeled station / base'), Line2D([], [], color='#151515', linestyle='--',
            label='Gainesville city limits')]
        if kind == 'travel': handles.append(Patch(facecolor='#e3e3e3', hatch='////',
            edgecolor='#aaa', label='Zero baseline fire rate'))
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5,0.01),
            ncol=2, frameon=False, fontsize=8.1)
        record(fig, stem, role, number, label, title, caption)


def e1_forests():
    frame = pd.read_csv(RES/'paired_comparisons.csv')
    frame = frame[frame.experiment=='E1'].set_index(['path','w','year'])
    order=[(p,w,year) for p,w in DRAW_CASES for year in (2035,2045)]
    for metrics, stem, role, number, label in [
        (('D_F',), 'fig03_policy_results', 'supplement', 'S1', 'fig:policy_results'),
        (('M_F','M_O'), 'fig06_policy_means', 'main', 3, 'fig:policy_means')]:
        fig, ax=forest_canvas(5.2 if len(metrics)==1 else 7.3, left=0.34)
        labels=[]
        for m, metric in enumerate(metrics):
            for k,key in enumerate(order):
                y=m*len(order)+k; row=frame.loc[key]
                paired_point(ax, 60*row[f'delta_{metric}'], 60*row[f'low_{metric}'],
                    60*row[f'high_{metric}'], y-0.10)
                ax.plot(60*row[f'fast_delta_{metric}'], y+0.13, 'D', color=GRAY, markersize=4)
                name='Fire' if metric=='M_F' else 'Other calls' if metric=='M_O' else 'Dispersion'
                labels.append(f'{name} · {key[0]}, w={int(key[1])}\n{key[2]}' +
                    (' · same config.' if row['identical'] else ' · different config.' if key[2]==2035 else ''))
        forest_style(ax, labels, '', 'PPO − stage greedy (s)  |  negative favors PPO')
        ax.tick_params(axis='y', labelsize=8.1)
        if len(metrics)>1: ax.axhline(5.5, color='#bbc3c8', linewidth=.7)
        event_legend(fig)
        title='Regional mean travel under PPO and stage greedy.' if len(metrics)>1 else 'E1 differences in spatial fire-travel dispersion.'
        caption=(f'{STREAM_NOTE} The six scenario–year checks pair Medium trend (w = 1), '
            'High trend (w = 1), and High proportional growth (w = 0) with 2035 and 2045. '
            f'{METRIC_NOTE} Negative differences favor PPO. '
            'The two trend configurations match in 2035; High proportional 2035 uses a different vehicle mix. ')
        if len(metrics)==1:
            caption+=('All four nonidentical simulator dispersion intervals cross zero. '
                'Event-stream intervals describe variation within each scenario–year and exclude evaluator discrepancy.')
        else:
            caption+=('All three 2045 regional mean contrasts favor PPO. Shared stage action-count limits allow '
                'different final fleets because a new station includes an engine. In 2045, PPO/greedy have '
                '30/29 engines and 5/5 quints under Medium trend; 31/28 engines and 5/6 quints under High trend; '
                'and 30/29 engines and 6/6 quints under High proportional growth. The contrasts combine '
                'location, fleet-size, and fleet-type choices. Some other-call intervals are narrower than their markers.')
        record(fig,stem,role,number,label,title,caption)


def draw_forest():
    comp=pd.read_csv(SOURCE/'evaluation_comparisons.csv')
    draws=pd.read_csv(SOURCE/'evaluation_draws.csv'); draws=draws[draws.kind=='draw']
    assert draws.groupby('policy').size().eq(48).all()
    means=draws.groupby('policy')[[f'mean {m}' for m in METRICS]].mean()
    fig,ax=forest_canvas(6.0, left=.32); labels=[]
    for j,metric in enumerate(METRICS):
        for k,(base,name) in enumerate([('do nothing','No action'),('city plan','Modeled Station 9'),('greedy','Stage greedy')]):
            row=comp[(comp.baseline==base)&(comp.metric==f'mean {metric}')].iloc[0]
            assert np.isclose(means.at[base,f'mean {metric}']-means.at['learned',f'mean {metric}'],row['learned better by (mean)'],atol=1e-10)
            paired_point(ax,-60*row['learned better by (mean)'],-60*row['high 95%'],
                -60*row['low 95%'],j*3+k,color=BROWN,marker='s')
            metric_name={'D_F':'Dispersion','M_F':'Fire mean','M_O':'Other-call mean'}[metric]
            labels.append(f'{metric_name} · {name}')
    forest_style(ax,labels,'','PPO − comparator (s)  |  negative favors PPO')
    for y in (2.5,5.5): ax.axhline(y,color='#bbc3c8',linewidth=.7)
    fig.text(.5,.08,'Fast D_F: PPO − greedy = +0.251 s; 95% CI [+0.164, +0.332]',
        ha='center',fontsize=8.1,color=BROWN)
    record(fig,'supp03_growth_draws','supplement','S3','fig:growth_draws',
        'Policy contrasts across 48 held-out growth draws.',
        'Squares show fast-evaluator differences in the mean of four evaluation years (2030, 2035, 2040, 2045), '
        'averaged over 48 held-out growth draws. Intervals are the existing 95% paired bootstrap intervals '
        f'over whole draws. {METRIC_NOTE} Negative values favor PPO on the plotted metric. '
        'PPO has a larger D_F than stage greedy by 0.251 s (95% CI 0.164–0.332 s). '
        'These intervals describe growth-draw variation, distinct from the event-stream variation in E1.')


def join_values(frame,value):
    assert frame.tract.is_unique
    series=frame.assign(tract=frame.tract.astype(str)).set_index('tract')[value]
    assert set(series.index)==set(tracts.tract)
    return np.where(tracts.fire_weight>0,series.reindex(tracts.tract).to_numpy(dtype=float),np.nan)


def consequence_maps():
    fast=pd.read_csv(RES/'fast_tract_outcomes.csv',dtype={'tract':str})
    simulated=pd.read_csv(RES/'spatial_contrasts.csv',dtype={'tract':str})
    for i,(path,w) in enumerate(DRAW_CASES):
        for source in ('fast','simulation'):
            if source=='fast':
                sub=fast[(fast.path==path)&(fast.w==w)&(fast.year==2045)]
                values=60*(join_values(sub[sub.policy=='learned'],'fire_mean')-join_values(sub[sub.policy=='do nothing'],'fire_mean'))
                norm=TwoSlopeNorm(vmin=-160,vcenter=0,vmax=160); ref='no action'
                stem='fig04_spatial_consequences' if i==0 else f'supp{4+i*2:02d}_high_w{int(w)}_fast'
                number='S2' if i==0 else f'S{4+i*2}'; role='supplement'
                label='fig:spatial_consequences' if i==0 else f'fig:high_w{int(w)}_fast'
            else:
                sub=simulated[(simulated.experiment=='E1')&(simulated.path==path)&(simulated.w==w)&(simulated.year==2045)]
                values=60*join_values(sub,'a_minus_b')
                norm=TwoSlopeNorm(vmin=-120,vcenter=0,vmax=120); ref='stage greedy'
                stem='fig07_spatial_policy_diff' if i==0 else f'supp{5+i*2:02d}_high_w{int(w)}_simulation'
                number=4 if i==0 else f'S{5+i*2}'; role='main' if i==0 else 'supplement'
                label='fig:spatial_policy_diff' if i==0 else f'fig:high_w{int(w)}_simulation'
            assert np.nanmax(np.abs(values))<=norm.vmax
            fig,ax=map_canvas(); choropleth(ax,values,'RdBu_r',norm,'',True)
            faster=tracts.loc[values/60 < -1e-10,'fire_weight'].sum()*100
            slower=tracts.loc[values/60 > 1e-10,'fire_weight'].sum()*100
            same=tracts.loc[np.isfinite(values)&(abs(values/60)<=1e-10),'fire_weight'].sum()*100
            a=resource_counts('learned',path,w); b=resource_counts('do nothing' if source=='fast' else 'greedy',path,w)
            ax.text(.5,-.02,f'Fixed fire-demand share: faster {faster:.1f}% · slower {slower:.1f}% · unchanged {same:.1f}%\n'
                f'PPO / {ref}: {a[0]}/{b[0]} locations · {a[1]+a[2]}/{b[1]+b[2]} vehicles',
                transform=ax.transAxes,ha='center',va='top',fontsize=8.1)
            map_bar(fig,norm,'RdBu_r',f'PPO − {ref}: tract mean fire travel (s)\nBlue: faster under PPO · red: slower under PPO',
                [-norm.vmax,-norm.vmax/2,0,norm.vmax/2,norm.vmax])
            fig.legend(handles=[Patch(facecolor='#e3e3e3',hatch='////',edgecolor='#aaa',
                label='Zero baseline fire rate or missing paired fire outcome')],
                loc='lower center',bbox_to_anchor=(.5,.005),frameon=False,fontsize=8.1)
            caption=(f'The 2045 {path} scenario uses '+('historical-trend' if w else 'proportional')+
                f' population allocation (w = {int(w)}). Differences compare frozen PPO with {ref}, using '+
                ('saved fast-evaluator tract means. ' if source=='fast' else 'tract means pooled over 15 paired E1 event streams. ')+
                'Travel is first-arriving eligible vehicle travel from en route to arrival. Negative values favor PPO. '
                f'The symmetric color range is ±{norm.vmax:.0f} s, shared with the other scenarios for this comparison. '
                'Annotations sum fixed baseline fire-rate weights in faster, slower, and unchanged tracts; differences '
                'within 10⁻¹⁰ min count as numerical equality. These categories use point estimates. '+
                ('No action retains the initial 25 locations and 28 vehicles.' if source=='fast' else
                 'The policies share stage action-count limits and select different final locations and fleets.'))
            record(fig,stem,role,number,label,f'Local fire-travel consequences: PPO versus {ref}.',caption)


def e2_figures():
    growth=pd.read_csv(RES/'e2_growth_by_tract.csv',dtype={'tract':str})
    growth=growth[growth.year==2035]
    norm=Normalize(0,100*(growth.population_multiplier.max()-1))
    common=stations.copy(); pos=candidates.set_index('candidate')
    common.loc[common.station=='GFR_01',['x','y']]=pos.loc['C18',['x','y']].to_numpy()
    common=pd.concat([common,pd.DataFrame([{'station':c,'x':pos.at[c,'x'],'y':pos.at[c,'y']}
        for c in ('C16','C19')])],ignore_index=True)
    for w in (0.,1.):
        sub=growth[growth.w==w].set_index('tract').reindex(tracts.tract)
        fig,ax=map_canvas(); choropleth(ax,100*(sub.population_multiplier-1),'YlGnBu',norm,'',True)
        station_points(ax,common)
        for station,text in [('E62','A: add engine at E62 base'),('CO_29','B: add engine at CO_29')]:
            loc=common.set_index('station').loc[station]; selected=(station==('E62' if w==0 else 'CO_29'))
            ax.scatter(loc.x,loc.y,s=75,facecolors=GOLD if selected else 'white',edgecolors=INK,linewidths=1.2,zorder=9)
            ax.annotate(text,(loc.x,loc.y),xytext=(-5 if station=='E62' else 5,10),
                textcoords='offset points',ha='right' if station=='E62' else 'left',fontsize=8.1,color=INK,
                path_effects=[pe.withStroke(linewidth=2.5,foreground='white')],zorder=10)
        ax.text(.5,-.02,f'County population in 2035: {sub.population.sum():,.0f}',
            transform=ax.transAxes,ha='center',va='top',fontsize=8.4)
        map_bar(fig,norm,'YlGnBu','Population growth from 2025 to the 2035 decision (%)')
        fig.legend(handles=[Line2D([],[],marker='o',color='none',markerfacecolor=GOLD,markeredgecolor=INK,
            label='PPO-selected engine location'),Line2D([],[],marker='^',color='none',markerfacecolor=INK,
            label='Common station / base configuration')],loc='lower center',bbox_to_anchor=(.5,.01),
            ncol=2,frameon=False,fontsize=8.1)
        record(fig,'fig08_e2_growth' if w else 'supp10_e2_proportional_growth',
            'main' if w else 'supplement',5 if w else 'S10','fig:e2_growth' if w else 'fig:e2_growth_s0',
            'Spatial growth and the engine-addition decision in E2.',
            f'The Medium county path uses '+('historical-trend allocation (S1, w = 1). ' if w else 'proportional allocation (S0, w = 0). ')+
            'Triangles show the common configuration immediately after the shared GFR_01 relocation to C18. '
            'The subsequent vehicle decision chooses between A, adding an engine at the E62 modeled base, '
            'and B, adding an engine at CO_29. Gold marks the frozen PPO choice. Both states have the same '
            'county population and resource configuration; S0 applies uniform proportional growth and '
            'selects A, while S1 selects B. Their growth maps share the same scale. Fire rates remain fixed; '
            'population changes affect non-fire workload.')
    remaining=pd.read_csv(RES/'e2_remaining_horizon.csv').set_index('w')
    fig,ax=forest_canvas(5.2,left=.28); labels=[]
    for i,w in enumerate((0.,1.)):
        row=remaining.loc[w]
        for j,m in enumerate(METRICS):
            y=i*3+j
            paired_point(ax,60*row[f'delta_{m}'],60*row[f'low_{m}'],60*row[f'high_{m}'],y-.1)
            ax.plot(60*row[f'fast_delta_{m}'],y+.13,'D',color=GRAY,markersize=4)
            labels.append(f'S{i}, w={int(w)} · ${m}$')
    forest_style(ax,labels,'','A − B (s)  |  negative favors A')
    ax.tick_params(axis='y',labelsize=10.5)
    ax.axhline(2.5,color='#bbc3c8',linewidth=.7)
    event_legend(fig)
    fig.text(.5,.105,'Fast return A − B: S0 +0.0552 min (PPO selects A); S1 −0.0031 min (PPO selects B)',
        ha='center',fontsize=8.1,color=BROWN)
    record(fig,'fig05_e2_case','main',6,'fig:e2_case','Action consequences in the matched E2 states.',
        'A adds an engine at the E62 modeled base; B adds one at CO_29. S0 uses proportional growth '
        '(w = 0), and S1 uses historical-trend allocation (w = 1), with the same Medium county population '
        'and post-relocation resource configuration. Each point averages the 2040 and 2045 outcomes. '
        f'{STREAM_NOTE} Each bootstrap sample uses the same stream indices in both years. {METRIC_NOTE} '
        'The frozen continuations all build at C12 and add an engine at CO_29 in 2040. The original fast '
        'reward favors A in S0 through a smaller other-call penalty, and B in S1 through lower dispersion '
        'with zero regional-mean penalties. The reward hinge starts at the exact 2025 baseline; the '
        '0.05-min tolerance applies to violation reporting. Simulator D_F and M_F intervals cross zero '
        'in both states, while M_O favors A in both.')
    spatial=pd.read_csv(RES/'spatial_contrasts.csv',dtype={'tract':str})
    spatial=spatial[spatial.experiment=='E2']; assert spatial.groupby(['w','tract']).size().eq(2).all()
    effects=spatial.groupby(['w','tract']).a_minus_b.sum(min_count=2).div(2).reset_index()
    extent=max(1,np.nanmax(abs(effects.a_minus_b))*60)
    norm=TwoSlopeNorm(vmin=-extent,vcenter=0,vmax=extent)
    for i,w in enumerate((0.,1.)):
        fig,ax=map_canvas(); values=60*join_values(effects[effects.w==w],'a_minus_b')
        choropleth(ax,values,'RdBu_r',norm,'',True)
        map_bar(fig,norm,'RdBu_r','A − B: mean tract fire-travel difference, 2040 and 2045 (s)\nBlue: A faster · red: B faster')
        fig.legend(handles=[Patch(facecolor='#e3e3e3',hatch='////',edgecolor='#aaa',label='Zero baseline fire rate or missing outcome')],
            loc='lower center',bbox_to_anchor=(.5,.01),frameon=False,fontsize=8.1)
        record(fig,f'supp{11+i}_e2_spatial_s{i}','supplement',f'S{11+i}',f'fig:e2_spatial_s{i}',
            f'Local E2 action consequences in state S{i}.',
            f'The Medium county path uses w = {int(w)}. A adds an engine at E62; B adds one at CO_29. '
            'Each tract value averages its A−B first-arriving fire-travel differences in 2040 and 2045, '
            'using the 15 paired event streams. Travel is en route to arrival. Negative differences '
            'favor A. Both state maps share a symmetric color scale. The resource configuration and '
            'county population match at the decision; the frozen continuations use the same 2040 actions. '
            'Hatched areas have zero baseline fire rate or a missing paired outcome.')


if __name__=='__main__':
    assert json.loads((RES/'execution_summary.json').read_text())['status']=='complete'
    tracts=gpd.read_file(RES/'tracts.geojson'); tracts['tract']=tracts.tract.astype(str)
    assert tracts.tract.is_unique and np.isclose(tracts.fire_weight.sum(),1)
    assert tracts.crs.to_epsg()==32617
    stations=pd.read_csv(RES/'stations.csv'); candidates=pd.read_csv(RES/'candidate_sites.csv')
    county=gpd.read_file(ROOT/'data/raw/gnv_alachua/alachua_county_boundary.gpkg').to_crs(tracts.crs)
    city=gpd.read_file(ROOT/'data/raw/gnv_alachua/gainesville_city_limits_2026.gpkg').to_crs(tracts.crs)
    bounds=county.total_bounds+np.array([-1200,-1200,1200,1200])
    plan=pickle.loads((RES/'plan.pkl').read_bytes())
    assert list(tracts.tract)==list(map(str,plan['tract_ids']))
    study_maps(); e1_forests(); draw_forest(); consequence_maps(); e2_figures()
    for role,name in [('main','figure_captions.md'),('supplement','supplementary_figure_captions.md')]:
        selected=sorted([r for r in MANIFEST if r['role']==role],key=lambda r:r['order'])
        (HERE/name).write_text('# '+('Research' if role=='main' else 'Supplementary')+' figure captions\n\n'+
            '\n\n'.join(r['caption'] for r in selected)+'\n')
    MANIFEST.append(dict(filename='figures/framework.pdf',png='figures/framework.png',block='fig02_framework',
        label='fig:framework',role='main',order=2,caption_number='2',caption_source='framework agent'))
    (HERE/'figure_manifest.json').write_text(json.dumps(dict(format='standalone-single-panel',
        main_count=6,supplementary_count=12,figures=sorted(MANIFEST,key=lambda r:(r['role']!='main',r['order']))),indent=2)+'\n')
    print('Rendered 17 single-panel research figures; framework supplied separately.')
