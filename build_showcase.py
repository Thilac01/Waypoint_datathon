"""Build accurate charts and a one-page report from the saved evidence.

Run decision_intelligence.py first. No model training or submission writes occur.
python build_showcase.py --output outputs
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.colors import toColor as HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph

NAVY, TEAL, ORANGE, MUTED = '#142c3d', '#007f7b', '#bd592e', '#516474'
BG, LINE = '#f3f7f8', '#dbe4e9'


def load(folder, name):
    return json.loads((folder / name).read_text())


def charts(out):
    r = out / 'reports'
    evidence = load(r, 'policy_sensitivity.json')
    forecast = pd.read_csv(r / 'ten_week_network_forecast.csv')
    segment = pd.read_csv(r / 'task2a_by_brand_depot.csv')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.labelcolor': NAVY, 'text.color': NAVY,
                         'axes.edgecolor': LINE, 'xtick.color': MUTED, 'ytick.color': MUTED})
    fig, ax = plt.subplots(figsize=(12, 3.5), layout='constrained')
    vals = [evidence['chilled_demand_m3'], evidence['optimistic_refrigerated_capacity_m3']]
    ax.barh([1, 0], vals, color=[ORANGE, TEAL], height=.5)
    ax.set_yticks([1, 0], ['Peak-day chilled demand', 'Optimistic refrigerated capacity'])
    ax.set_xlim(0, 215)
    for i, value in enumerate(vals):
        ax.text(value + 2, 1-i, f'{value:.3f} m³', va='center', fontweight='bold')
    ax.set_xlabel('Volume (m³); capacity assumes two full-volume trips per refrigerated vehicle')
    ax.set_title(f"A {evidence['optimistic_shortfall_m3']:.3f} m³ shortfall, before other constraints", loc='left', fontweight='bold', pad=18)
    ax.grid(axis='x', alpha=.15)
    ax.set_axisbelow(True)
    fig.savefig(r / 'capacity_vs_demand.png', dpi=170, facecolor='white')
    plt.close(fig)

    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={'width_ratios':[1.1, 1]}, layout='constrained')
    a.plot(forecast.iso_week, forecast.pred_total_volume_m3, color=NAVY, marker='o', lw=2.5, label='Total')
    a.plot(forecast.iso_week, forecast.pred_chilled_volume_m3, color=TEAL, marker='o', lw=2.5, label='Chilled (part of total)')
    peak = forecast.loc[forecast.pred_total_volume_m3.idxmax()]
    a.annotate(f'Week {int(peak.iso_week)}: {peak.pred_total_volume_m3:,.0f} m³',
               (peak.iso_week, peak.pred_total_volume_m3), xytext=(16.2, 2110),
               fontsize=10, arrowprops={'arrowstyle':'-', 'color': MUTED})
    a.set_ylim(0, 2270)
    a.set_xticks(forecast.iso_week)
    a.set_xlabel('ISO week, 2026')
    a.set_ylabel('Predicted network demand (m³/week)')
    a.set_title('Anticipate the next ten weeks', loc='left', fontweight='bold')
    a.legend(loc='lower right', frameon=False, fontsize=10)
    a.grid(alpha=.15)
    data = segment[(segment.target == 'total') & (segment.slice == 'depot+brand')].copy()
    data['label'] = data.depot + ' / ' + data.brand
    y = np.arange(len(data))
    b.barh(y, data.wape * 100, color=[ORANGE if x == 'Tech' else TEAL for x in data.brand], height=.58)
    b.set_yticks(y, data.label, fontsize=10)
    b.invert_yaxis()
    b.set_xlim(0, 36)
    for i, v in enumerate(data.wape):
        b.text(v*100+.6, i, f'{v:.1%}', va='center', fontsize=10)
    b.set_xlabel('Final holdout WAPE (%) · lower is better')
    b.set_title('Keep the weak segments visible', loc='left', fontweight='bold')
    b.grid(axis='x', alpha=.15)
    b.set_axisbelow(True)
    fig.savefig(r / 'forecast_and_errors.png', dpi=170, facecolor='white')
    plt.close(fig)


def report(out, dest):
    pdfmetrics.registerFont(TTFont('Showcase', font_manager.findfont('DejaVu Sans')))
    pdfmetrics.registerFont(TTFont('Showcase-Bold', font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans', weight='bold'))))
    pdfmetrics.registerFontFamily('Showcase', normal='Showcase', bold='Showcase-Bold')
    r = out / 'reports'
    e, p = load(r, 'decision_evidence.json'), load(r, 'policy_sensitivity.json')
    t1, t2, alloc = [load(r, n) for n in ['task1_metrics.json', 'task2a_metrics.json', 'task2b_summary.json']]
    policies = pd.read_csv(r / 'policy_comparison.csv')
    balanced = policies[policies.policy.eq('Balanced')].iloc[0]
    forecast = pd.read_csv(r / 'ten_week_network_forecast.csv')
    W, H = landscape(A4)
    c = canvas.Canvas(str(dest), pagesize=(W, H))
    c.setTitle('Alt-F4 | Explainable Logistics Decision Intelligence')
    c.setAuthor('Alt-F4; AI-assisted draft, review disclosed separately')

    def box(x, y, w, h, color='white', radius=8):
        c.setFillColor(HexColor(color)); c.roundRect(x,y,w,h,radius,fill=1,stroke=0)
    def text(x,y,s,size=10,color=NAVY,bold=False):
        c.setFillColor(HexColor(color)); c.setFont('Showcase-Bold' if bold else 'Showcase',size); c.drawString(x,y,str(s))
    def para(x,top,w,s,size=9,color=MUTED,leading=None,bold=False):
        st=ParagraphStyle('p',fontName='Showcase-Bold' if bold else 'Showcase',fontSize=size,
                          leading=leading or size*1.35,textColor=HexColor(color))
        v=Paragraph(s,st); aw,ah=v.wrap(w,200); v.drawOn(c,x,top-ah); return ah

    c.setFillColor(HexColor(BG)); c.rect(0,0,W,H,fill=1,stroke=0)
    c.setFillColor(HexColor(NAVY)); c.rect(0,H-91,W,91,fill=1,stroke=0)
    text(30,H-23,'ALT-F4  /  TECH-TRIATHLON 2026',9,'#88dfd5',True)
    text(30,H-51,'Explainable Logistics Decision Intelligence',23,'white',True)
    text(30,H-73,'Predict delivery risk. Anticipate demand. Make scarce refrigeration an explicit business decision.',10,'#d4e4eb')

    # Dominant evidence: a physical lower bound and a solver-certified service ceiling.
    box(30,303,782,187)
    text(47,470,f"WHY ALL {alloc['served'] + alloc['deferred']} ORDERS CANNOT BE DELIVERED",11,NAVY,True)
    text(47,446,'Refrigeration is a physical bottleneck',16,NAVY,True)
    barx, barw = 47, 342
    scale = barw / 190
    for label,val,y,col in [('Chilled demand',p['chilled_demand_m3'],406,ORANGE),
                             ('Optimistic capacity',p['optimistic_refrigerated_capacity_m3'],359,TEAL)]:
        text(barx,y+18,label,9,MUTED)
        box(barx,y,scale*val,11,col,2)
        text(barx+barw-63,y+18,f'{val:.3f} m³',10,col,True)
    para(47,341,347,f"{p['available_reefers']} refrigerated vehicles × two full-volume trips. This optimistic bound ignores packing, time, weight and access limits.",8.2)
    c.setStrokeColor(HexColor(LINE)); c.line(418,322,418,463)
    text(441,448,f"{p['optimistic_shortfall_m3']:.3f} m³",29,ORANGE,True)
    text(441,431,'shortfall before the other constraints',10,MUTED)
    for x,value,label in [(441,str(alloc['served']),'orders served'),(560,str(alloc['deferred']),'deferred'),(672,f"{alloc['relative_gap']:.0%}",'solver gap')]:
        text(x,390,value,30,NAVY,True); text(x,374,label,9,MUTED)
    para(441,357,341,f"<b>Proof:</b> the separate maximum-count solve also certifies {alloc['max_count_comparison']['served']}. Every allocation passes the official checker. Optimality covers the implemented competition rules.",9)

    # Each prediction is connected to a distinct action, with scope kept explicit.
    for x in [30,294,558]: box(x,170,254,120)
    text(45,270,'01  /  PREDICT DELIVERY RISK',9,TEAL,True)
    text(45,246,f"{t1['service']['holdout_metrics']['mae']:.2f} min MAE  ·  {t1['lateness']['holdout_metrics']['roc_auc']:.3f} AUC",12.7,NAVY,True)
    para(45,235,224,f"{e['task1']['flagged_at_0_5']:,} of {e['task1']['planned_orders']:,} test orders flag at ≥50% late risk. <b>Action:</b> review planned slack and handling buffers before dispatch.",9)
    text(45,180,'Six-week test horizon; risks are predictions.',7.5,MUTED)

    text(309,270,'02  /  ANTICIPATE DEMAND',9,TEAL,True)
    text(309,246,f"Week {int(e['forecast_peak_week']['iso_week'])} peak: {e['forecast_peak_week']['pred_total_volume_m3']:,.0f} m³",12.7,NAVY,True)
    para(309,235,224,f"Including {e['forecast_peak_week']['pred_chilled_volume_m3']:,.0f} m³ chilled. <b>Action:</b> plan stock and refrigerated availability ahead of the ten-week forecast.",9)
    # Sparkline, clearly labelled as predicted total demand.
    values=forecast.pred_total_volume_m3.to_numpy()
    path=c.beginPath()
    for i,v in enumerate(values):
        xx=310+i*12; yy=180+(v-values.min())/(values.max()-values.min())*14
        if i==0:path.moveTo(xx,yy)
        else:path.lineTo(xx,yy)
    c.setStrokeColor(HexColor(TEAL));c.setLineWidth(1.5);c.drawPath(path)
    text(428,183,'Weeks 14–23',7.5,MUTED)

    text(573,270,'03  /  ALLOCATE CAPACITY',9,TEAL,True)
    text(573,246,f"{balanced.served_orders} served; {balanced.chilled_volume_delivered_m3:.3f} m³ chilled",12.2,NAVY,True)
    para(573,235,224,'<b>Action:</b> retain the balanced dispatch, explain each deferral, and choose explicitly whether backlog fairness should outweigh throughput.',9)
    text(573,180,'Peak-day scenario; whole-order decisions.',7.5,MUTED)

    box(30,42,474,115)
    text(45,139,'THE COST OF CHANGING POLICY',10,NAVY,True)
    headers=[('Policy',45),('Served',172),('Chilled m³',219),('Points*',296),('Deferred',358),('Backlog**',420)]
    for label,x in headers:text(x,122,label,8,MUTED,True)
    for i,row in enumerate(policies.itertuples()):
        yy=104-i*16
        vals=[row.policy,str(row.served_orders),f'{row.chilled_volume_delivered_m3:.3f}',f'{row.balanced_reference_points:,}',str(row.deferred_orders),f'{row.repeat_deferred_orders_served}/10']
        for (_,x),v in zip(headers,vals):text(x,yy,v,8.4,ORANGE if row.policy=='Fairness-first' else NAVY)
    text(45,55,'*Common balanced weights. **Previously deferred orders served. Hard rules unchanged.',7.2,MUTED)

    box(516,42,296,115)
    text(531,139,'HONEST EVALUATION',10,NAVY,True)
    para(531,127,266,f"Total-demand WAPE: <b>{t2['total']['holdout_metrics']['wape']:.2%}</b>; chilled: <b>{t2['chilled']['holdout_metrics']['wape']:.2%}</b>. Tech total-demand WAPE: <b>{t2['total']['holdout_by_series']['Kandy:Tech']['wape']:.1%} Kandy</b>, <b>{t2['total']['holdout_by_series']['Peliyagoda:Tech']['wape']:.1%} Peliyagoda</b>. Several per-series R² values are negative.",8.4)
    para(531,87,266,'Nine simple Tech baselines on earlier chronological folds give modest/inconsistent gains. Verified predictions retained; final holdout excluded from that experiment.',8.2)
    para(30,32,782,'Evidence: saved chronological holdouts, prediction CSVs and CP-SAT reports. The three task horizons are distinct, not one jointly solved route plan. Policy points are team choices, not organizer scores. AI assistance is disclosed; team review remains required.',7.1)
    c.showPage(); c.save()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='outputs')
    p.add_argument('--pdf',default='Alt-F4_Decision_Intelligence.pdf')
    a=p.parse_args(); out=Path(a.output)
    charts(out); report(out,Path(a.pdf))
    print('Created charts and',a.pdf)


if __name__=='__main__':main()
