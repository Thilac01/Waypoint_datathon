"""Make a narrated replay of real executed-notebook outputs (not a live recording).

Optional presentation tool: Python + FFmpeg (with flite), PyMuPDF and WeasyPrint.
No web requests or organizer data reads. Review or replace the synthetic narration.
python build_demo.py
"""
from pathlib import Path
import argparse
import base64
import hashlib
import html
import json
import re
import subprocess

import pandas as pd
import nbformat
import fitz
from weasyprint import HTML
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter


NARRATION = [
    ('A dispatch decision we can defend',
     "This is Alt F four's Explainable Logistics Decision Intelligence submission. The business question is not just which model has the best score. It is which orders can be delivered, with the fleet actually available. Our verified solution serves seventy nine of eighty five orders. This walkthrough uses real executed notebook outputs. The narration is synthetic and requires team review."),
    ('Correct labels come before clever models',
     "First, the notebook constructs waiting aware service labels. If a vehicle arrives at eight forty, the outlet opens at nine, and handling ends at nine twelve, service is twelve minutes, not thirty two. The twenty minute wait is excluded. Arrival after window close is late; arrival exactly at close is not. We join dispatched orders to their route legs one to one, exclude orders that never ran, and remove actual outcomes before building planned features."),
    ('Train in the past; assess in the future',
     "We compare eight explicit configurations for each delivery target, including boosted trees, extra trees, and linear baselines. Two earlier four week folds choose models and ensemble weights. The six week final holdout comes later, from January fourth to February fourteenth. Service prediction blends CatBoost and histogram boosting. Late probability blends X G Boost, CatBoost and Light G B M. The displayed results are from that frozen holdout, not the training data."),
    ('Make performance differences visible',
     "Service mean absolute error is three point seven five minutes, and lateness area under the curve is zero point nine seven five. But the brand and depot table matters. Fresh dominates the sample. Tech has larger absolute handling errors, and its segment samples are much smaller. For operational review, one thousand and seventeen of the five thousand and fourteen test orders have predicted late probability of at least fifty percent. These are risks, not observed delays."),
    ('Forecast demand before committing capacity',
     "Demand uses order date and includes every order, even deferred orders and orders that never ran. We build complete weekly series for each depot and brand, and forecast ten weeks from a fixed origin. Candidate models include rolling means, a seasonal baseline, ridge regression and boosting. The submitted network forecast peaks in week fifteen at about two thousand and seven cubic metres, including six hundred and eighty two chilled. This supports inventory and refrigerated availability planning."),
    ('Show the weakness; avoid last-minute overfitting',
     "Overall total demand weighted absolute percentage error is six point zero eight percent. That hides Tech errors of twenty four point seven percent in Kandy and twenty nine point eight percent in Peliyagoda. Several series have negative R squared. We tested nine simple Tech baselines using only earlier chronological folds. The Kandy bias correction improved validation error by less than one percent; the Peliyagoda alternatives were worse. We kept the verified predictions and report these limitations openly."),
    ('A physical bottleneck, then a feasibility proof',
     "The peak day needs one hundred and eighty one point six cubic metres of chilled capacity. Four available refrigerated vehicles can carry at most one hundred and seventy two point four cubic metres across two full volume trips each. That is an optimistic bound: demand already exceeds it by nine point two cubic metres. C P Sat then enforces whole orders, depot, brand and district compatibility, van access, weight, volume and published time budgets. A separate maximum count solve proves the ceiling of seventy nine deliveries."),
    ('Policy changes have a measurable cost',
     "We changed only the priority weights, leaving every hard constraint unchanged. Fresh first and balanced both serve seventy nine orders and deliver one hundred and forty point seven cubic metres chilled. Fairness first serves seventy seven, but clears all ten previously deferred orders instead of nine. That choice sacrifices two deliveries and fourteen point four six nine cubic metres of chilled volume. The comparison uses common reference points, because different policy scores have different scales. All three solutions are optimal and pass the original checker."),
    ('Reproduce the result and state the limits',
     "The final notebook cell reloads the saved models and reproduces both prediction files. Template checks preserve every identifier, column and row order. The allocation checker passes, and all three original submission files remain unchanged. These task horizons are distinct; they are not one jointly solved real world route plan. Optimality covers the published competition constraints, not unmodelled traffic, detailed arrival windows or remaining weekly fuel. The package discloses the A I assistance and requires human team review. Our contribution is a decision that can be explained, checked and defended."),
]


def run(args):
    subprocess.run([str(x) for x in args], check=True, stdout=subprocess.DEVNULL)


def duration(path):
    return float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',str(path)]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='.')
    parser.add_argument('--duration',type=float,default=250)
    args=parser.parse_args()
    root=Path(args.root).resolve(); reports=root/'outputs/reports'
    work=root/'demo_build'; work.mkdir(exist_ok=True)
    nb=nbformat.read(root/'Alt-F4_FinalNotebook.ipynb',as_version=4)
    code=[c for c in nb.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in code), 'Execute the notebook first'
    assert not any(o.output_type=='error' for c in code for o in c.outputs)

    def cell(needle):return next(c for c in code if needle in c.source)
    def python(source):return highlight(source,PythonLexer(),HtmlFormatter())
    def source(name):return '<p class="source">Source: '+html.escape(name)+'</p>'
    def table(df):return df.to_html(index=False,border=0,classes='evidence',float_format=lambda v:f'{v:,.3f}')
    def img(path):
        b=base64.b64encode(Path(path).read_bytes()).decode()
        return '<img class="plot" src="data:image/png;base64,'+b+'">'
    def outputs(c):
        pieces=[]
        for o in c.outputs:
            if o.output_type=='stream':pieces.append('<pre class="stream">'+html.escape(o.text)+'</pre>')
            elif 'text/html' in o.get('data',{}):pieces.append(o.data['text/html'])
        return pieces

    t1=json.loads((reports/'task1_metrics.json').read_text())
    baseline=json.loads((reports/'tech_baseline_experiment.json').read_text())
    policy=pd.read_csv(reports/'policy_comparison.csv')
    services=pd.read_csv(reports/'task1_by_brand_depot.csv')
    service=services.query("target == 'service' and slice == 'depot+brand'")[['depot','brand','n','mae','rmse']]
    policyview=policy[['policy','served_orders','chilled_volume_delivered_m3','balanced_reference_points','deferred_orders','repeat_deferred_orders_served','solver_gap']].copy()
    policyview.columns=['Policy','Served','Chilled m³','Common points','Deferred','Backlog served / 10','Gap']
    modelrows=[]
    for target in ['service','lateness']:
        for model,weight in t1[target]['selection']:
            modelrows.append({'Target':target,'Selected model':model,'Weight':weight})
    label=cell("arrival = w.minutes").source
    label_excerpt='\n'.join(label.splitlines()[3:11])
    final=cell("TASK1 INPUTS")
    final_stream='\n'.join(o.text for o in final.outputs if o.output_type=='stream')
    assert 'Both saved-model outputs reproduce' in final_stream
    proof=json.loads((reports/'task2b_summary.json').read_text())
    baseview=pd.DataFrame(baseline['comparison'])[['depot','reference_model','reference_tuning_rmse','best_new_baseline','baseline_tuning_rmse']]
    baseview.columns=['Depot','Verified model','Earlier RMSE','New baseline','Earlier RMSE']
    # All evidence tables below are views of saved notebook outputs or source reports.
    scenes=[
        '<div class="cover"><div class="big">79 <span>orders served</span></div><div class="big">6 <span>orders deferred</span></div><div class="big">0% <span>solver gap</span></div></div>'
        '<p class="lead">Predict delays → anticipate demand → allocate constrained capacity</p>'
        +img(reports/'capacity_vs_demand.png')+source('task2b_summary.json · policy_sensitivity.json'),
        '<h2>Actual label-construction cell</h2>'+python(label_excerpt)+
        '<div class="note">Illustrative early arrival: 08:40 → window opens 09:00 → leave 09:12<br><b>12 minutes service · 20 minutes waiting</b></div>'+
        source('Alt-F4_FinalNotebook.ipynb · executed label-construction cell'),
        '<div class="twocol"><section><h2>Chronological protocol</h2>'+table(pd.DataFrame(t1['tuning_windows'],columns=['Tuning start','Tuning end']))+
        '<div class="note">Frozen final holdout<br><b>4 Jan – 14 Feb 2026</b></div>'+python(cell('if RUN_TRAINING:\n    task1_results').source.split('display(')[0])+ '</section><section><h2>Selected model architecture</h2>'+table(pd.DataFrame(modelrows))+'</section></div>'+source('Task1 training/evaluation cell · task1_metrics.json'),
        '<div class="metrics"><b>3.752 min</b> service MAE <b>0.9753</b> late AUC <b>94.50%</b> accuracy at 0.5</div>'+
        '<h2>Final holdout: service performance by depot and brand</h2>'+table(service)+
        '<p class="note">5,035 held-out orders; unequal segment sizes are shown explicitly.<br>Test horizon: 1,017 / 5,014 predicted late risks ≥ 0.5, not observed late outcomes.</p>'+source('task1_by_brand_depot.csv · task1_metrics.json · decision_evidence.json'),
        img(reports/'forecast_and_errors.png')+
        '<p class="note">Order-date demand includes deferred and not-run orders. Ten future weeks use only information available at each forecast origin.</p>'+source('Executed forecasting cell · submission_task2a.csv · task2a_by_brand_depot.csv'),
        '<h2>Earlier validation only · final holdout excluded</h2>'+table(baseview)+
        '<div class="note">Tuning origins: 1 Sep and 10 Nov 2025<br>Excluded final holdout begins: <b>19 Jan 2026</b></div>'+
        '<p class="lead">Kandy: 0.90% RMSE improvement. Peliyagoda: 13.24% worse.<br><b>Retain verified forecasts; report the weakness.</b></p>'+source('Executed baseline-experiment cell · tech_baseline_experiment.json'),
        img(reports/'capacity_vs_demand.png')+
        '<div class="twocol"><section><h2>Same published hard rules</h2><p>Whole orders · matching depot · one brand and district per trip<br>Refrigeration · van access · weight and volume<br>At most two trips · Fresh 270 min · Style/Tech 480 min</p></section>'+
        '<section><h2>Actual maximum-count solver result</h2><pre class="stream">'+html.escape(json.dumps({k:proof['max_count_comparison'][k] for k in ['solver_status','objective_value','objective_bound','served','deferred','relative_gap']},indent=2))+'</pre></section></div>'+source('Executed allocation cell · task2b_summary.json'),
        '<h2>Only priority weights changed</h2>'+table(policyview)+
        '<div class="note"><b>Fairness clears 10/10 repeat-deferred orders</b><br>Cost versus balanced: 2 fewer deliveries and 14.469 m³ less chilled volume.</div>'+
        '<h2>Real changed order decisions</h2>'+table(pd.read_csv(reports/'policy_changed_orders.csv'))+source('Executed policy cell · policy_comparison.csv · policy_changed_orders.csv'),
        '<h2>Final notebook cell: reload and verify saved models</h2>'+python('\n'.join(final.source.splitlines()[-6:]))+
        '<pre class="stream success">'+html.escape(final_stream)+'</pre>'+
        '<div class="note">Official allocation checker: <b>PASSED</b><br>All three verified submission hashes: <b>UNCHANGED</b></div>'+
        '<p>Limits: distinct task horizons, weak Tech series, competition-model feasibility, human review still required.</p>'+source('Final inference cell · official_checker.txt · submission_freeze.json'),
    ]
    css='''@page {size:1920px 1080px;margin:0} *{box-sizing:border-box}
      body{margin:0;height:1080px;overflow:hidden;background:#eef3f6;color:#183044;font:23px "DejaVu Sans",sans-serif}
      header{position:absolute;top:0;left:0;width:1920px;height:152px;padding:28px 66px;background:#142c3d;color:white}
      header small{font-size:18px;color:#84d7cb} h1{font-size:39px;margin:13px 0 0}
      main{position:absolute;top:152px;left:0;width:1920px;height:852px;padding:28px 66px 20px;background:white;overflow:hidden}
      h2{font-size:27px;margin:10px 0 16px}p{line-height:1.5}
      footer{position:absolute;top:1004px;left:0;width:1920px;height:76px;padding:20px 66px;color:#536576;font-size:18px}
      .source{font-size:17px;color:#667888;margin:17px 0}
      table{border-collapse:collapse;font-size:22px;width:100%;margin:18px 0}
      th{background:#eaf2f5;text-align:left;font-size:19px;padding:11px 12px}
      td{border-bottom:1px solid #d6e1e8;padding:10px 12px;text-align:left}
      th,td{white-space:normal}tr:nth-child(even){background:#f8fafb}
      .note{background:#e8f5f2;border-left:6px solid #00877e;padding:18px 24px;margin:18px 0;line-height:1.6}
      .lead{font-size:29px;line-height:1.5}.plot{display:block;width:100%;max-height:605px;object-fit:contain}
      main.capacity .plot{height:455px;max-height:455px}
      pre{font:21px/1.45 "DejaVu Sans Mono",monospace;white-space:pre-wrap;margin:12px 0}
      .highlight{background:#f4f7fa;padding:18px 26px;border:1px solid #d6e1e8}
      .stream{background:#f4f7fa;padding:18px 24px}.success{border-left:6px solid #00877e}
      .twocol{display:flex;gap:36px}.twocol section{width:50%}
      .cover{display:flex;gap:85px;padding:10px 0}.big{font-size:65px;font-weight:700;color:#007f7b}
      .big span{font-size:25px;color:#183044;font-weight:400}.metrics{padding:15px 0;font-size:25px}.metrics b{font-size:36px;color:#007f7b;margin-left:30px}
    '''+HtmlFormatter().get_style_defs('.highlight')
    rendered=[]
    for i,((title,narration),scene) in enumerate(zip(NARRATION,scenes)):
        stem=work/f'{i+1:02d}'
        tag='<main class="capacity">' if i==6 else '<main>'
        doc='<html><head><meta charset="utf-8"><style>'+css+'</style></head><body><header><small>ALT-F4 · Alt-F4_FinalNotebook.ipynb · executed evidence</small><h1>'+html.escape(title)+'</h1></header>'+tag+scene+'</main><footer>Chapter '+str(i+1)+' / 9 · Narrated replay of executed outputs · Synthetic narration · Team review required</footer></body></html>'
        stem.with_suffix('.html').write_text(doc)
        HTML(string=doc,base_url=str(root)).write_pdf(stem.with_suffix('.pdf'))
        pdf=fitz.open(stem.with_suffix('.pdf'))
        assert len(pdf)==1,(i,len(pdf))
        pdf[0].get_pixmap(matrix=fitz.Matrix(4/3,4/3),alpha=False).save(stem.with_suffix('.png'))
        stem.with_suffix('.txt').write_text(narration)
        run(['ffmpeg','-y','-loglevel','error','-f','lavfi','-i',f'flite=textfile={stem}.txt:voice=slt','-ar','48000',stem.with_suffix('.wav')])
        rendered.append(stem)
        print('Rendered actual-evidence chapter',i+1,flush=True)
    raw=sum(duration(p.with_suffix('.wav')) for p in rendered)
    pad=1.5
    rate=raw/(args.duration-pad*len(rendered))
    assert .5<=rate<=2
    chapters=[]; cursor=0
    for i,p in enumerate(rendered):
        audio=p.with_suffix('.adjusted.wav')
        run(['ffmpeg','-y','-loglevel','error','-i',p.with_suffix('.wav'),'-af',f'atempo={rate:.8f},apad=pad_dur={pad}',audio])
        seconds=duration(audio)
        run(['ffmpeg','-y','-loglevel','error','-loop','1','-framerate','24','-i',p.with_suffix('.png'),'-i',audio,
             '-t',f'{seconds:.6f}','-c:v','libx264','-preset','veryfast','-tune','stillimage','-crf','21','-threads','3',
             '-pix_fmt','yuv420p','-c:a','aac','-b:a','160k','-ar','48000',p.with_suffix('.mp4')])
        chapters.append({'title':NARRATION[i][0],'start':cursor,'duration':seconds,'narration':NARRATION[i][1]})
        cursor+=seconds
        print('Encoded chapter',i+1,'duration',round(seconds,1),flush=True)
    concat=work/'concat.txt'
    concat.write_text('\n'.join("file '"+str(p.with_suffix('.mp4'))+"'" for p in rendered))
    rawmp4=work/'replay.mp4'
    run(['ffmpeg','-y','-loglevel','error','-f','concat','-safe','0','-i',concat,'-c','copy',rawmp4])
    def stamp(t):
        n=int(round(t*1000));return f'{n//3600000:02d}:{n//60000%60:02d}:{n//1000%60:02d},{n%1000:03d}'
    captions=[]
    for chapter in chapters:
        sentences=re.split(r'(?<=[.!?])\s+',chapter['narration'])
        words=sum(len(s.split()) for s in sentences); pos=chapter['start']
        spoken=chapter['duration']-pad
        for s in sentences:
            end=pos+spoken*len(s.split())/words
            captions.append(f'{len(captions)+1}\n{stamp(pos)} --> {stamp(end)}\n{s}\n')
            pos=end
    srt=root/'Alt-F4_Demo.srt';srt.write_text('\n'.join(captions))
    output=root/'Alt-F4_Demo_Review.mp4'
    run(['ffmpeg','-y','-loglevel','error','-i',rawmp4,'-i',srt,'-c:v','copy','-c:a','copy','-c:s','mov_text',
         '-metadata','title=Alt-F4 | Explainable Logistics Decision Intelligence',
         '-metadata','comment=Narrated replay of executed notebook outputs; synthetic narration; team review required.',
         '-metadata:s:s:0','language=eng','-movflags','+faststart',output])
    (root/'DEMO_CHAPTERS.json').write_text(json.dumps({'duration_seconds':duration(output),'synthetic_narration':True,
        'visuals':'Rendered excerpts of the executed notebook and actual reports; not a live screen recording.',
        'notebook_sha256':hashlib.sha256((root/'Alt-F4_FinalNotebook.ipynb').read_bytes()).hexdigest(),
        'chapters':chapters},indent=2))
    outline='# Alt-F4 demo storyline\n\nThe MP4 is a review draft: synthetic narration over a replay of real executed notebook outputs, not a live screen recording. '
    outline+='The team must understand and review it. For a human recording, open `Alt-F4_FinalNotebook.ipynb` or its HTML export and follow this script. '
    outline+='Show the specified real cells and reports; do not substitute fabricated results.\n\n'
    for c in chapters:
        outline+=f"## {stamp(c['start'])[:8]} — {c['title']}\n\n{c['narration']}\n\n"
    outline+='## Finish before submission\n\nReview the disclosure and limitations; check the final inference cell and official checker result. Upload the reviewed 3–5 minute video to YouTube as unlisted and provide its actual link in the competition form. No video has been uploaded or submission made by this code. Do not publish organizer datasets.\n'
    (root/'DEMO_OUTLINE.md').write_text(outline)
    print('Created',output.name,'seconds:',round(duration(output),2),flush=True)


if __name__=='__main__':main()
