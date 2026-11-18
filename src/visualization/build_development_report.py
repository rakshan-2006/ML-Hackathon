"""Measured preparation evidence PDF; intentionally NOT the final team report."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
from src.common.paths import PROJECT_ROOT


def build():
    root=PROJECT_ROOT
    table=pd.read_csv(root/'results/rehearsal/student_test_api.csv',dtype={'trip_id':str})
    summary=json.loads((root/'results/rehearsal/summary.json').read_text())
    profile=json.loads((root/'results/rehearsal/profile/profile.json').read_text())
    if summary['status']!='development_baseline_rehearsal':
        raise ValueError('This builder is for the clearly labelled development evidence only')
    figures=root/'results/hmm/development'; figures.mkdir(parents=True,exist_ok=True)
    x=np.arange(len(table)); fig,ax=plt.subplots(figsize=(10,3.6))
    ax.bar(x-.18,table.top1_accuracy_classifier*100,.36,label='Classifier')
    ax.bar(x+.18,table.top1_accuracy_hmm*100,.36,label='Classifier + untuned HMM')
    ax.set(xticks=x,xticklabels=table.trip_id,ylabel='Trusted derived-alignment Top-1 (%)',ylim=(0,105),
           title='Student-test development baseline: all ten trips')
    ax.tick_params(axis='x',labelrotation=45); ax.legend(loc='lower right'); fig.tight_layout()
    fig.savefig(figures/'top1_by_trip.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,3.2))
    ax.bar(x,table.total_ms/1000,label='Server inference + evaluation')
    ax.set(xticks=x,xticklabels=table.trip_id,ylabel='Seconds',title='Measured local HTTP service processing')
    ax.tick_params(axis='x',labelrotation=45); fig.tight_layout()
    fig.savefig(figures/'api_latency.png',dpi=160); plt.close(fig)
    destination=root/'output/pdf/pathsnap-development-evidence.pdf'; destination.parent.mkdir(parents=True,exist_ok=True)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='SmallB',fontName='Helvetica',fontSize=9,leading=12,spaceAfter=7))
    styles.add(ParagraphStyle(name='DraftB',fontName='Helvetica-Bold',fontSize=11,leading=15,textColor=colors.HexColor('#a04b00'),spaceAfter=12))
    story=[]
    def p(text,style='SmallB'): story.append(Paragraph(text,styles[style]))
    def title(text): p(text,'Heading1')
    def image(path,width=480,height=None):
        value=Image(str(path)); ratio=value.imageHeight/value.imageWidth
        value.drawWidth=width; value.drawHeight=height or width*ratio; story.append(value); story.append(Spacer(1,8))
    def grid(rows,widths):
        t=Table(rows,colWidths=widths,repeatRows=1)
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#16344c')),
            ('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
            ('FONTSIZE',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),6),
            ('TOPPADDING',(0,0),(-1,-1),6),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.whitesmoke,colors.white]),
            ('VALIGN',(0,0),(-1,-1),'TOP')]))
        story.append(t); story.append(Spacer(1,10))
    title('PathSnap | Working integration evidence')
    p('DEVELOPMENT DRAFT - not the final submission report','DraftB')
    p('Prepared 18 November 2026. Person B implementation on a separate worktree; main integrated through 55e90cb. '
      'Evaluator archive remains sealed. Final selected model, OOF decoder selection and physical two-laptop acceptance are pending.')
    p('5. HMM and Viterbi reconstruction','Heading2')
    p('Hidden states are directed candidate arcs, retaining raw observation and candidate identity. '
      'Emission evidence is weight * log(clipped classifier positive-class probability). This discriminative posterior is an '
      'approximation to an HMM emission likelihood. Transitions penalize absolute network/GPS distance mismatch divided by beta, '
      'subject to directed reachability, elapsed-time speed bounds and same-arc jitter tolerance.')
    p('The recurrence is D[t,j] = emission[t,j] + max_i(D[t-1,i] + transition[t,i,j]), with deterministic ties and backtracking. '
      'Empty/invalid rows keep prediction -1. Large/nonpositive timestamp changes and impossible transitions restart segments. '
      'Inserted path arcs are marked inferred and never become observation predictions; geometry does not bridge disconnected segments.')
    p('Fixed development configuration: beta 10 m, emission weight 1, gap split 300 s, speed bound 50 m/s, slack 50 m. '
      'No final decoder tuning has been run. Scorer: verified release GB, model version hgb-baseline-ac522dfa.')
    a=summary['aggregate']
    p(f"All ten permitted student-test trips matched offline geometry and metrics through genuine HTTP calls. Weighted trusted Top-1: "
      f"classifier {a['top1_classifier']:.4%}; HMM {a['top1_hmm']:.4%}. These are development measurements, not official evaluator scores.")
    rows=[['Trip','Raw obs','Classifier Top-1','HMM Top-1','Server seconds']]
    for row in table.itertuples():
        rows.append([row.trip_id,str(row.n_obs),f'{row.top1_accuracy_classifier:.2%}',f'{row.top1_accuracy_hmm:.2%}',f'{row.total_ms/1000:.2f}'])
    grid(rows,[80,65,110,110,115])
    p('Failure example: trip 00000007 drops from 91.82% to 88.15%. Connectivity evidence can outweigh a correct pointwise choice; '
      'the untuned decoder does not improve every trip. No predictions were manually repaired.')
    story.append(PageBreak())
    title('Measured comparison and geographical view')
    image(figures/'top1_by_trip.png')
    p('Metric policy: trusted valid derived reference alignment only; absent predictions/candidate misses count wrong. '
      'Weighted overall Top-1 sums correct/evaluated counts. Ambiguous/unaligned labels are excluded with coverage reported. '
      'Point-sequence continuity uses common gap boundaries and excludes missing pairs. Assembled connectivity is separate because '
      'inferred arcs and decoder restarts can improve it mechanically. Arc set F1 ignores order/multiplicity; exact LCS F1 is order-aware. '
      'Undefined metrics serialize as null. Full sequence/continuity/count/timing evidence is in the per-trip CSV.')
    p('6. Folium visualisation','Heading2')
    image(root/'docs/evidence/development_map.jpg',width=480)
    p('Actual student-test trip 00000052: GPS, sampled network, approximate reference, classifier and HMM layers. '
      'Three HTML maps were selected by shortest/median/longest trajectory before observing scores. '
      'This view displays all 262 GPS points and 2,000 of 3,577 background arcs; inference uses every observation/candidate. '
      'Basemap credit: OpenStreetMap contributors; Leaflet. CDN/Internet access is required for the interactive basemap.')
    story.append(PageBreak())
    title('7. API, UI and performance')
    image(root/'docs/evidence/local_ui.jpg',width=480)
    p('Real local Streamlit UI calls FastAPI only. The development warning and versions are visible. Prediction results persist across '
      'component reruns. References stay server-side; evaluation happens only after route-free prediction. The graph cache holds at most '
      'one graph; requests are serialized to bound memory. UI timeout defaults to 300 s and is configurable.')
    p('Extreme-case pipeline profiling','Heading2')
    rows=[['Selection','Trip','Cold / warm sec','Peak process MiB']]
    for r in profile['results']:
        rows.append([r['criterion'],r['trip_id'],f"{r['cold_graph']['wall_ms']/1000:.2f} / {r['warm_graph']['wall_ms']/1000:.2f}",
                     f"{r['peak_process_working_set_bytes']/1024**2:.1f}"])
    grid(rows,[100,100,140,140])
    p('Profile hardware: Windows, Intel64 family 6 model 154, 12 logical CPUs; Python 3.11.9; OMP_NUM_THREADS=4. '
      'Each case used a fresh process; memory is the OS lifetime peak resident working set, not Python-only allocation. '
      'Cold means application graph uncached, not flushed OS disk cache. Warm reuses graph/model but recomputes features and decoding. '
      'These profile times exclude optional reference evaluation/HTTP. Full HTTP evidence includes both. '
      'The longest track has 19,313 observations; largest network has 307,392 arcs.')
    p('The repeated ten-trip API processing ranged from 0.64 to 12.10 seconds under concurrent local checks. '
      'Real server log records are preserved with request IDs. Local loopback is verified; physical Laptop 1/Laptop 2 '
      'networking, receiving-console screenshots and fallback-hotspot checks are still pending. No local screenshot is presented as that proof.')
    story.append(PageBreak())
    title('Consolidated report assembly and final gates')
    p('Required structure from Deliverables.pdf, pp.2-7','Heading2')
    p('2.2 Dataset/EDA: integrate A\'s measured missingness, features/justification, projection, GPS/network plots and partition-labelled data analysis. '
      'B adds actual Folium map evidence. The supplied route is reference information with derived observation alignment, not verified physical ground truth.')
    p('3.1-3.3 Model experimentation: integrate a measured table for all five models, every model\'s task-specific configuration/loss, '
      'strengths/limits, held-out Top-1/macro-F1/log loss, over/underfit evidence and honest comparison figures. '
      'KNN implementation on main does not establish completed training. B has not fabricated missing model experiments.')
    p('4 Bayesian optimisation: A must supply actual objective/search distributions, trial record, final parameters, cost and '
      'tuned/untuned comparable results. Decoder grid selection is a separate experiment and cannot substitute for Bayesian classifier search.')
    p('5 HMM: the preceding development measurements are a working contribution. Final results await A\'s selected split model, '
      'Predictor and complete final-parameter OOF scores. B\'s selection command verifies exact candidate keys, all canonical folds, '
      'version/config/table hashes and declared fold holdout provenance before selecting a decoder using training OOF only. '
      'All trial configurations and per-trip results must then be retained; student-test confirms the final configuration.')
    p('6 Folium: replace or supplement development maps with final-model maps and preserve the imperfect/harmful cases. '
      '7 Deployment: repeat all ten HTTP/offline checks with final artifacts, then complete the physical two-laptop run and console/UI evidence. '
      'Publish the checksummed final bundle and verify a fresh download/startup. The final-only startup gate rejects development bundles.')
    p('8 Integrity and assistance','Heading2')
    p('Codex assisted implementation, code review, test execution, evidence/chart generation and this draft prose. '
      'The team must review, understand and defend the code and measured claims, and follow the organizer\'s disclosure format. '
      'No human approval, completed final experiment, official evaluation or physical deployment has been invented.')
    p('Source traceability','Heading2')
    p('All numbers: results/rehearsal/student_test_api.csv, summary.json and profile/profile.json. '
      'Logs: results/rehearsal/server_inference.jsonl. Figures: results/hmm/development. '
      'Map/UI captures: docs/evidence. Interactive maps: maps/development. '
      'Editable PDF builder: src/visualization/build_development_report.py. '
      'Full runbook and acceptance matrix: docs/person_b_runbook.md and docs/reports/person_b_acceptance.md.')
    p('This working PDF is not a final accepted team deliverable. Exactly one reviewed consolidated final PDF remains to be assembled '
      'after the dependencies above. The real evaluator ZIP remains sealed throughout preparation. Any later import/inference needs '
      'explicit official-stage authorization, and evaluator outputs must never be used for tuning.','DraftB')
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8); canvas.setFillColor(colors.HexColor('#536577'))
        canvas.drawString(42,25,'PATHSNAP | DEVELOPMENT EVIDENCE | evaluator archive sealed')
        canvas.drawRightString(A4[0]-42,25,str(doc.page))
    SimpleDocTemplate(str(destination),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=38,bottomMargin=42,
        title='PathSnap development integration evidence - draft',author='PathSnap team, AI-assisted working draft').build(story,onFirstPage=footer,onLaterPages=footer)
    print(destination)
    return destination


if __name__=='__main__': build()
