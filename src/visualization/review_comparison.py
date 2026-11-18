"""Generate evidence-backed review figures; run python -m src.visualization.review_comparison."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.common.paths import PROJECT_ROOT

BLUE, AMBER, GREEN = '#2563a6', '#d58b1f', '#228154'


def cv_f1(card):
    values = card.get('results', {}).get('cv_mean') or card['cv_fold_mean']
    return float(values['f1_macro'] if 'f1_macro' in values else values['macro_f1'])


def save(fig, folder, name):
    for extension in ['png', 'svg']:
        fig.savefig(folder / f'{name}.{extension}', dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    root = PROJECT_ROOT
    out = root / 'results/figures/review'
    out.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(root/'results/models/gradient_boosting/student_test_metrics.csv').iloc[0]
    tuned = pd.read_csv(root/'results/models/gradient_boosting_tuned/student_test_metrics.csv').iloc[0]
    bcard = json.loads((root/'results/models/gradient_boosting/model_card.json').read_text())
    tcard = json.loads((root/'results/models/gradient_boosting_tuned/model_card.json').read_text())
    hmm = json.loads((root/'results/rehearsal/summary.json').read_text())
    trips = pd.read_csv(root/'results/rehearsal/student_test_api.csv', dtype={'trip_id': str})
    assert len(trips) == trips.trip_id.nunique() == hmm['trip_count'] == 10
    assert int(base.n_trusted_obs) == int(tuned.n_trusted_obs) == int(trips.n_evaluated_hmm.sum()) == 16073
    assert np.isclose(base.top1_accuracy, hmm['aggregate']['top1_classifier'], rtol=0, atol=1e-12)
    after = trips.n_correct_hmm.sum()/trips.n_evaluated_hmm.sum()
    before = trips.n_correct_classifier.sum()/trips.n_evaluated_classifier.sum()
    assert np.isclose(after, hmm['aggregate']['top1_hmm'], rtol=0, atol=1e-12)
    assert np.isclose(before, base.top1_accuracy, rtol=0, atol=1e-12)
    gain = 100*(after-before)
    bcv, tcv = cv_f1(bcard), cv_f1(tcard)
    stages = pd.DataFrame([
        dict(stage='Before tuning: baseline GB (selected)', top1_accuracy=float(before),
             correct=int(base.n_top1_correct), denominator=16073, scope='student_test'),
        dict(stage='After Bayesian tuning: trial-winning GB (not selected)', top1_accuracy=float(tuned.top1_accuracy),
             correct=int(tuned.n_top1_correct), denominator=16073, scope='student_test'),
        dict(stage='Selected baseline GB + fixed untuned HMM', top1_accuracy=float(after),
             correct=int(trips.n_correct_hmm.sum()), denominator=16073, scope='student_test'),
    ])
    stages.to_csv(out/'stage_top1.csv', index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})

    fig, ax = plt.subplots(figsize=(10.8, 6.4))
    values = stages.top1_accuracy.to_numpy()*100
    bars = ax.bar(range(3), values, color=[BLUE,AMBER,GREEN], width=.58)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, value+1.1, f'{value:.2f}%', ha='center', fontsize=16, weight='bold')
    ax.set_xticks(range(3), ['Before tuning\nBaseline GB — selected',
                           'After Bayesian tuning\nGB trial winner — not selected',
                           'After HMM\nSelected baseline GB + HMM'])
    ax.set_ylim(0,108); ax.set_yticks(range(0,101,20)); ax.set_ylabel('GPS-to-road Top-1 accuracy (%)')
    ax.set_axisbelow(True); ax.grid(axis='y', alpha=.18)
    ax.text(.5,.965,f'HMM gain over selected baseline: +{gain:.2f} percentage points',
            transform=ax.transAxes,ha='center',va='top',fontsize=12,color=GREEN,weight='bold')
    fig.suptitle('PathSnap: baseline, Bayesian tuning and HMM', fontsize=17, weight='bold', y=.98)
    fig.text(.5,.025,'Same metric: 10 student-test trips, 16,073 trusted GPS observations.\n'
             'HMM uses the selected baseline GB, not the tuned trial. Decoder defaults are fixed and untuned.',
             ha='center',fontsize=10,color='#444444')
    fig.subplots_adjust(top=.87,bottom=.21)
    save(fig,out,'01_baseline_tuned_hmm_top1')

    panels = [
        ('Mean five-fold CV macro-F1\nSelection objective — higher is better', [bcv*100,tcv*100], '%',5),
        ('Student-test GPS Top-1\nHigher is better', [base.top1_accuracy*100,tuned.top1_accuracy*100], '%',2),
        ('Student-test macro-F1\nHigher is better', [base.f1_macro*100,tuned.f1_macro*100], '%',4),
        ('Student-test log loss\nLower is better', [base.log_loss,tuned.log_loss], '',5),
    ]
    fig, axes = plt.subplots(2,2,figsize=(11,8))
    for ax,(title,vals,suffix,decimals) in zip(axes.flat,panels):
        bars=ax.bar(['Baseline GB','Tuned GB'],vals,color=[BLUE,AMBER],width=.55)
        upper=105 if suffix else max(vals)*1.35
        ax.set_ylim(0,upper); ax.set_title(title,fontsize=12,pad=12)
        ax.grid(axis='y',alpha=.16); ax.set_axisbelow(True)
        for bar,val in zip(bars,vals):
            ax.text(bar.get_x()+bar.get_width()/2,val+upper*.025,
                    f'{val:.{decimals}f}{suffix}',ha='center',fontsize=11,weight='bold')
    fig.suptitle('Bayesian tuning: small trade-offs, baseline retained',fontsize=16,weight='bold')
    fig.text(.5,.025,f'Tuned minus baseline CV macro-F1: {(tcv-bcv)*100:+.6f} percentage points.\n'
             'Five attempted trials (3 startup, 2 adaptive); 4 completed, 1 pruned. Selection used training CV, not test scores.',
             ha='center',fontsize=10,color='#444444')
    fig.tight_layout(rect=[0,.08,1,.95])
    save(fig,out,'02_before_after_tuning')

    fig,ax=plt.subplots(figsize=(14,6.5))
    x=np.arange(len(trips)); width=.36
    first=trips.top1_accuracy_classifier.to_numpy()*100
    second=trips.top1_accuracy_hmm.to_numpy()*100
    for offset,vals,color,label in [(-width/2,first,BLUE,'Selected GB alone'),(width/2,second,GREEN,'Selected GB + HMM')]:
        bars=ax.bar(x+offset,vals,width=width,color=color,label=label)
        for bar,val in zip(bars,vals):
            ax.text(bar.get_x()+bar.get_width()/2,val+1,f'{val:.1f}',ha='center',fontsize=8)
    ax.set_xticks(x,trips.trip_id,rotation=35,ha='right')
    ax.set_ylim(0,110); ax.set_yticks(range(0,101,20)); ax.set_ylabel('GPS-to-road Top-1 accuracy (%)')
    ax.set_xlabel('Student-test trip ID');ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.1),ncol=2,frameon=False)
    fig.suptitle('HMM before/after: per-trip results',fontsize=17,weight='bold',y=.99)
    fig.text(.5,.015,f'Observation-weighted aggregate: {before:.2%} → {after:.2%} (+{gain:.2f} pp). '
             'This is not the mean of trip percentages.\nTrip 00000007 worsens; HMM does not improve every trip. Fixed untuned decoder defaults.',ha='center',fontsize=10)
    fig.subplots_adjust(top=.82,bottom=.24)
    save(fig,out,'03_hmm_per_trip')

    provenance=dict(student_test_trips=10,trusted_observations=16073,
                    baseline_gps_top1=float(before),tuned_gps_top1=float(tuned.top1_accuracy),
                    selected_baseline_plus_hmm_gps_top1=float(after),hmm_gain_percentage_points=float(gain),
                    baseline_cv_macro_f1=bcv,tuned_cv_macro_f1=tcv,
                    selected_model='baseline gradient_boosting',hmm_applied_to='baseline gradient_boosting',
                    decoder_status='development-untuned-v1',
                    sources=['results/models/gradient_boosting/student_test_metrics.csv',
                             'results/models/gradient_boosting_tuned/student_test_metrics.csv',
                             'results/models/gradient_boosting/model_card.json',
                             'results/models/gradient_boosting_tuned/model_card.json',
                             'results/rehearsal/summary.json','results/rehearsal/student_test_api.csv'])
    (out/'figure_data.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (out/'README.md').write_text(f'''# Review figures: baseline, tuning and HMM

Generated with `python -m src.visualization.review_comparison`. PNG and SVG versions are supplied.

1. [Three-stage GPS Top-1 comparison](01_baseline_tuned_hmm_top1.png): baseline GB **{before:.2%}**, tuned GB **{tuned.top1_accuracy:.2%}**, selected baseline GB + HMM **{after:.2%}**.
2. [Before/after tuning](02_before_after_tuning.png): training-CV selection objective alongside student-test Top-1, macro-F1 and log loss.
3. [HMM per trip](03_hmm_per_trip.png): paired classifier/HMM scores across all ten student-test trips.

**Interpretation:** The tuned trial slightly improves test Top-1 but does not beat baseline training-CV macro-F1, so the baseline remains selected. HMM was applied to that selected baseline, not to the tuned trial. Its gain is **+{gain:.2f} percentage points over baseline**. No tuned-model-plus-HMM result is claimed.

All Top-1 scores use 16,073 trusted GPS observations; candidate misses count as failures. The aggregate HMM score is weighted by observations. Fixed untuned decoder defaults are used, and trip 00000007 worsens.

Evidence: [baseline](../../models/gradient_boosting/student_test_metrics.csv), [tuned](../../models/gradient_boosting_tuned/student_test_metrics.csv), [HMM summary](../../rehearsal/summary.json), [per-trip data](../../rehearsal/student_test_api.csv). Exact plotting values are in [figure_data.json](figure_data.json) and [stage_top1.csv](stage_top1.csv).
''',encoding='utf-8')
    print(json.dumps(provenance,indent=2))
    print(f'Figures saved to {out}')


if __name__=='__main__':
    main()
