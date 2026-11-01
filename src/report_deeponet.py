"""Generate the human-readable report from measured results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
FIELDS=('u','w','p','k','omega','nut')

def main():
    s=json.loads((ROOT/'outputs/deeponet_sst/evaluation/summary.json').read_text())
    cv=json.loads((ROOT/'outputs/deeponet_cv/summary.json').read_text())
    hist=pd.read_csv(ROOT/'outputs/deeponet_sst/loss_history.csv')
    lines=['# DeepONet: measured results for the supplied NACA0012 project','',
           'These results were computed from the supplied analysis archive. Lower error is better. Percentages are relative L2 errors, not classification accuracy. Read DEEPONET_README.md for architecture, equations, commands and limitations.','',
           '## Training-only five-fold cross-validation','',
           'Both models use fixed settings. POD and scalers are rebuilt on each fold\'s 32 training cases. These are aggregate out-of-fold errors across 40 cases; they are not means of percentages. Physics fine-tuning is not included in this CV.','',
           '| Field | MLP 32–32 tanh | DeepONet data | Relative error reduction |',
           '|---|---:|---:|---:|']
    for f in FIELDS:
        a=cv['out_of_fold_relative_l2']['mlp']['all'][f]; b=cv['out_of_fold_relative_l2']['deeponet']['all'][f]
        lines.append(f'| {f} | {a:.3%} | {b:.3%} | {(a-b)/a:.1%} |')
    fold=pd.read_csv(ROOT/'outputs/deeponet_cv/fold_errors.csv')
    lines+=['','Fold-to-fold standard deviation of global relative L2 errors (percentage points; five folds):','',
            '| Model | u | w | p | k | omega | nut |','|---|---:|---:|---:|---:|---:|---:|']
    for model in ['mlp','deeponet']:
        rows=fold[(fold.model==model)&(fold.region=='all')]
        values=[rows[rows.field==f].relative_l2.std(ddof=1)*100 for f in FIELDS]
        lines.append('| '+model+' | '+' | '.join(f'{v:.3f}' for v in values)+' |')
    models=['MLP32-refit-data','Saved-MLP32-SST','Saved-MLP64-SST','DeepONet-data','DeepONet-SST','POD-projection']
    for split,title in [('validation','Seven existing validation cases'),('additional_test','Seven additional archived CFD cases')]:
        lines+=['',f'## {title}','',
                '| Model | u | w | p | k | omega | nut |','|---|---:|---:|---:|---:|---:|---:|']
        group=s['datasets'][split]
        for model in models:
            values=[group[model]['regions']['all'][f]['relative_l2'] for f in FIELDS]
            lines.append('| '+model+' | '+' | '.join(f'{v:.3%}' for v in values)+' |')
        lines+=['','Coefficient and physics metrics:','',
                '| Model | Coefficient standardized RMSE | Coefficient relative L2 | Coefficient R2 | Normalized physics score |',
                '|---|---:|---:|---:|---:|']
        for model in models+['CFD']:
            r=group[model]
            lines.append(f"| {model} | {r['coefficient_standardized_rmse']:.5f} | {r['coefficient_relative_l2']:.3%} | {r['coefficient_r2_uniform']:.5f} | {r['normalized_physics_score']:.5g} |")
    lines+=['','## Comparison with the original three-field POD-MLP','',
            'All methods below use the original dimensional Cartesian validation targets and grid. The original model has four modes and three fields; DeepONet uses 16 modes and six fields, so this comparison includes representation differences.','',
            '| Model | u | w | p |','|---|---:|---:|---:|']
    for model,metrics in s['original_grid_validation'].items():
        lines.append('| '+model+' | '+' | '.join(f'{metrics[f]:.3%}' for f in FIELDS[:3])+' |')
    lines+=['','## Wall pressure and runtime','',
            '| Model | Validation Cp RMSE |','|---|---:|']
    for model in models:
        lines.append(f"| {model} | {s['datasets']['validation'][model]['cp_rmse']:.6f} |")
    for model,timing in s['resident_cpu_inference'].items():
        lines+=['',f"{model}: median resident CPU coefficient prediction plus six-field reconstruction {timing['median_seconds']*1000:.3f} ms; p95 {timing['p95_seconds']*1000:.3f} ms. File loading and CFD runtime are not included."]
    first,last=hist.iloc[0],hist.iloc[-1]
    lines+=['','## Did physics fine-tuning help?','',
            f"The mean training physics loss changed from {first.physics:.6f} after the first update to {last.physics:.6f} after the last update. Data loss changed from {first.data:.8f} to {last.data:.8f}; wall loss from {first.wall:.8f} to {last.wall:.8f}.",'',
            'The difference between DeepONet-data and DeepONet-SST includes field and wall supervision as well as PDE residuals; it is not a pure residual-only ablation. Judge both CFD error and residual consistency. A lower training residual does not establish better held-out accuracy.','']
    for split in ['validation','additional_test']:
        data=s['datasets'][split]['DeepONet-data']; phys=s['datasets'][split]['DeepONet-SST']
        wins=[f for f in FIELDS if phys['regions']['all'][f]['relative_l2']<data['regions']['all'][f]['relative_l2']]
        lines.append(f"On {split}, SST fine-tuning reduces field error for {', '.join(wins) if wins else 'none of the six fields'}. Its normalized physics score is {phys['normalized_physics_score']:.5f}, versus {data['normalized_physics_score']:.5f} for data-only.")
    lines+=['','## Physics generalization failure','',
            'The very large held-out physics scores are real diagnostics from the inherited residual code, not percentages. The omega transport term dominates. DeepONet-SST predicts negative omega at about 0.107% of validation points and 0.720% of additional-test points. The residual code clamps omega to 1e-8 internally; the cross-diffusion term divides by omega, making it extremely sensitive in these regions. Clamping inside the residual does not enforce positive predicted fields.',
            '',
            'This model is therefore not established as a physically reliable replacement. A next study should evaluate a positivity-preserving turbulence representation, additional training-only collocation conditions, and an independently held-out test set. Those changes would alter the representation/training protocol and were not tuned on these evaluation cases.',
            '',
            'The saved 64–64 SST MLP remains better on five of the six native validation fields; DeepONet improves k. Against the original three-field POD-MLP on its own grid, DeepONet improves u but worsens w and p. Consequently there is no across-the-board DeepONet winner.',
            '', '## Interpretation and limits','',
            'DeepONet outperforms the paired data-only tanh MLP on every global field in the five-fold experiment. This supports this fixed training configuration. It does not prove that DeepONet is universally superior: capacities and optimizer budgets differ, the dataset is small, and only one seed and one architecture were evaluated.',
            '',s['protocol'],'',
            'The additional CFD runs meet the archived residual-control criterion but fail its stricter last-50 force-stability check. Their results should be treated as diagnostic comparisons. No new CFD cases or untouched test set were generated.',
            '',
            'The original raw 40+7 CFD archive could not be accessed. Saved native snapshots, saved MLP checkpoints, saved original-grid targets, and additional CFD exports were sufficient for this analysis. Total CL/CD are not evaluated: the old project\'s CL/CD MLP is a separate task and is not the reconstructed field model.',
            '',
            'The residual equations were verified against the original source on a synthetic state. Four unit tests passed, checkpoint inference was compared with saved validation predictions, and all original archive files were verified unchanged. See outputs/deeponet_sst/verification.json.',
            '',
            'Detailed coefficient errors, per-case and regional field errors, positivity fractions, wall errors and six residual RMS values are in outputs/deeponet_sst/evaluation. The pressure parity plots and Cp curves are diagnostic; coefficient errors depend on the chosen scaled POD basis.',
            '',
            '![Field comparison](outputs/deeponet_sst/evaluation/field_comparison.png)','',
            '![Coefficient parity](outputs/deeponet_sst/evaluation/coefficient_parity.png)','']
    (ROOT/'DEEPONET_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(ROOT/'DEEPONET_RESULTS.md')

if __name__=='__main__':main()
