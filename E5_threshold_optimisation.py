from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score

FOLDER = Path(__file__).resolve().parent
INPUT = FOLDER / 'validation_predictions_four_models.csv'
OUTPUT = FOLDER / 'results'
OUTPUT.mkdir(exist_ok=True)

MODELS = ['E1_Standard_Logistic_Regression', 'E2_Random_Forest',
          'E3_Gradient_Boosting', 'E4_SVM']

data = pd.read_csv(INPUT)
required = {'model', 'group', 'target_week_start', 'posting_next_week', 'predicted_probability'}
if not required.issubset(data.columns):
    raise ValueError(f'Missing columns: {sorted(required - set(data.columns))}')
if set(data['model'].unique()) != set(MODELS):
    raise ValueError('Model labels do not match the expected four models.')

results = []
for name in MODELS:
    part = data.loc[data['model'] == name].copy()
    if len(part) != 6337:
        raise ValueError(f'{name}: expected 6337 validation observations; found {len(part)}')
    if not pd.to_datetime(part['target_week_start'], utc=True).dt.year.eq(2025).all():
        raise ValueError(f'{name}: found observations outside target year 2025')
    if part.duplicated(['group', 'target_week_start']).any():
        raise ValueError(f'{name}: duplicate group-week observations')
    if part[['posting_next_week', 'predicted_probability']].isna().any().any():
        raise ValueError(f'{name}: missing target or probability')
    y = part['posting_next_week'].astype(int).to_numpy()
    p = part['predicted_probability'].astype(float).to_numpy()
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError(f'{name}: invalid target or probability')

    precision, recall, thresholds = precision_recall_curve(y, p)
    f1_values = np.divide(2 * precision[:-1] * recall[:-1],
                          precision[:-1] + recall[:-1],
                          out=np.zeros_like(thresholds),
                          where=(precision[:-1] + recall[:-1]) > 0)
    best = int(np.argmax(f1_values))  # first threshold in case of a tie
    threshold = float(thresholds[best])
    prediction = (p >= threshold).astype(int)
    default_prediction = (p >= 0.50).astype(int)

    part['predicted_at_optimal_threshold'] = prediction
    part.to_csv(OUTPUT / f'{name}_validation_predictions.csv', index=False)
    results.append({
        'model': name, 'best_threshold': threshold,
        'precision': precision_score(y, prediction, zero_division=0),
        'recall': recall_score(y, prediction, zero_division=0),
        'f1': f1_score(y, prediction, zero_division=0),
        'f1_at_0_50': f1_score(y, default_prediction, zero_division=0),
        'true_positives': int(((y == 1) & (prediction == 1)).sum()),
        'false_positives': int(((y == 0) & (prediction == 1)).sum()),
    })

comparison = pd.DataFrame(results).sort_values('f1', ascending=False)
comparison.to_csv(OUTPUT / 'E5_four_model_comparison.csv', index=False)
print('\nE5: 2025 validation-only threshold optimisation\n')
print(comparison.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
print('\nSelected:', comparison.iloc[0]['model'])
print('Frozen threshold:', f"{comparison.iloc[0]['best_threshold']:.8f}")
print('\nSaved files in:', OUTPUT)
print('Do not use 2026 test data to select or adjust the threshold.')
