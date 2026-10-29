from pathlib import Path
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, accuracy_score, average_precision_score,
                             roc_auc_score)

folder = Path(__file__).resolve().parent
train = pd.read_csv(folder / 'train.csv')
valid = pd.read_csv(folder / 'validation.csv')

numeric = ['attacks_7d', 'attacks_30d', 'attacks_90d',
           'attacks_120d', 'attacks_365d', 'days_since_last_post']
category = ['campaign_status']
features = numeric + category

for data, year_min, year_max in [(train, 2022, 2024), (valid, 2025, 2025)]:
    needed = features + ['posting_next_week', 'next_week_status', 'target_week_start']
    if not set(needed).issubset(data.columns):
        raise ValueError(f'Missing columns: {set(needed) - set(data.columns)}')
    target_dates = pd.to_datetime(data['target_week_start'], utc=True)
    if not target_dates.dt.year.between(year_min, year_max).all():
        raise ValueError('Incorrect target-year partition')
    if data[features].isna().any().any():
        raise ValueError('Missing input feature values')
    if not (data['posting_next_week'].astype(int) ==
            data['next_week_status'].eq('attack_week').astype(int)).all():
        raise ValueError('Binary target does not match next-week status')

X_train, y_train = train[features], train['posting_next_week'].astype(int)
X_valid, y_valid = valid[features], valid['posting_next_week'].astype(int)

preprocess = ColumnTransformer([
    ('numbers', StandardScaler(), numeric),
    ('status', OneHotEncoder(handle_unknown='ignore'), category)
])

model = Pipeline([
    ('preprocess', preprocess),
    ('svm', SVC(kernel='rbf', C=1.0, gamma='scale',
                class_weight='balanced', probability=True, random_state=42))
])
model.fit(X_train, y_train)

probability = model.predict_proba(X_valid)[:, 1]
predicted = (probability >= 0.50).astype(int)

tn, fp, fn, tp = confusion_matrix(y_valid, predicted, labels=[0, 1]).ravel()
metrics = {
    'model': 'E4_balanced_RBF_SVM',
    'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn),
    'precision': precision_score(y_valid, predicted, zero_division=0),
    'recall': recall_score(y_valid, predicted, zero_division=0),
    'F1': f1_score(y_valid, predicted, zero_division=0),
    'accuracy': accuracy_score(y_valid, predicted),
    'average_precision': average_precision_score(y_valid, probability),
    'ROC_AUC': roc_auc_score(y_valid, probability)
}

days_next_week = valid['days_since_last_post'].to_numpy() + 7
five_status = np.where(predicted == 1, 'attack_week',
    np.select([days_next_week <= 30, days_next_week <= 90,
               days_next_week <= 365],
              ['active', 'declining', 'dormant'], default='inactive'))
metrics['five_status_accuracy'] = accuracy_score(valid['next_week_status'], five_status)
metrics['five_status_macro_F1'] = f1_score(
    valid['next_week_status'], five_status,
    labels=['attack_week', 'active', 'declining', 'dormant', 'inactive'],
    average='macro', zero_division=0)

output = folder / 'E4_results'
output.mkdir(exist_ok=True)
pd.DataFrame([metrics]).to_csv(output / 'E4_metrics.csv', index=False)
predictions = valid[['group', 'week_start', 'next_week_status', 'posting_next_week']].copy()
predictions['posting_probability'] = probability
predictions['predicted_posting'] = predicted
predictions['predicted_next_week_status'] = five_status
predictions.to_csv(output / 'E4_predictions.csv', index=False)
dump(model, output / 'E4_model.joblib')

print('E4 validation results (2025):')
for key, value in metrics.items():
    print(f'{key}: {value:.4f}' if isinstance(value, float) else f'{key}: {value}')
print('Saved files to:', output)
