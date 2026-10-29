"""E1: unweighted logistic regression for next-week observed ransomware postings.

Run directly in VS Code (Run Python File). No command-line arguments.
The 2026 test dataset is deliberately not used.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    accuracy_score, balanced_accuracy_score, average_precision_score, roc_auc_score
)

# STEP 1: Load the already-separated historical data.
folder = Path(__file__).resolve().parent
train = pd.read_csv(folder / 'train.csv')          # 2022-2024
validation = pd.read_csv(folder / 'validation.csv') # 2025
assert set(train['target_year']) == {2022, 2023, 2024}
assert set(validation['target_year']) == {2025}

# STEP 2: Select only information available at the end of the current week.
numeric_features = [
    'attacks_7d', 'attacks_30d', 'attacks_90d',
    'attacks_120d', 'attacks_365d', 'days_since_last_post'
]
features = numeric_features + ['campaign_status']
X_train = train[features]
y_train = train['posting_next_week'].astype(int)
X_validation = validation[features]
y_validation = validation['posting_next_week'].astype(int)

# STEP 3: Scale numerical columns, encode status, and train logistic regression.
# The pipeline learns preprocessing ONLY from the training dataset.
preprocessing = ColumnTransformer([
    ('numbers', StandardScaler(), numeric_features),
    ('status', OneHotEncoder(handle_unknown='ignore'), ['campaign_status'])
])
model = Pipeline([
    ('preprocessing', preprocessing),
    ('logistic_regression', LogisticRegression(max_iter=5000, random_state=42))
])
model.fit(X_train, y_train)

# STEP 4: Predict the probability of a posting in the following week.
probability = model.predict_proba(X_validation)[:, 1]
predicted_posting = (probability >= 0.50).astype(int)

# STEP 5: Evaluate binary predictions against actual 2025 outcomes.
tn, fp, fn, tp = confusion_matrix(y_validation, predicted_posting, labels=[0, 1]).ravel()
results = {
    'true_positives': int(tp), 'false_positives': int(fp),
    'false_negatives': int(fn), 'true_negatives': int(tn),
    'precision': precision_score(y_validation, predicted_posting, zero_division=0),
    'recall': recall_score(y_validation, predicted_posting, zero_division=0),
    'f1': f1_score(y_validation, predicted_posting, zero_division=0),
    'binary_accuracy': accuracy_score(y_validation, predicted_posting),
    'balanced_accuracy': balanced_accuracy_score(y_validation, predicted_posting),
    'average_precision': average_precision_score(y_validation, probability),
    'roc_auc': roc_auc_score(y_validation, probability),
}

# E0-compatible five-status evaluation: if no posting, advance recency by 7 days.
future_days = validation['days_since_last_post'].to_numpy() + 7
no_posting_status = np.select(
    [future_days <= 30, future_days <= 90, future_days <= 365],
    ['active', 'declining', 'dormant'], default='inactive'
)
predicted_status = np.where(predicted_posting == 1, 'attack_week', no_posting_status)
results['five_status_accuracy'] = accuracy_score(validation['next_week_status'], predicted_status)
results['five_status_macro_f1'] = f1_score(
    validation['next_week_status'], predicted_status, average='macro', zero_division=0
)

# STEP 6: Save predictions, metrics, and trained model for later experiments.
output = folder / 'E1_results'
output.mkdir(exist_ok=True)
pd.DataFrame([results]).to_csv(output / 'metrics.csv', index=False)
predictions = validation[['group', 'week_start', 'next_week_status', 'posting_next_week']].copy()
predictions['predicted_probability'] = probability
predictions['predicted_posting'] = predicted_posting
predictions['predicted_next_week_status'] = predicted_status
predictions.to_csv(output / 'validation_predictions.csv', index=False)
dump(model, output / 'E1_model.joblib')

print('\nE1 RESULTS — 2025 VALIDATION')
for metric, value in results.items():
    print(f'{metric}: {value:.4f}' if isinstance(value, float) else f'{metric}: {value}')
print('\nSaved results in:', output)
