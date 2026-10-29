from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
    f1_score, accuracy_score, balanced_accuracy_score,
    average_precision_score, roc_auc_score)

folder = Path(__file__).resolve().parent
train = pd.read_csv(folder / 'train.csv')
validation = pd.read_csv(folder / 'validation.csv')
assert set(train['target_year']) == {2022, 2023, 2024}
assert set(validation['target_year']) == {2025}

numbers = ['attacks_7d', 'attacks_30d', 'attacks_90d',
           'attacks_120d', 'attacks_365d', 'days_since_last_post']
features = numbers + ['campaign_status']
X_train = train[features]
y_train = train['posting_next_week'].astype(int)
X_val = validation[features]
y_val = validation['posting_next_week'].astype(int)

class_counts = y_train.value_counts()
weights_by_class = {label: len(y_train) / (2 * count)
                    for label, count in class_counts.items()}
sample_weights = y_train.map(weights_by_class).to_numpy()

preprocessing = ColumnTransformer([
    ('numbers', StandardScaler(), numbers),
    ('status', OneHotEncoder(handle_unknown='ignore'), ['campaign_status'])
])
model = Pipeline([
    ('preprocessing', preprocessing),
    ('boosting', HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
        l2_regularization=1.0, random_state=42
    ))
])
model.fit(X_train, y_train, boosting__sample_weight=sample_weights)

probability = model.predict_proba(X_val)[:, 1]
predicted = (probability >= 0.50).astype(int)
tn, fp, fn, tp = confusion_matrix(y_val, predicted, labels=[0, 1]).ravel()
results = {
    'true_positives': int(tp), 'false_positives': int(fp),
    'false_negatives': int(fn), 'true_negatives': int(tn),
    'precision': precision_score(y_val, predicted, zero_division=0),
    'recall': recall_score(y_val, predicted, zero_division=0),
    'f1': f1_score(y_val, predicted, zero_division=0),
    'binary_accuracy': accuracy_score(y_val, predicted),
    'balanced_accuracy': balanced_accuracy_score(y_val, predicted),
    'average_precision': average_precision_score(y_val, probability),
    'roc_auc': roc_auc_score(y_val, probability)
}

future_days = validation['days_since_last_post'].to_numpy() + 7
no_post_status = np.select(
    [future_days <= 30, future_days <= 90, future_days <= 365],
    ['active', 'declining', 'dormant'], default='inactive'
)
predicted_status = np.where(predicted == 1, 'attack_week', no_post_status)
results['five_status_accuracy'] = accuracy_score(validation['next_week_status'], predicted_status)
results['five_status_macro_f1'] = f1_score(
    validation['next_week_status'], predicted_status, average='macro', zero_division=0
)

out = folder / 'E3_results'
out.mkdir(exist_ok=True)
pd.DataFrame([results]).to_csv(out / 'metrics.csv', index=False)
predictions = validation[['group', 'week_start', 'next_week_status', 'posting_next_week']].copy()
predictions['predicted_probability'] = probability
predictions['predicted_posting'] = predicted
predictions['predicted_next_week_status'] = predicted_status
predictions.to_csv(out / 'validation_predictions.csv', index=False)
dump(model, out / 'E3_model.joblib')
print('\nE3 — 2025 VALIDATION RESULTS')
for name, value in results.items():
    print(f'{name}: {value:.4f}' if isinstance(value, float) else f'{name}: {value}')
print('\nResults saved to:', out)
