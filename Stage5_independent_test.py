from pathlib import Path
import json
import numpy as np
import pandas as pd
from joblib import load, dump
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
    f1_score, accuracy_score, average_precision_score, roc_auc_score)

folder = Path(__file__).resolve().parent
selection = json.loads((folder / "E6_selection.json").read_text())
assert selection["selected_experiment"] == "E1_Standard_Logistic_Regression"
threshold = float(selection["selected_threshold"])
test = pd.read_csv(folder / "test.csv")
assert len(test) == 6302 and test["posting_next_week"].sum() == 200
assert test["target_year"].eq(2026).all()

numbers = ["attacks_7d", "attacks_30d", "attacks_90d", "attacks_120d",
           "attacks_365d", "days_since_last_post"]
features = numbers + ["campaign_status"]
assert test[features].notna().all().all()

model_file = folder / "E1_model.joblib"
if model_file.exists():
    model = load(model_file)
else:
    train = pd.read_csv(folder / "train.csv")
    assert set(train["target_year"]) == {2022, 2023, 2024}
    preprocess = ColumnTransformer([
        ("numbers", StandardScaler(), numbers),
        ("status", OneHotEncoder(handle_unknown="ignore"), ["campaign_status"])
    ])
    model = Pipeline([
        ("preprocessing", preprocess),
        ("logistic_regression", LogisticRegression(max_iter=5000, random_state=42))
    ])
    model.fit(train[features], train["posting_next_week"].astype(int))
    dump(model, model_file)

actual = test["posting_next_week"].astype(int).to_numpy()
probability = model.predict_proba(test[features])[:, 1]
predicted = (probability >= threshold).astype(int)

def evaluate(name, predictions, scores):
    tn, fp, fn, tp = confusion_matrix(actual, predictions, labels=[0, 1]).ravel()
    return {"model": name, "TP": int(tp), "FP": int(fp),
            "FN": int(fn), "TN": int(tn),
            "precision": precision_score(actual, predictions, zero_division=0),
            "recall": recall_score(actual, predictions, zero_division=0),
            "F1": f1_score(actual, predictions, zero_division=0),
            "binary_accuracy": accuracy_score(actual, predictions),
            "average_precision": average_precision_score(actual, scores),
            "ROC_AUC": roc_auc_score(actual, scores)}

rows = [evaluate("E0_no_posting_baseline", np.zeros(len(test), dtype=int), np.zeros(len(test))),
        evaluate("E1_Standard_Logistic_Regression", predicted, probability)]

future_days = test["days_since_last_post"].to_numpy() + 7
no_post_status = np.select([future_days <= 30, future_days <= 90,
                            future_days <= 365],
                           ["active", "declining", "dormant"], default="inactive")
labels = ["attack_week", "active", "declining", "dormant", "inactive"]
actual_status = test["next_week_status"]
for row, binary in zip(rows, [np.zeros(len(test), dtype=int), predicted]):
    status = np.where(binary == 1, "attack_week", no_post_status)
    row["five_status_accuracy"] = accuracy_score(actual_status, status)
    row["five_status_macro_F1"] = f1_score(actual_status, status,
                                           labels=labels, average="macro",
                                           zero_division=0)

output = folder / "results"
output.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(output / "test_metrics.csv", index=False)
predictions = test[["group", "week_start", "target_week_start",
                    "posting_next_week", "next_week_status"]].copy()
predictions["E1_probability"] = probability
predictions["E1_predicted_posting"] = predicted
predictions["E1_predicted_status"] = np.where(predicted == 1,
                                              "attack_week", no_post_status)
predictions["E0_predicted_status"] = no_post_status
predictions.to_csv(output / "test_predictions.csv", index=False)
print("\nSTAGE 5: INDEPENDENT 2026 TEST RESULTS")
print("Frozen E1 threshold:", threshold)
print(pd.DataFrame(rows).to_string(index=False))
print("Saved results in:", output)
