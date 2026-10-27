import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import joblib
from pathlib import Path

df_train = pd.read_csv("train.csv")
y_true = df_train['model'].values

# 1. Validation (OOF) Metrics from our 5-fold GroupKFold CV
oof_probs = np.load("artifacts/oof_stage4_probabilities.npy")
from src.config import TARGET_MODELS
y_val_pred_idx = np.argmax(oof_probs, axis=1)
y_val_pred = [TARGET_MODELS[i] for i in y_val_pred_idx]

val_acc = accuracy_score(y_true, y_val_pred)
val_bacc = balanced_accuracy_score(y_true, y_val_pred)

print(f"Validation (5-Fold OOF) Standard Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
print(f"Validation (5-Fold OOF) Balanced Accuracy: {val_bacc:.4f} ({val_bacc*100:.2f}%)")

# 2. Training Metrics on the fitted models
fe = joblib.load("artifacts/feature_extractor.joblib")
flat_model = joblib.load("artifacts/flat_model.joblib")
family_ensemble = joblib.load("artifacts/family_ensemble.joblib")

X_train = fe.transform(df_train['text'].fillna('').tolist())
raw_p = flat_model.predict_proba(X_train)
gated_p = family_ensemble.apply_gated_blend(X_train, raw_p)

# Training predictions
y_train_pred_idx = np.argmax(gated_p, axis=1)
y_train_pred = [TARGET_MODELS[i] for i in y_train_pred_idx]

train_acc = accuracy_score(y_true, y_train_pred)
train_bacc = balanced_accuracy_score(y_true, y_train_pred)

print(f"Training Set Standard Accuracy:            {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"Training Set Balanced Accuracy:            {train_bacc:.4f} ({train_bacc*100:.2f}%)")

# Also calculate provider-level family accuracy
PROVIDER_FAMILIES = {
    'claude': ['claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5'],
    'gemini': ['gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash'],
    'glm': ['glm-5.3', 'glm-5.3-flash'],
    'gpt': ['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra'],
    'kimi': ['kimi-k3'],
    'qwen': ['qwen3.8-max']
}
m_to_f = {m: f for f, ms in PROVIDER_FAMILIES.items() for m in ms}
y_true_fam = [m_to_f[m] for m in y_true]
y_val_fam = [m_to_f[m] for m in y_val_pred]
y_tr_fam = [m_to_f[m] for m in y_train_pred]

print(f"\nProvider Family Validation Accuracy:       {accuracy_score(y_true_fam, y_val_fam):.4f} ({accuracy_score(y_true_fam, y_val_fam)*100:.2f}%)")
print(f"Provider Family Training Accuracy:         {accuracy_score(y_true_fam, y_tr_fam):.4f} ({accuracy_score(y_true_fam, y_tr_fam)*100:.2f}%)")
