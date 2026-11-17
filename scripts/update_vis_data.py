import safetensors.torch
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import json

p_meta = 'kaggle_training_output/v20_output/export_embeddings_gemini/benchmark_metadata.parquet'
df = pd.read_parquet(p_meta)
gem_mask = df['model'].str.startswith('gemini').values
classes = ['gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash']
y_true = np.array([classes.index(m) for m in df['model'].values[gem_mask]])

# Load existing vis_data.json
with open('local_data/vis_data.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

# Qwen Solo Plain (EXP-00)
sq = safetensors.torch.load_file('kaggle_training_output/v20_output/export_embeddings_gemini/benchmark_champion_logits.safetensors')
p_qwen_solo = torch.softmax(sq['logits_plain'][gem_mask, 5:9], dim=1)

# Qwen Specialist Gate (EXP-03c)
gh = safetensors.torch.load_file('kaggle_training_output/v20_output/export_gemini_gate_mlp/heads.safetensors')
X_qwen = safetensors.torch.load_file('kaggle_training_output/v20_output/export_embeddings_gemini/benchmark_features_concat.safetensors')['benchmark_features_concat'][gem_mask].float()
chunks = torch.split(X_qwen, 5120, dim=1)
normed = []
for i, c in enumerate(chunks):
    w = gh[f'norms.{i}.weight']
    b = gh[f'norms.{i}.bias']
    c = (c - c.mean(dim=-1, keepdim=True)) / (c.std(dim=-1, keepdim=True) + 1e-5)
    normed.append(c * w + b)
Z_q = torch.cat(normed, dim=1)
lp_ridge = (torch.matmul(Z_q, gh['plain.weight'].t()) + gh['plain.bias'])[:, :4]
h1 = torch.relu(torch.matmul(Z_q, gh['mlp_w1'].t()) + gh['mlp_b1'])
lp_mlp = (torch.matmul(h1, gh['mlp_w2'].t()) + gh['mlp_b2'])[:, :4]
p_qwen_gate_raw = 0.5 * torch.softmax(lp_ridge, dim=1) + 0.5 * torch.softmax(lp_mlp, dim=1)
# Blended with solo champion (gw=0.35) as in EXP-03c
p_qwen_exp03c = (1 - 0.35) * p_qwen_solo + 0.35 * p_qwen_gate_raw

# DeBERTa Gate (EXP-06c_01a)
dgh = safetensors.torch.load_file('.kaggle_training_output/export_deberta_gemini_gate_mlp/heads.safetensors')
X_deb = safetensors.torch.load_file('.kaggle_training_output/export_embeddings_deberta_finetuned/benchmark_features_concat.safetensors')['benchmark_features_concat'][gem_mask].float()
dchunks = torch.split(X_deb, 1024, dim=1)
dnormed = []
for i, c in enumerate(dchunks):
    w = dgh[f'norms.{i}.weight']
    b = dgh[f'norms.{i}.bias']
    c = (c - c.mean(dim=-1, keepdim=True)) / (c.std(dim=-1, keepdim=True) + 1e-5)
    dnormed.append(c * w + b)
Z_d = torch.cat(dnormed, dim=1)
p_deb_gate = torch.softmax((torch.matmul(Z_d, dgh['plain.weight'].t()) + dgh['plain.bias'])[:, :4], dim=1)

# Grand Blend (Qwen EXP-03c + DeBERTa Gate w=0.15)
p_grand = 0.85 * p_qwen_exp03c + 0.15 * p_deb_gate

# Predictions
preds_qwen = p_qwen_solo.argmax(dim=1).numpy()
preds_exp03c = p_qwen_exp03c.argmax(dim=1).numpy()
preds_deb = p_deb_gate.argmax(dim=1).numpy()
preds_grand = p_grand.argmax(dim=1).numpy()

d['bacc_qwen'] = float(balanced_accuracy_score(y_true, preds_qwen))
d['bacc_exp03c'] = float(balanced_accuracy_score(y_true, preds_exp03c))
d['bacc_deb'] = float(balanced_accuracy_score(y_true, preds_deb))
d['bacc_grand'] = float(balanced_accuracy_score(y_true, preds_grand))

d['cm_qwen'] = [[int(v) for v in row] for row in confusion_matrix(y_true, preds_qwen)]
d['cm_exp03c'] = [[int(v) for v in row] for row in confusion_matrix(y_true, preds_exp03c)]
d['cm_deb'] = [[int(v) for v in row] for row in confusion_matrix(y_true, preds_deb)]
d['cm_grand'] = [[int(v) for v in row] for row in confusion_matrix(y_true, preds_grand)]

# Sweep for Grand Blend curve (w between Qwen EXP-03c and DeBERTa Gate)
blend_curve = []
for w in np.linspace(0.0, 1.0, 21):
    p_b = (1 - w) * p_qwen_exp03c + w * p_deb_gate
    p_preds = p_b.argmax(dim=1).numpy()
    bacc = float(balanced_accuracy_score(y_true, p_preds) * 100)
    blend_curve.append({'w': round(float(w), 2), 'bacc': round(bacc, 2)})
d['grand_blend_curve'] = blend_curve

with open('local_data/vis_data.json', 'w', encoding='utf-8') as f:
    json.dump(d, f)

print('Updated vis_data.json successfully!')
print('Qwen Solo BAcc:', d['bacc_qwen'])
print('Qwen EXP-03c BAcc:', d['bacc_exp03c'])
print('DeBERTa Gate BAcc:', d['bacc_deb'])
print('Grand Blend BAcc:', d['bacc_grand'])
