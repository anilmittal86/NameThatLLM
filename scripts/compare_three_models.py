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

# 1. Plain head Qwen 14B in house model
sq = safetensors.torch.load_file('kaggle_training_output/v20_output/export_embeddings_gemini/benchmark_champion_logits.safetensors')
qwen_lp = sq['logits_plain'][gem_mask, 5:9]
p_qwen = torch.softmax(qwen_lp, dim=1)
preds_qwen = p_qwen.argmax(dim=1).numpy()

# 2. Finetuned Qwen 14B on Gemini (v14 specialist)
q_spec_heads = safetensors.torch.load_file('local_data/training_v14_output/export_gemini_specialist_14b/heads.safetensors')
p_qwen_feat = 'kaggle_training_output/v20_output/export_embeddings_gemini/benchmark_features_concat.safetensors'
X_qwen = safetensors.torch.load_file(p_qwen_feat)['benchmark_features_concat'][gem_mask].float()
chunks = torch.split(X_qwen, 5120, dim=1)
normed_chunks = []
for i, chunk in enumerate(chunks):
    w = q_spec_heads[f'norms.{i}.weight']
    b = q_spec_heads[f'norms.{i}.bias']
    chunk = (chunk - chunk.mean(dim=-1, keepdim=True)) / (chunk.std(dim=-1, keepdim=True) + 1e-5)
    normed_chunks.append(chunk * w + b)
Z_qwen_spec = torch.cat(normed_chunks, dim=1)
q_spec_lp = torch.matmul(Z_qwen_spec, q_spec_heads['plain.weight'].t()) + q_spec_heads['plain.bias']
p_qwen_spec = torch.softmax(q_spec_lp, dim=1)
preds_qwen_spec = p_qwen_spec.argmax(dim=1).numpy()

# 3. Finetuned DeBERTa on Gemini (EXP-06c_01a Gate)
gate_heads = safetensors.torch.load_file('.kaggle_training_output/export_deberta_gemini_gate_mlp/heads.safetensors')
p_deb_feat = '.kaggle_training_output/export_embeddings_deberta_finetuned/benchmark_features_concat.safetensors'
X_deb = safetensors.torch.load_file(p_deb_feat)['benchmark_features_concat'][gem_mask].float()
chunks = torch.split(X_deb, 1024, dim=1)
normed_chunks = []
for i, chunk in enumerate(chunks):
    w = gate_heads[f'norms.{i}.weight']
    b = gate_heads[f'norms.{i}.bias']
    chunk = (chunk - chunk.mean(dim=-1, keepdim=True)) / (chunk.std(dim=-1, keepdim=True) + 1e-5)
    normed_chunks.append(chunk * w + b)
Z_deb_gate = torch.cat(normed_chunks, dim=1)
gate_lp = (torch.matmul(Z_deb_gate, gate_heads['plain.weight'].t()) + gate_heads['plain.bias'])[:, :4]
p_deb_gate = torch.softmax(gate_lp, dim=1)
preds_deb_gate = p_deb_gate.argmax(dim=1).numpy()

# 4. Ensemble (Qwen In-House + DeBERTa Gate)
p_ens = 0.80 * p_qwen + 0.20 * p_deb_gate
preds_ens = p_ens.argmax(dim=1).numpy()

def format_row(name, preds):
    bacc = balanced_accuracy_score(y_true, preds)
    correct = sum(preds == y_true)
    recalls = [(preds[y_true == i] == i).mean() * 100 for i in range(4)]
    cm = confusion_matrix(y_true, preds)
    return {
        'name': name,
        'bacc': bacc * 100,
        'correct': int(correct),
        'recalls': [float(r) for r in recalls],
        'cm': [[int(v) for v in row] for row in cm]
    }

m1 = format_row('1. In-House Qwen-14B Plain Head', preds_qwen)
m2 = format_row('2. Fine-Tuned Qwen-14B Specialist (v14)', preds_qwen_spec)
m3 = format_row('3. Fine-Tuned DeBERTa Specialist (EXP-06c_01a Gate)', preds_deb_gate)
m4 = format_row('4. Ensemble (Qwen 80% + DeBERTa 20%)', preds_ens)

print(json.dumps({
    'm1': {'bacc': m1['bacc'], 'correct': m1['correct'], 'recalls': m1['recalls'], 'cm': m1['cm']},
    'm2': {'bacc': m2['bacc'], 'correct': m2['correct'], 'recalls': m2['recalls'], 'cm': m2['cm']},
    'm3': {'bacc': m3['bacc'], 'correct': m3['correct'], 'recalls': m3['recalls'], 'cm': m3['cm']},
    'm4': {'bacc': m4['bacc'], 'correct': m4['correct'], 'recalls': m4['recalls'], 'cm': m4['cm']},
}))
