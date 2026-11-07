import itertools
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from scipy.special import softmax
from namethatllm.config import LABELS, FAMILIES, calibrate_model_lp
from namethatllm.postprocess import (
    apply_claude_specialist_gate,
    apply_family_specialist_lora_gate,
    apply_unified_handwriting_shield,
)

def sinkhorn_func(log_probs, iters=60, strength=1.0, mix=1.0):
    P = softmax(log_probs, axis=1)
    N, K = P.shape
    u = np.ones(N)
    v = np.ones(K)
    for _ in range(iters):
        v = (N / K) / np.maximum(P.T @ u, 1e-12)
        u = 1.0 / np.maximum(P @ v, 1e-12)
    P_bal = P * np.outer(u, v)
    P_bal = P_bal / P_bal.sum(axis=1, keepdims=True)
    if strength < 1.0 or mix < 1.0:
        return (1.0 - mix) * P + mix * ((1.0 - strength) * P + strength * P_bal)
    return P_bal

d = np.load('local_data/local_eval_cache_v68_backup.npz', allow_pickle=True)
y_true = d['y_unseen']
texts = list(d['unseen_texts'])

models = {
    'p2': (calibrate_model_lp(d['m_2_unseen_lp'], 'qwen3_14b_p2'), 0.52),
    'inh': (calibrate_model_lp(d['m_4_unseen_lp'], 'qwen3_14b_inhouse'), 0.33),
    'deb': (calibrate_model_lp(d['m_5_unseen_lp'], 'deberta_p2'), 0.33),
    'svc': (calibrate_model_lp(d['m_0_unseen_lp'], 'svc_4view'), 0.30),
    'sb': (calibrate_model_lp(d['extra_1_unseen_lp'], 'spectral_bank'), 0.15),
    'lex': (calibrate_model_lp(d['m_1_unseen_lp'], 'lexical_5v'), 0.10),
}

cl_logits = d['claude_spec_logits_unseen']
gpt_logits = d['gpt_spec_logits_unseen']
gem_logits = d['gemini_spec_logits_unseen']
glm_logits = d['glm_spec_logits_unseen']

combos = [
    ('qwen3_14b_p2 (Solo)', ['p2']),
    ('qwen3_14b_inhouse (Solo)', ['inh']),
    ('deberta_p2 (Solo)', ['deb']),
    ('svc_4view (Solo)', ['svc']),
    ('spectral_bank (Solo)', ['sb']),
    ('lexical_5v (Solo)', ['lex']),
    ('p2 + svc_4view', ['p2', 'svc']),
    ('p2 + qwen3_14b_inhouse', ['p2', 'inh']),
    ('p2 + deberta_p2', ['p2', 'deb']),
    ('inh + svc_4view', ['inh', 'svc']),
    ('deb + svc_4view', ['deb', 'svc']),
    ('p2 + inh + svc', ['p2', 'inh', 'svc']),
    ('p2 + deb + svc', ['p2', 'deb', 'svc']),
    ('p2 + inh + deb', ['p2', 'inh', 'deb']),
    ('p2 + inh + deb + svc', ['p2', 'inh', 'deb', 'svc']),
    ('p2 + inh + svc + sb', ['p2', 'inh', 'svc', 'sb']),
    ('p2 + inh + deb + svc + sb', ['p2', 'inh', 'deb', 'svc', 'sb']),
    ('Full Clean (p2+inh+deb+svc+sb+lex)', ['p2', 'inh', 'deb', 'svc', 'sb', 'lex']),
]

headers = [
    "Model Combination", "Raw", "SoftSK", "PureSK", "+Claude", "+Gemini", "+GPT", "+GLM", "+AllSpec", "FullPipe"
]
col_w = [36, 7, 7, 7, 8, 8, 7, 7, 9, 9]

header_str = " | ".join(h.ljust(w) for h, w in zip(headers, col_w))
sep_str = "-+-".join("-" * w for w in col_w)

print("=" * len(header_str))
print("COMPLETE ACCURACY MATRIX: CLEAN MODEL COMBINATIONS x POST-PROCESSING STEPS")
print("=" * len(header_str))
print(header_str)
print(sep_str)

results = []
for name, keys in combos:
    tot_w = sum(models[k][1] for k in keys)
    lp_mix = sum((models[k][1] / tot_w) * models[k][0] for k in keys)
    
    # 1. Raw
    p_raw = softmax(lp_mix, axis=1)
    acc_raw = balanced_accuracy_score(y_true, p_raw.argmax(axis=1)) * 100
    
    # 2. Soft Sinkhorn (strength=0.65, mix=0.35)
    p_soft_sk = sinkhorn_func(lp_mix, iters=60, strength=0.65, mix=0.35)
    acc_soft_sk = balanced_accuracy_score(y_true, p_soft_sk.argmax(axis=1)) * 100
    
    # 3. Pure Sinkhorn (strength=1.0, mix=1.0)
    p_pure_sk = sinkhorn_func(lp_mix, iters=60, strength=1.0, mix=1.0)
    acc_pure_sk = balanced_accuracy_score(y_true, p_pure_sk.argmax(axis=1)) * 100
    
    # 4. + Claude
    p_cl = apply_claude_specialist_gate(p_pure_sk, lp_mix, cl_logits, gate_weight=0.15)
    acc_cl = balanced_accuracy_score(y_true, p_cl.argmax(axis=1)) * 100
    
    # 5. + Gemini
    p_gem = apply_family_specialist_lora_gate(p_pure_sk, lp_mix, gem_logits, family_cols=FAMILIES['gemini'], gate_weight=0.35)
    acc_gem = balanced_accuracy_score(y_true, p_gem.argmax(axis=1)) * 100

    # 6. + GPT
    p_gpt = apply_family_specialist_lora_gate(p_pure_sk, lp_mix, gpt_logits, family_cols=FAMILIES['gpt'], gate_weight=0.30)
    acc_gpt = balanced_accuracy_score(y_true, p_gpt.argmax(axis=1)) * 100

    # 7. + GLM
    p_glm = apply_family_specialist_lora_gate(p_pure_sk, lp_mix, glm_logits, family_cols=FAMILIES['glm'], gate_weight=0.30)
    acc_glm = balanced_accuracy_score(y_true, p_glm.argmax(axis=1)) * 100
    
    # 8. + All Specialists
    p_all_spec = p_pure_sk.copy()
    p_all_spec = apply_claude_specialist_gate(p_all_spec, lp_mix, cl_logits, gate_weight=0.15)
    p_all_spec = apply_family_specialist_lora_gate(p_all_spec, lp_mix, gem_logits, family_cols=FAMILIES['gemini'], gate_weight=0.35)
    p_all_spec = apply_family_specialist_lora_gate(p_all_spec, lp_mix, gpt_logits, family_cols=FAMILIES['gpt'], gate_weight=0.30)
    p_all_spec = apply_family_specialist_lora_gate(p_all_spec, lp_mix, glm_logits, family_cols=FAMILIES['glm'], gate_weight=0.30)
    acc_all_spec = balanced_accuracy_score(y_true, p_all_spec.argmax(axis=1)) * 100
    
    # 9. Full Pipeline (All Specialists + Handwriting Shield)
    p_full = apply_unified_handwriting_shield(p_all_spec, texts, curly_quote_multiplier=0.368, continuous_hw_weight=0.04)
    acc_full = balanced_accuracy_score(y_true, p_full.argmax(axis=1)) * 100

    row_vals = [
        name,
        f"{acc_raw:5.2f}%",
        f"{acc_soft_sk:5.2f}%",
        f"{acc_pure_sk:5.2f}%",
        f"{acc_cl:5.2f}%",
        f"{acc_gem:5.2f}%",
        f"{acc_gpt:5.2f}%",
        f"{acc_glm:5.2f}%",
        f"{acc_all_spec:5.2f}%",
        f"{acc_full:5.2f}%",
    ]
    row_str = " | ".join(v.ljust(w) for v, w in zip(row_vals, col_w))
    print(row_str)
    results.append((name, acc_raw, acc_pure_sk, acc_all_spec, acc_full))

print("=" * len(header_str))

# Summary of best performers
best_pure = max(results, key=lambda x: x[2])
best_full = max(results, key=lambda x: x[4])
print(f"\nTOP PERFORMER (Pure Sinkhorn):    {best_pure[0]} -> {best_pure[2]:.2f}%")
print(f"TOP PERFORMER (Full Pipeline):     {best_full[0]} -> {best_full[4]:.2f}%")

