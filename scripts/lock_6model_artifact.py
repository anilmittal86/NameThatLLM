import os, sys, json, numpy as np
from scipy.special import softmax, log_softmax
from scipy.optimize import minimize, minimize_scalar
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from namethatllm.postprocess import apply_claude_specialist_gate

cache_path = 'kaggle_out_v88/local_eval_cache.npz' if os.path.exists('kaggle_out_v88/local_eval_cache.npz') else 'kaggle_out_v87/local_eval_cache.npz'
print(f"Loading cache: {cache_path}")
cache = np.load(cache_path, allow_pickle=True)
y_unseen = cache['y_unseen']
model_names_all = list(cache['model_names'])
N = len(y_unseen)
groups = cache.get('groups_072_u')
if groups is None or len(groups) != N:
    groups = cache.get('groups_unseen')

# 6 models (drop redundant qwen3_14b_v5 and zero-lift trunks)
active_6_names = [
    'LinearSVC (4-View)',
    'Qwen3-14B Part-2',
    'qwen3_14b_inhouse',
    'qwen3_14b_ts',
    'qwen3_8b_v5',
    'DeBERTa-v3-Large'
]
active_6_indices = [model_names_all.index(name) for name in active_6_names]
M = len(active_6_names)
K = 18
eps = 0.05
lam = 0.10  # Optimal interior regularization parameter on v88

raw_lps_6 = np.array([cache[f'm_{i}_unseen_lp'] for i in active_6_indices])
P_clipped_6 = np.zeros_like(raw_lps_6)
for m in range(M):
    p_m = softmax(raw_lps_6[m], axis=1)
    p_m = np.clip(p_m, 1e-4, 1.0)
    p_m /= p_m.sum(axis=1, keepdims=True)
    P_clipped_6[m] = p_m
log_P_6 = np.log(P_clipped_6)

def damped_sinkhorn(P, alpha=0.50, iters=100, eps_val=1e-12):
    P_eps = np.clip(P, eps_val, 1.0)
    N_cur, K_cur = P_eps.shape
    p_bar = P_eps.mean(axis=0)
    c_target = (1.0 - alpha) * p_bar + alpha * (1.0 / K_cur)
    c_target = c_target / c_target.sum()
    r = np.ones(N_cur)
    c = N_cur * c_target
    u = np.ones(N_cur)
    for _ in range(iters):
        v = c / np.maximum(np.dot(P_eps.T, u), eps_val)
        u = r / np.maximum(np.dot(P_eps, v), eps_val)
    S = u[:, None] * P_eps * v[None, :]
    S = S / S.sum(axis=1, keepdims=True)
    return S

def get_loss_and_grad(theta, Z_tr, Y_smooth, lam):
    N_tr = len(Y_smooth)
    w = softmax(theta)
    U = np.tensordot(w, Z_tr, axes=(0, 0))
    P = softmax(U, axis=1)
    nll = -np.mean(np.sum(Y_smooth * np.log(np.clip(P, 1e-12, 1.0)), axis=1))
    reg = lam * np.sum((w - 1.0 / M) ** 2)
    loss = nll + reg
    G = (P - Y_smooth) / N_tr
    grad_w_nll = np.tensordot(Z_tr, G, axes=((1, 2), (0, 1)))
    grad_w_reg = 2.0 * lam * (w - 1.0 / M)
    d_w = grad_w_nll + grad_w_reg
    grad_theta = w * (d_w - np.dot(w, d_w))
    return loss, grad_theta

gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(raw_lps_6[0], y_unseen, groups=groups))
spec_logits = cache['claude_spec_logits_u']
claude_classes = list(cache['claude_classes'])

fold_weights = []
fold_model_temps = []
fold_global_temps = []
fold_gws = []

for fold, (tr_idx, val_idx) in enumerate(splits):
    N_tr = len(tr_idx)
    y_tr = y_unseen[tr_idx]
    Z_fold = np.zeros_like(log_P_6)
    model_temps = np.zeros(M)
    for m in range(M):
        log_p_m_tr = log_P_6[m, tr_idx, :]
        def nll_temp(T):
            scaled_lp = log_softmax(log_p_m_tr / T, axis=1)
            return -np.mean(scaled_lp[np.arange(N_tr), y_tr])
        res_t = minimize_scalar(nll_temp, bounds=(0.05, 10.0), method='bounded')
        T_opt = float(res_t.x)
        model_temps[m] = T_opt
        Z_fold[m] = log_softmax(log_P_6[m] / T_opt, axis=1)
    fold_model_temps.append(model_temps)
    
    Y_smooth_tr = np.full((N_tr, K), eps / K)
    Y_smooth_tr[np.arange(N_tr), y_tr] += (1.0 - eps)
    Z_tr = Z_fold[:, tr_idx, :]
    theta0 = np.zeros(M)
    res = minimize(
        fun=lambda th: get_loss_and_grad(th, Z_tr, Y_smooth_tr, lam),
        x0=theta0, method='L-BFGS-B', jac=True, options={'maxiter': 500}
    )
    w_k = softmax(res.x)
    fold_weights.append(w_k)

    U_tr = np.tensordot(w_k, Z_tr, axes=(0, 0))
    def nll_global_temp(T_g):
        scaled = log_softmax(U_tr / T_g, axis=1)
        return -np.mean(scaled[np.arange(N_tr), y_tr])
    res_tg = minimize_scalar(nll_global_temp, bounds=(0.1, 5.0), method='bounded')
    T_global_opt = float(res_tg.x)
    fold_global_temps.append(T_global_opt)
    
    P_tr = softmax(U_tr / T_global_opt, axis=1)
    best_gw_tr = 0.0
    best_bacc_tr = float(balanced_accuracy_score(y_tr, damped_sinkhorn(P_tr, alpha=0.50).argmax(axis=1)))
    for trial_gw in [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20]:
        P_tr_g = apply_claude_specialist_gate(
            P_mix=P_tr, R_main=P_tr, claude_spec_logits=spec_logits[tr_idx],
            claude_classes=claude_classes, gate_weight=trial_gw
        )
        b = float(balanced_accuracy_score(y_tr, damped_sinkhorn(P_tr_g, alpha=0.50).argmax(axis=1)))
        if b > best_bacc_tr:
            best_bacc_tr = b
            best_gw_tr = trial_gw
    fold_gws.append(best_gw_tr)

mean_T_m = np.mean(fold_model_temps, axis=0)
mean_w_m = np.mean(fold_weights, axis=0)
mean_T_global = float(np.mean(fold_global_temps))
mean_gw = float(np.mean(fold_gws))

prod_config_path = 'configs/production_ensemble.json'
try:
    with open(prod_config_path, 'r') as f:
        prod_cfg = json.load(f)
except Exception:
    prod_cfg = {}

# Map model names to internal config keys
model_key_map = {
    'LinearSVC (4-View)': 'svc_4view',
    'Qwen3-14B Part-2': 'qwen3_14b_p2',
    'qwen3_14b_inhouse': 'qwen3_14b_inhouse',
    'qwen3_14b_ts': 'qwen3_14b_ts',
    'qwen3_8b_v5': 'qwen3_8b_v5',
    'DeBERTa-v3-Large': 'deberta_p2'
}

prod_cfg.update({
    'model_names': active_6_names,
    'models': {model_key_map[name]: float(mean_w_m[i]) for i, name in enumerate(active_6_names)},
    'clip_min': 1e-4,
    'mean_T_m': {name: float(mean_T_m[i]) for i, name in enumerate(active_6_names)},
    'mean_w_m': {name: float(mean_w_m[i]) for i, name in enumerate(active_6_names)},
    'mean_T_global': mean_T_global,
    'mean_gw': mean_gw,
    'claude_classes': [str(c) for c in claude_classes],
    'sinkhorn_alpha': 0.50,
    'lambda_reg': lam,
    'label_smoothing_eps': eps
})
prod_cfg.setdefault('postprocess', {})['claude_specialist_weight'] = mean_gw
prod_cfg['postprocess']['sinkhorn_alpha'] = 0.50

with open(prod_config_path, 'w') as f:
    json.dump(prod_cfg, f, indent=2)

print(f"Updated {prod_config_path} with v88 6-model artifact.")

def predict_from_artifact(lps_list, art, spec_log=None):
    Z_cal = []
    for m, name in enumerate(art['model_names']):
        p = softmax(lps_list[m], axis=1)
        p = np.clip(p, art['clip_min'], 1.0)
        p /= p.sum(axis=1, keepdims=True)
        log_p = np.log(p)
        T_m = art['mean_T_m'][name]
        Z_cal.append(log_softmax(log_p / T_m, axis=1))
    
    weights = np.array([art['mean_w_m'][name] for name in art['model_names']])
    U = sum(w * z for w, z in zip(weights, Z_cal))
    U_scaled = U / art['mean_T_global']
    P_blend = softmax(U_scaled, axis=1)
    
    if spec_log is not None and art.get('mean_gw', 0) > 0:
        P_blend = apply_claude_specialist_gate(
            P_mix=P_blend, R_main=P_blend, claude_spec_logits=spec_log,
            claude_classes=art['claude_classes'], gate_weight=art['mean_gw']
        )
    P_final = damped_sinkhorn(P_blend, alpha=art['sinkhorn_alpha'])
    return P_final

P_inf = predict_from_artifact(raw_lps_6, prod_cfg, spec_logits)
assert np.allclose(P_inf.sum(axis=1), 1.0, atol=1e-6), "Simplex normalization failed!"
bacc_6 = balanced_accuracy_score(y_unseen, P_inf.argmax(axis=1))
print(f"6-Model Full Holdout Balanced Accuracy: {bacc_6:.4f}")

for name in active_6_names:
    w_val = prod_cfg['mean_w_m'][name] * 100
    t_val = prod_cfg['mean_T_m'][name]
    print(f"  {name:<23}: share={w_val:5.2f}% | T_m={t_val:.2f}")
print(f"Mean Global T: {mean_T_global:.3f} | Mean gw: {mean_gw:.3f} | Sinkhorn Alpha: 0.50 | Lambda: {lam}")
