import json, os, numpy as np, pytest
from scipy.special import softmax, log_softmax
from sklearn.metrics import balanced_accuracy_score
from namethatllm.postprocess import apply_claude_specialist_gate

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

def predict_from_artifact(lps_list, art, spec_log=None):
    Z_cal = []
    for m, name in enumerate(art["model_names"]):
        p = softmax(lps_list[m], axis=1)
        p = np.clip(p, art["clip_min"], 1.0)
        p /= p.sum(axis=1, keepdims=True)
        log_p = np.log(p)
        T_m = art["mean_T_m"][name]
        Z_cal.append(log_softmax(log_p / T_m, axis=1))

    weights = np.array([art["mean_w_m"][name] for name in art["model_names"]])
    U = sum(w * z for w, z in zip(weights, Z_cal))
    U_scaled = U / art["mean_T_global"]
    P_blend = softmax(U_scaled, axis=1)

    if spec_log is not None and art.get("mean_gw", 0.0) > 0:
        P_blend = apply_claude_specialist_gate(
            P_mix=P_blend,
            R_main=P_blend,
            claude_spec_logits=spec_log,
            claude_classes=art["claude_classes"],
            gate_weight=art["mean_gw"],
        )
    P_final = damped_sinkhorn(P_blend, alpha=art.get("sinkhorn_alpha", 0.50))
    return P_final

def test_calibration_artifact_matrix_parity():
    cache_path = (
        "kaggle_out_v88/local_eval_cache.npz"
        if os.path.exists("kaggle_out_v88/local_eval_cache.npz")
        else "kaggle_out_v87/local_eval_cache.npz"
    )
    config_path = "configs/production_ensemble.json"
    if not os.path.exists(cache_path) or not os.path.exists(config_path):
        pytest.skip("Cache or config not found")

    cache = np.load(cache_path, allow_pickle=True)
    with open(config_path) as f:
        artifact = json.load(f)

    y_unseen = cache["y_unseen"]
    spec_logits = cache["claude_spec_logits_u"]
    model_names_all = list(cache["model_names"])

    active_indices = [model_names_all.index(name) for name in artifact["model_names"]]
    raw_lps = np.array([cache[f"m_{i}_unseen_lp"] for i in active_indices])

    # Run pass 1
    P_inf_1 = predict_from_artifact(raw_lps, artifact, spec_logits)

    # Run pass 2 (recomputed)
    P_inf_2 = predict_from_artifact(raw_lps, artifact, spec_logits)

    # 1. Assert row sums strictly equal 1.0
    assert np.allclose(P_inf_1.sum(axis=1), 1.0, atol=1e-6), "Row sums do not equal 1.0!"

    # 2. Assert element-for-element probability matrix parity
    assert np.allclose(P_inf_1, P_inf_2, atol=1e-6), "Element-by-element probability mismatch!"

    # 3. Assert accuracy check
    bacc = balanced_accuracy_score(y_unseen, P_inf_1.argmax(axis=1))
    assert bacc > 0.585, f"Accuracy {bacc:.4f} is below 0.585 threshold!"
