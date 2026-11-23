import os, sys, time, json
import numpy as np
from scipy.special import softmax, log_softmax
from scipy.optimize import minimize, minimize_scalar
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
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

def get_loss_and_grad(theta, Z_tr, Y_smooth, lam, M):
    N_tr = len(Y_smooth)
    w = softmax(theta)
    U = np.tensordot(w, Z_tr, axes=(0, 0)) # (N_tr, K)
    P = softmax(U, axis=1)
    
    # Label smoothed cross-entropy
    nll = -np.mean(np.sum(Y_smooth * np.log(np.clip(P, 1e-12, 1.0)), axis=1))
    
    # Regularization toward uniform weights
    reg = lam * np.sum((w - 1.0 / M) ** 2)
    loss = nll + reg
    
    # Gradient w.r.t theta
    G = (P - Y_smooth) / N_tr
    grad_w_nll = np.tensordot(Z_tr, G, axes=((1, 2), (0, 1)))
    grad_w_reg = 2.0 * lam * (w - 1.0 / M)
    d_w = grad_w_nll + grad_w_reg
    grad_theta = w * (d_w - np.dot(w, d_w))
    return loss, grad_theta

def run_5fold_pooling(
    cache_path,
    model_subset_names=None,
    lambdas=[0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50],
    eps=0.05,
    sinkhorn_alpha=0.50
):
    print(f"\n{'='*80}")
    print(f"Loading cache: {cache_path}")
    cache = np.load(cache_path, allow_pickle=True)
    y_unseen = cache['y_unseen']
    all_names = list(cache['model_names'])
    N = len(y_unseen)
    K = 18

    groups = cache.get('groups_072_u')
    if groups is None or len(groups) != N:
        groups = cache.get('groups_unseen')

    # Claude specialist logits and classes
    spec_logits = cache['claude_spec_logits_u']
    claude_classes = list(cache['claude_classes'])

    # Determine active models
    if model_subset_names is None:
        # Default: all models except 'Qwen3-8B Claude Head'
        claude_idx = all_names.index('Qwen3-8B Claude Head') if 'Qwen3-8B Claude Head' in all_names else -1
        active_indices = [i for i in range(len(all_names)) if i != claude_idx]
        active_names = [all_names[i] for i in active_indices]
    else:
        active_indices = [all_names.index(name) for name in model_subset_names]
        active_names = list(model_subset_names)

    M = len(active_names)
    print(f"Candidate Models ({M}): {active_names}")

    raw_lps = np.array([cache[f'm_{i}_unseen_lp'] for i in active_indices]) # (M, N, K)

    # Pre-clip probabilities to kill extreme negative log tails
    P_clipped = np.zeros_like(raw_lps)
    for m in range(M):
        p_m = softmax(raw_lps[m], axis=1)
        p_m = np.clip(p_m, 1e-4, 1.0)
        p_m /= p_m.sum(axis=1, keepdims=True)
        P_clipped[m] = p_m
    log_P = np.log(P_clipped)

    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(raw_lps[0], y_unseen, groups=groups))

    results_by_lam = {}
    trial_gws = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20]

    for lam in lambdas:
        oof_blended_lp = np.zeros((N, K))
        oof_post_probs = np.zeros((N, K))
        fold_weights = []
        fold_model_temps = []
        fold_global_temps = []
        fold_stds = []
        fold_gws = []

        for fold, (tr_idx, val_idx) in enumerate(splits):
            N_tr = len(tr_idx)
            y_tr = y_unseen[tr_idx]

            # 1. Pre-Calibrate Every Model Separately on Train Fold
            Z_fold = np.zeros_like(log_P)
            model_temps = np.zeros(M)
            model_stds = np.zeros(M)

            for m in range(M):
                log_p_m_tr = log_P[m, tr_idx, :]
                def nll_temp(T):
                    scaled_lp = log_softmax(log_p_m_tr / T, axis=1)
                    return -np.mean(scaled_lp[np.arange(N_tr), y_tr])
                res_t = minimize_scalar(nll_temp, bounds=(0.05, 10.0), method='bounded')
                T_opt = float(res_t.x)
                model_temps[m] = T_opt
                Z_fold[m] = log_softmax(log_P[m] / T_opt, axis=1)
                model_stds[m] = np.std(Z_fold[m, tr_idx, :])

            fold_model_temps.append(model_temps)
            fold_stds.append(model_stds)

            # 2. Setup Label Smoothed Target on Train Fold
            Y_smooth_tr = np.full((N_tr, K), eps / K)
            Y_smooth_tr[np.arange(N_tr), y_tr] += (1.0 - eps)

            # 3. Optimize Simplex Weights w = softmax(theta)
            Z_tr = Z_fold[:, tr_idx, :]
            theta0 = np.zeros(M) # uniform initialization

            res = minimize(
                fun=lambda th: get_loss_and_grad(th, Z_tr, Y_smooth_tr, lam, M),
                x0=theta0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-9}
            )
            w_k = softmax(res.x)
            fold_weights.append(w_k)

            # 4. Global Temperature Scaling on Blended Train Logits
            U_tr = np.tensordot(w_k, Z_tr, axes=(0, 0))
            def nll_global_temp(T_g):
                scaled = log_softmax(U_tr / T_g, axis=1)
                return -np.mean(scaled[np.arange(N_tr), y_tr])
            res_tg = minimize_scalar(nll_global_temp, bounds=(0.1, 5.0), method='bounded')
            T_global_opt = float(res_tg.x)
            fold_global_temps.append(T_global_opt)

            # 5. Fit Claude Gate Weight Nested Strictly Inside Train Fold
            P_tr = softmax(U_tr / T_global_opt, axis=1)
            best_gw_fold = 0.0
            best_bacc_tr = float(balanced_accuracy_score(y_tr, damped_sinkhorn(P_tr, alpha=sinkhorn_alpha).argmax(axis=1)))
            for trial_gw in trial_gws:
                P_tr_g = apply_claude_specialist_gate(
                    P_mix=P_tr,
                    R_main=P_tr,
                    claude_spec_logits=spec_logits[tr_idx],
                    claude_classes=claude_classes,
                    gate_weight=trial_gw
                )
                b_tr = float(balanced_accuracy_score(y_tr, damped_sinkhorn(P_tr_g, alpha=sinkhorn_alpha).argmax(axis=1)))
                if b_tr > best_bacc_tr:
                    best_bacc_tr = b_tr
                    best_gw_fold = trial_gw
            fold_gws.append(best_gw_fold)

            # 6. Apply to Validation Fold
            Z_val = Z_fold[:, val_idx, :]
            U_val = np.tensordot(w_k, Z_val, axes=(0, 0)) / T_global_opt
            P_val = softmax(U_val, axis=1)
            oof_blended_lp[val_idx] = log_softmax(U_val, axis=1)

            # Apply nested fold Claude gate to validation fold
            if best_gw_fold > 0:
                P_val_g = apply_claude_specialist_gate(
                    P_mix=P_val,
                    R_main=P_val,
                    claude_spec_logits=spec_logits[val_idx],
                    claude_classes=claude_classes,
                    gate_weight=best_gw_fold
                )
            else:
                P_val_g = P_val

            oof_post_probs[val_idx] = P_val_g

        # Evaluate OOF Sinkhorn Balanced Accuracy
        P_oof_raw = softmax(oof_blended_lp, axis=1)
        P_oof_sink_no_claude = damped_sinkhorn(P_oof_raw, alpha=sinkhorn_alpha)
        bacc_no_claude = float(balanced_accuracy_score(y_unseen, P_oof_sink_no_claude.argmax(axis=1)))

        P_oof_sink_nested_claude = damped_sinkhorn(oof_post_probs, alpha=sinkhorn_alpha)
        bacc_nested_claude = float(balanced_accuracy_score(y_unseen, P_oof_sink_nested_claude.argmax(axis=1)))

        avg_w = np.mean(fold_weights, axis=0)
        avg_stds = np.mean(fold_stds, axis=0)
        avg_model_temps = np.mean(fold_model_temps, axis=0)
        avg_global_temp = float(np.mean(fold_global_temps))
        avg_gw = float(np.mean(fold_gws))

        results_by_lam[lam] = {
            'bacc_no_claude': bacc_no_claude,
            'bacc_nested_claude': bacc_nested_claude,
            'avg_gw': avg_gw,
            'fold_weights': np.array(fold_weights),
            'avg_w': avg_w,
            'avg_stds': avg_stds,
            'avg_model_temps': avg_model_temps,
            'avg_global_temp': avg_global_temp,
            'P_oof_sink': P_oof_sink_nested_claude
        }
        print(f"Lambda = {lam:<6}: OOF BAcc = {bacc_no_claude:.4f} | + Nested Claude Gate (mean gw={avg_gw:.3f}) = {bacc_nested_claude:.4f} | Global T = {avg_global_temp:.3f}")

    best_lam = max(lambdas, key=lambda l: results_by_lam[l]['bacc_nested_claude'])
    best_res = results_by_lam[best_lam]
    print(f"\n>>> Best Lambda: {best_lam} | Nested OOF BAcc: {best_res['bacc_nested_claude']:.4f} (Raw Blend Sinkhorn: {best_res['bacc_no_claude']:.4f})")
    
    # Audit weights
    avg_w = best_res['avg_w']
    avg_stds = best_res['avg_stds']
    eff_contrib = avg_w * avg_stds
    eff_share = (eff_contrib / np.sum(eff_contrib)) * 100
    avg_temps = best_res['avg_model_temps']
    fold_w = best_res['fold_weights']
    
    ranks = np.argsort(-avg_w)
    header = ['Rank', 'Model Name', 'Calib T_m', 'Z_m Std', 'Nominal w_m', 'w_m * std', 'Eff Share %', 'Fold Stability']
    print(f"{header[0]:<4} | {header[1]:<23} | {header[2]:<9} | {header[3]:<7} | {header[4]:<11} | {header[5]:<9} | {header[6]:<11} | {header[7]:<14}")
    print('-' * 105)
    for r, idx in enumerate(ranks, 1):
        w_pct = avg_w[idx] * 100
        std_fold = np.std(fold_w[:, idx]) * 100
        print(f"{r:<4} | {active_names[idx]:<23} | {avg_temps[idx]:<9.2f} | {avg_stds[idx]:<7.2f} | {w_pct:6.2f}%    | {eff_contrib[idx]:<9.4f} | {eff_share[idx]:6.2f}%     | +/- {std_fold:4.2f}%")
    print('-' * 105)

    return {
        'best_lam': best_lam,
        'results_by_lam': results_by_lam,
        'active_names': active_names,
        'best_res': best_res
    }

if __name__ == '__main__':
    cache_file = 'kaggle_out_v88/local_eval_cache.npz'
    if not os.path.exists(cache_file):
        print(f"Error: {cache_file} does not exist.")
        sys.exit(1)

    print("================================================================================")
    print("EXPERIMENT 1: Full 14-Candidate Model Pool in v88 (All models except Claude Head)")
    print("================================================================================")
    res_14 = run_5fold_pooling(cache_file, model_subset_names=None)

    print("\n================================================================================")
    print("EXPERIMENT 2: 7-Model Pool in v88 (Active 6 Baseline + Gemma-2-9B)")
    print("================================================================================")
    models_7 = [
        'LinearSVC (4-View)',
        'Qwen3-14B Part-2',
        'qwen3_14b_inhouse',
        'qwen3_14b_ts',
        'qwen3_8b_v5',
        'DeBERTa-v3-Large',
        'Gemma-2-9B'
    ]
    res_7 = run_5fold_pooling(cache_file, model_subset_names=models_7)

    print("\n================================================================================")
    print("EXPERIMENT 3: Locked 6-Model Baseline Pool in v88 (Identical to Production Config)")
    print("================================================================================")
    models_6 = [
        'LinearSVC (4-View)',
        'Qwen3-14B Part-2',
        'qwen3_14b_inhouse',
        'qwen3_14b_ts',
        'qwen3_8b_v5',
        'DeBERTa-v3-Large'
    ]
    res_6 = run_5fold_pooling(cache_file, model_subset_names=models_6)
