import json
import os
import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score

from namethatllm.ablation import pool_calibrated_log_probs
from namethatllm.postprocess import apply_claude_specialist_gate, damped_sinkhorn


def predict_from_artifact(lps_list, art, spec_log=None):
    """Reproduce calibrated inference blending using canonical namethatllm functions."""
    P_blend = pool_calibrated_log_probs(
        lps_list,
        art["model_names"],
        art["mean_T_m"],
        art["mean_w_m"],
        art["mean_T_global"],
        clip_min=float(art.get("clip_min", 1e-4)),
    )
    if spec_log is not None and float(art.get("mean_gw", 0.0)) > 0.0:
        P_blend = apply_claude_specialist_gate(
            P_mix=P_blend,
            R_main=P_blend,
            claude_spec_logits=spec_log,
            claude_classes=art["claude_classes"],
            gate_weight=float(art["mean_gw"]),
        )
    return damped_sinkhorn(P_blend, alpha=float(art.get("sinkhorn_alpha", 0.50)))


def _load_bge_v2_blend_logits(bge_path: str, y_unseen: np.ndarray) -> np.ndarray:
    with np.load(bge_path, allow_pickle=True) as z:
        assert np.array_equal(y_unseen, z["y_bench"])
        ln = z["bench_neural_logits"].astype(np.float64)
        lr = z["bench_ridge_logits"].astype(np.float64)
    lr_scaled = (lr - lr.mean()) * (ln.std() / max(float(lr.std()), 1e-12))
    return 0.35 * ln + 0.65 * lr_scaled


def test_calibration_artifact_matrix_parity():
    cache_path = (
        "kaggle_out_v88/local_eval_cache.npz"
        if os.path.exists("kaggle_out_v88/local_eval_cache.npz")
        else "kaggle_out_v87/local_eval_cache.npz"
    )
    bge_path = "reports/bge_v2_stage1_preds.npz"
    config_path = "configs/production_ensemble.json"
    if not os.path.exists(cache_path) or not os.path.exists(config_path):
        pytest.skip("Cache or config not found")

    with open(config_path, encoding="utf-8") as f:
        artifact = json.load(f)

    with np.load(cache_path, allow_pickle=True) as cache:
        y_unseen = cache["y_unseen"]
        spec_logits = cache["claude_spec_logits_u"]
        model_names_all = list(cache["model_names"])
        lp_by_name = {name: cache[f"m_{i}_unseen_lp"] for i, name in enumerate(model_names_all)}

    if "BGE-M3-v2" in artifact["model_names"]:
        if not os.path.exists(bge_path):
            pytest.skip("BGE-M3 v2 stage-1 predictions file not found")
        lp_by_name["BGE-M3-v2"] = _load_bge_v2_blend_logits(bge_path, y_unseen)

    raw_lps = [lp_by_name[name] for name in artifact["model_names"]]

    P_inf_1 = predict_from_artifact(raw_lps, artifact, spec_logits)
    P_inf_2 = predict_from_artifact(raw_lps, artifact, spec_logits)

    assert np.allclose(P_inf_1.sum(axis=1), 1.0, atol=1e-6), "Row sums do not equal 1.0!"
    assert np.allclose(P_inf_1, P_inf_2, atol=1e-6), "Element-by-element probability mismatch!"

    bacc = balanced_accuracy_score(y_unseen, P_inf_1.argmax(axis=1))
    assert bacc > 0.595, f"Accuracy {bacc:.4f} is below 0.595 threshold!"
