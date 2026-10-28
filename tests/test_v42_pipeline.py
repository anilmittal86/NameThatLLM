"""
Name That LLM - Version 42 Test Suite
=====================================
Tests:
1. Pure Two-Tier Separation (85% Main Stack + 15% Complementary Stack).
2. Dual-Output Lexical Engine outputs (log_p_svc_main and p_lex_comp).
3. Unsupervised 64-Dim Spectral Bank Extraction (Zero Label Leakage) & Prior Scaling.
4. Ledoit-Wolf Shrinkage Floors (se_floor = 0.10 for Part 2, se_floor = 0.35 for others).
5. Mass-Preserving Claude Specialist Gate within Claude Family.
6. Gentle Curly-Quote Shield with Simplex Normalization.
7. Official 300-Step Exact Multiplicative Sinkhorn Balancing.
8. Ast Syntax Validation of Notebook Code Cells & Subprocess Script Template.
"""

import unittest
import numpy as np
import json
import ast
import re
from pathlib import Path
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.covariance import ledoit_wolf_shrinkage

LABELS = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5", "claude-fable-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3"
]
K = len(LABELS)
LAB2I = {l: i for i, l in enumerate(LABELS)}

FAMILIES = {
    "gpt": [0, 1, 2, 3, 4],
    "gemini": [5, 6, 7, 8],
    "claude": [9, 10, 11, 12, 13],
    "glm": [15, 16],
}

def align_cols(arr: np.ndarray, src_labels: list) -> np.ndarray:
    src_list = [x.decode('utf-8') if isinstance(x, (bytes, np.bytes_)) else str(x) for x in src_labels]
    idx = [src_list.index(l) for l in LABELS]
    if arr.ndim == 1:
        return arr[idx]
    elif arr.ndim == 2:
        if arr.shape[0] == len(src_list) and arr.shape[1] != len(src_list):
            return arr[idx, :]
        elif arr.shape[1] == len(src_list):
            return arr[:, idx]
    return arr[..., idx]

def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    c = np.max(x, axis=axis, keepdims=True)
    return x - c - np.log(np.sum(np.exp(x - c), axis=axis, keepdims=True) + 1e-15)

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    c = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - c)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-15)

def sinkhorn_balancing(P: np.ndarray, iters: int = 300) -> np.ndarray:
    if len(P) < 50:
        return P
    P = np.clip(P.astype(np.float64), 1e-12, 1.0)
    target_col = 1.0 / P.shape[1]
    for _ in range(iters):
        P /= P.sum(axis=1, keepdims=True).clip(min=1e-12)
        P /= (P.sum(axis=0, keepdims=True) / (target_col * P.shape[0])).clip(min=1e-12)
    P /= P.sum(axis=1, keepdims=True).clip(min=1e-12)
    return P

def _two_way(Z: np.ndarray, y: np.ndarray, pinv: np.ndarray, G: int, K: int = 18, iters: int = 4):
    N, d = Z.shape
    mu = np.zeros((K, d), dtype=np.float64)
    P = np.zeros((G, d), dtype=np.float64)
    ng = np.bincount(pinv, minlength=G).clip(min=1)
    for _ in range(iters):
        R = Z - P[pinv]
        for k in range(K):
            idx = (y == k)
            if idx.any():
                mu[k] = R[idx].mean(axis=0)
        R = Z - mu[y]
        P[:] = 0.0
        np.add.at(P, pinv, R)
        P /= ng[:, None]
    return mu, P, ng

def _shrunk(S: np.ndarray, alpha: float) -> np.ndarray:
    d = S.shape[0]
    tr = np.trace(S) / d
    return (1.0 - alpha) * S + alpha * tr * np.eye(d)

def fit_shared_gaussian_lowram(X_tr_f32: np.ndarray, y: np.ndarray, prompts: np.ndarray,
                               d_target: int = 256, se_floor: float = 0.10) -> dict:
    N, D = X_tr_f32.shape
    pu, pinv = np.unique(prompts, return_inverse=True)
    G = len(pu)
    d = min(d_target, N - G, D)

    mx = X_tr_f32.mean(axis=0, keepdims=True)
    sx = X_tr_f32.std(axis=0, keepdims=True)
    sx[sx < 1e-6] = 1.0

    R = ((X_tr_f32 - mx) / sx).astype(np.float32)
    Gram = (R @ R.T) / float(D)
    lam_r, U = np.linalg.eigh(Gram.astype(np.float64))
    idx_top = np.argsort(lam_r)[::-1][:d]
    lam_top = np.maximum(lam_r[idx_top], 1e-9)
    U_top = U[:, idx_top]
    U_f32 = U_top.astype(np.float32)

    V_f32 = (R.T @ U_f32) / (np.sqrt(lam_top.astype(np.float32) * float(D))[None, :])
    del R, Gram, U, U_f32

    V_top64 = V_f32[:, :min(64, d)].copy()
    Z = (((X_tr_f32 - mx) / sx) @ V_f32).astype(np.float64)

    mu, P, ng = _two_way(Z, y, pinv, G, K=18, iters=4)
    E = Z - mu[y] - P[pinv]
    dof = max(N - G - len(np.unique(y)) + 1, 1)
    Se_raw = (E.T @ E) / dof
    ae = max(float(ledoit_wolf_shrinkage(E, assume_centered=True)), se_floor)
    Se = _shrunk(Se_raw, ae)

    Pc = P - P.mean(0)
    Sp_raw = (Pc.T @ Pc) / max(G - 1, 1) - np.mean(1.0 / ng) * Se_raw
    ap = float(ledoit_wolf_shrinkage(Pc, assume_centered=True))
    tr = max(np.trace(Sp_raw), 1e-8 * np.trace(Se))
    Sp = (1.0 - ap) * Sp_raw + ap * (tr / d) * np.eye(d)

    L = np.linalg.cholesky(Se)
    Li = np.linalg.inv(L)
    M = 0.5 * ((Li @ Sp @ Li.T) + (Li @ Sp @ Li.T).T)
    lam, Q = np.linalg.eigh(M)
    lam = np.clip(lam, 1e-4, None)
    A = Q.T @ Li
    W_f32 = (V_f32 @ A.T.astype(np.float32)).astype(np.float32)
    m_whitened = mu @ A.T

    return {
        "mx": mx, "sx": sx, "W": W_f32, "m": m_whitened, "lam": lam,
        "V_top64": V_top64, "T1": 2.40, "Tg": 2.35, "ae": ae
    }


class TestVersion42Pipeline(unittest.TestCase):
    def test_two_tier_separation_blend(self):
        """Verify that 0.85 * R_main + 0.15 * P_comp strictly preserves probability simplex."""
        np.random.seed(42)
        N = 50
        logits_main = np.random.randn(N, 18)
        R_main = softmax(logits_main, axis=1)

        p_comp1 = softmax(np.random.randn(N, 18), axis=1)
        p_comp2 = softmax(np.random.randn(N, 18), axis=1)
        p_comp3 = softmax(np.random.randn(N, 18), axis=1)

        comp_list = [(0.35, p_comp1), (0.20, p_comp2), (0.30, p_comp3)]
        comp_weight_total = sum(w for w, _ in comp_list)
        P_comp = sum(w * P_c for w, P_c in comp_list) / comp_weight_total

        # Complementary stack is stochastic
        self.assertTrue(np.allclose(P_comp.sum(axis=1), 1.0))

        P_mix = 0.85 * R_main + 0.15 * P_comp
        # Mixture is strictly stochastic
        self.assertTrue(np.allclose(P_mix.sum(axis=1), 1.0))
        self.assertTrue((P_mix >= 0.0).all())
        self.assertTrue((P_mix <= 1.0).all())

    def test_unsupervised_spectral_bank_extraction_no_leakage(self):
        """Verify that V_top64 does NOT use class labels y, preventing leakage."""
        np.random.seed(42)
        N = 180
        D = 128
        X_tr = np.random.randn(N, D).astype(np.float32)
        prompts = np.repeat(np.arange(10), 18)
        y_true = np.tile(np.arange(18), 10)
        y_shuffled = np.random.permutation(y_true)

        prm1 = fit_shared_gaussian_lowram(X_tr, y_true, prompts, d_target=64, se_floor=0.10)
        prm2 = fit_shared_gaussian_lowram(X_tr, y_shuffled, prompts, d_target=64, se_floor=0.10)

        # V_top64 is purely unsupervised and must be identical regardless of y!
        self.assertTrue(np.allclose(prm1["V_top64"], prm2["V_top64"], atol=1e-6))

        # Check prior scaling
        scale = 0.85
        S_tr_64 = (((X_tr - prm1["mx"]) / prm1["sx"]) @ prm1["V_top64"]).astype(np.float32)
        S_norm = (S_tr_64 / np.linalg.norm(S_tr_64, axis=1, keepdims=True).clip(min=1e-6)) * scale

        row_norms = np.linalg.norm(S_norm, axis=1)
        self.assertTrue(np.allclose(row_norms, scale, atol=1e-5))

    def test_ledoit_wolf_shrinkage_floor(self):
        """Verify that se_floor is strictly respected."""
        np.random.seed(42)
        N = 180
        D = 128
        X_tr = np.random.randn(N, D).astype(np.float32)
        prompts = np.repeat(np.arange(10), 18)
        y = np.tile(np.arange(18), 10)

        prm_floor_035 = fit_shared_gaussian_lowram(X_tr, y, prompts, d_target=64, se_floor=0.35)
        self.assertGreaterEqual(prm_floor_035["ae"], 0.35)

        prm_floor_010 = fit_shared_gaussian_lowram(X_tr, y, prompts, d_target=64, se_floor=0.10)
        self.assertGreaterEqual(prm_floor_010["ae"], 0.10)

    def test_gentle_claude_specialist_gate_mass_preservation(self):
        """Verify that the Claude specialist gate preserves exact total probability mass."""
        np.random.seed(42)
        N = 20
        claude_cols = FAMILIES["claude"]
        P_mix = softmax(np.random.randn(N, 18), axis=1)
        P_claude_head = softmax(np.random.randn(N, len(claude_cols)), axis=1)

        # Passage 0: predict Claude opus
        pred_main = np.zeros(N, dtype=int)
        pred_main[0] = claude_cols[0]
        # Passage 1: predict GPT
        pred_main[1] = 0

        claude_mask = np.isin(pred_main, claude_cols)

        orig_mass = P_mix.sum(axis=1)
        orig_claude_mass = P_mix[:, claude_cols].sum(axis=1)

        if claude_mask.any():
            for i in np.where(claude_mask)[0]:
                tot_mass = P_mix[i, claude_cols].sum()
                blended = 0.85 * P_mix[i, claude_cols] + 0.15 * (tot_mass * P_claude_head[i])
                P_mix[i, claude_cols] = blended

        new_mass = P_mix.sum(axis=1)
        new_claude_mass = P_mix[:, claude_cols].sum(axis=1)

        # Total mass remains 1.0
        self.assertTrue(np.allclose(new_mass, 1.0))
        # Total mass within Claude family is PRESERVED exactly
        self.assertTrue(np.allclose(new_claude_mass, orig_claude_mass))

    def test_gentle_curly_quote_shield(self):
        """Verify curly quote guard applies np.exp(-1.0) and normalizes rows."""
        claude_and_glm_idx = FAMILIES["claude"] + FAMILIES["glm"]
        P_mix = np.ones((2, 18), dtype=np.float64) / 18.0

        test_texts = ["Regular text with no curly quotes.", "Special text with \u201ccurly\u201d quotes."]
        for i, txt in enumerate(test_texts):
            txt_s = str(txt)
            if any(q in txt_s for q in ['\u2018', '\u2019', '\u201c', '\u201d']):
                P_mix[i, claude_and_glm_idx] *= np.exp(-1.0)

        P_mix /= P_mix.sum(axis=1, keepdims=True).clip(min=1e-12)

        # Row 0 (no quotes): uniform
        self.assertTrue(np.allclose(P_mix[0], 1.0 / 18.0))
        # Row 1 (quotes): Claude/GLM classes are dampened
        dampened = P_mix[1, claude_and_glm_idx[0]]
        undampened = P_mix[1, 0]
        self.assertLess(dampened, undampened)
        self.assertTrue(np.allclose(P_mix.sum(axis=1), 1.0))

    def test_sinkhorn_exact_marginal_equalization(self):
        np.random.seed(42)
        N = 180
        logits = np.random.randn(N, 18)
        P = softmax(logits, axis=1)
        P_sink = sinkhorn_balancing(P, iters=300)

        self.assertTrue(np.allclose(P_sink.sum(axis=1), 1.0))
        target = N / 18.0
        col_sums = P_sink.sum(axis=0)
        self.assertTrue(np.allclose(col_sums, target, atol=1e-10))

    def test_notebook_ast_and_subprocess_syntax(self):
        """Verify that all code cells in kaggle_submission.ipynb are syntactically valid Python."""
        nb_path = Path("kaggle_submission.ipynb")
        self.assertTrue(nb_path.exists())
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        found_subprocess = False
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                code_text = "".join(cell["source"])
                # Parse AST
                parsed = ast.parse(code_text)
                self.assertIsNotNone(parsed)

                # Check for run_lexical.py script template
                if "run_lexical.py" in code_text and "lexical_code = f\"\"\"" in code_text:
                    found_subprocess = True
                    m = re.search(r'lexical_code\s*=\s*f?"""(.*?)"""', code_text, re.DOTALL)
                    if m:
                        sub_code = m.group(1)
                        sub_code_clean = re.sub(r'\{[a-zA-Z0-9_]+\}', '"dummy"', sub_code)
                        sub_code_clean = sub_code_clean.replace('{{', '{').replace('}}', '}')
                        parsed_sub = ast.parse(sub_code_clean)
                        self.assertIsNotNone(parsed_sub)

        self.assertTrue(found_subprocess)


if __name__ == "__main__":
    unittest.main()
