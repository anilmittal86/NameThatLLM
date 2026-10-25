"""
Configuration module for Name That LLM (18-Class Authorship Attribution).
Defines paths, 18 model targets, 6 provider families, hardware guards, and seeds.
"""

import os
from pathlib import Path
from typing import Dict, List

# -----------------------------------------------------------------------------
# 1. Random Seed & Paths
# -----------------------------------------------------------------------------
RANDOM_SEED: int = 42

# Root directory resolution
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = WORKSPACE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset path resolution (local vs Kaggle)
CANDIDATE_TRAIN_PATHS = [
    WORKSPACE_DIR / "train.csv",
    Path("/kaggle/input/competitions/name-that-llm/train.csv"),
    Path("/kaggle/input/name-that-llm/train.csv"),
]

CANDIDATE_TEST_PATHS = [
    WORKSPACE_DIR / "test.csv",
    Path("/kaggle/input/competitions/name-that-llm/test.csv"),
    Path("/kaggle/input/name-that-llm/test.csv"),
]

CANDIDATE_SAMPLE_SUB_PATHS = [
    WORKSPACE_DIR / "sample_submission.parquet",
    Path("/kaggle/input/competitions/name-that-llm/sample_submission.parquet"),
    Path("/kaggle/input/name-that-llm/sample_submission.parquet"),
]


def resolve_existing_path(candidates: List[Path], default: Path) -> Path:
    for p in candidates:
        if p.exists():
            return p
    return default


TRAIN_PATH = resolve_existing_path(CANDIDATE_TRAIN_PATHS, WORKSPACE_DIR / "train.csv")
TEST_PATH = resolve_existing_path(CANDIDATE_TEST_PATHS, WORKSPACE_DIR / "test.csv")
SAMPLE_SUB_PATH = resolve_existing_path(
    CANDIDATE_SAMPLE_SUB_PATHS, WORKSPACE_DIR / "sample_submission.parquet"
)
SUBMISSION_PATH = WORKSPACE_DIR / "submission.parquet"

# -----------------------------------------------------------------------------
# 2. 18 Target Model Labels
# -----------------------------------------------------------------------------
TARGET_MODELS: List[str] = [
    "claude-fable-5",
    "claude-opus-4-6",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "glm-5.3",
    "glm-5.3-flash",
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
    "kimi-k3",
    "qwen3.8-max",
]

NUM_CLASSES: int = len(TARGET_MODELS)
assert NUM_CLASSES == 18, f"Expected 18 classes, got {NUM_CLASSES}"

MODEL_TO_IDX: Dict[str, int] = {m: i for i, m in enumerate(TARGET_MODELS)}
IDX_TO_MODEL: Dict[int, str] = {i: m for i, m in enumerate(TARGET_MODELS)}

# -----------------------------------------------------------------------------
# 3. 6 Provider / Family Mappings
# -----------------------------------------------------------------------------
PROVIDER_FAMILIES: Dict[str, List[str]] = {
    "claude": [
        "claude-fable-5",
        "claude-opus-4-6",
        "claude-opus-4-8",
        "claude-opus-5",
        "claude-sonnet-5",
    ],
    "gemini": [
        "gemini-3.1-pro-preview",
        "gemini-3.6-flash",
        "gemini-3.7-flash",
        "gemini-3.8-flash",
    ],
    "glm": [
        "glm-5.3",
        "glm-5.3-flash",
    ],
    "gpt": [
        "gpt-5.5",
        "gpt-5.6-luna",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-6-astra",
    ],
    "kimi": [
        "kimi-k3",
    ],
    "qwen": [
        "qwen3.8-max",
    ],
}

FAMILY_NAMES: List[str] = sorted(list(PROVIDER_FAMILIES.keys()))
NUM_FAMILIES: int = len(FAMILY_NAMES)
assert NUM_FAMILIES == 6, f"Expected 6 families, got {NUM_FAMILIES}"

FAMILY_TO_IDX: Dict[str, int] = {f: i for i, f in enumerate(FAMILY_NAMES)}
IDX_TO_FAMILY: Dict[int, str] = {i: f for i, f in enumerate(FAMILY_NAMES)}


def get_model_family(model_name: str) -> str:
    """Returns the provider family for a given model string."""
    for family, models in PROVIDER_FAMILIES.items():
        if model_name in models:
            return family
    raise ValueError(f"Unknown model name: {model_name}")


MODEL_TO_FAMILY: Dict[str, str] = {m: get_model_family(m) for m in TARGET_MODELS}
MODEL_TO_FAMILY_IDX: Dict[str, int] = {
    m: FAMILY_TO_IDX[MODEL_TO_FAMILY[m]] for m in TARGET_MODELS
}

# Within-family indices (for specialist training)
FAMILY_MODEL_INDICES: Dict[str, List[int]] = {
    f: [MODEL_TO_IDX[m] for m in models] for f, models in PROVIDER_FAMILIES.items()
}

# -----------------------------------------------------------------------------
# 4. Hardware & Inference Constraints
# -----------------------------------------------------------------------------
# Kaggle 2xT4 GPU environment does not support native bfloat16.
# Inference must always default to fp16 when CUDA is available, with fp32 fallback.
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_FP16 = torch.cuda.is_available()  # fp16 on T4 GPUs, fp32 on CPU
DTYPE = torch.float16 if USE_FP16 else torch.float32

if __name__ == "__main__":
    print(f"Config loaded successfully.")
    print(f"Workspace: {WORKSPACE_DIR}")
    print(f"Artifacts: {ARTIFACTS_DIR}")
    print(f"Classes ({NUM_CLASSES}): {TARGET_MODELS}")
    print(f"Families ({NUM_FAMILIES}): {FAMILY_NAMES}")
    print(f"Device: {DEVICE}, Dtype: {DTYPE}")
