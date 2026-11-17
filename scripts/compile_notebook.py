"""
Single-Source-of-Truth Kaggle Notebook Compiler.
Reads modular Python source files directly from `src/namethatllm/*.py` and
config JSONs to assemble `kaggle_submission.ipynb` and `lora_training.ipynb`.
"""

import ast
import json
import re
import argparse
from pathlib import Path
from typing import List

SUBMISSION_MODULES = [
    "config.py",
    "wpca_gaussian.py",
    "spectral_bank.py",
    "neural_extractor.py",
    "postprocess.py",
    "ablation.py",
    "pipeline.py",
]

TRAINING_MODULES = [
    "config.py",
    "wpca_gaussian.py",
    "neural_extractor.py",
    "ablation.py",
    "trainer.py",
]

def strip_internal_imports(source: str) -> str:
    """Removes internal `from namethatllm... import ...` or `from .xxx import ...` lines, including multi-line."""
    lines, in_paren = [], False
    for line in source.splitlines():
        if in_paren:
            if ")" in line:
                in_paren = False
            continue
        if re.match(r"^\s*(from\s+(\.|namethatllm)\S*\s+import\s+|import\s+namethatllm)", line):
            if "(" in line and ")" not in line:
                in_paren = True
            continue
        lines.append(line)
    return "\n".join(lines).strip() + "\n"

def _build_notebook(cells_code: List[str], output_path: Path) -> Path:
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "id": f"cell_{idx}",
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in code.splitlines()]
            }
            for idx, code in enumerate(cells_code)
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    return output_path

def compile_submission_notebook(config_path: Path, output_path: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    pkg_dir = repo_root / "src" / "namethatllm"
    cfg_data = json.loads(config_path.read_text(encoding="utf-8"))

    lex_src = strip_internal_imports((pkg_dir / "lexical_engine.py").read_text(encoding="utf-8"))
    ast.parse(lex_src, filename="lexical_engine.py")
    cells = [f"%%writefile /kaggle/working/run_lexical.py\n{lex_src}"]

    for mod in SUBMISSION_MODULES:
        raw = strip_internal_imports((pkg_dir / mod).read_text(encoding="utf-8"))
        if mod == "config.py":
            raw = f"ENSEMBLE_CONFIG = {repr(cfg_data)}\n\n" + raw
        ast.parse(raw, filename=mod)
        cells.append(f"# === MODULE: src/namethatllm/{mod} ===\n{raw}")

    entry = "if __name__ == '__main__':\n    run_submission_pipeline(ENSEMBLE_CONFIG)\n"
    ast.parse(entry, filename="entrypoint")
    cells.append(entry)
    return _build_notebook(cells, output_path)

compile_notebook = compile_submission_notebook

def compile_training_notebook(config_path: Path, output_path: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    pkg_dir = repo_root / "src" / "namethatllm"
    cfg_data = json.loads(config_path.read_text(encoding="utf-8"))

    cells = []
    for mod in TRAINING_MODULES:
        raw = strip_internal_imports((pkg_dir / mod).read_text(encoding="utf-8"))
        if mod == "config.py":
            raw = f"TRAINING_CONFIG = {repr(cfg_data)}\n\n" + raw
        ast.parse(raw, filename=mod)
        cells.append(f"# === MODULE: src/namethatllm/{mod} ===\n{raw}")

    entry = "if __name__ == '__main__':\n    run_training_pipeline(TRAINING_CONFIG)\n"
    ast.parse(entry, filename="entrypoint_training")
    cells.append(entry)
    return _build_notebook(cells, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile Kaggle notebooks from src/namethatllm")
    parser.add_argument("--target", choices=["submission", "training", "all"], default="all")
    parser.add_argument("--config", default="configs/production_ensemble.json")
    parser.add_argument("--output", default="kernels/submission/kaggle_submission.ipynb")
    parser.add_argument("--training-config", default="configs/training_lora.json")
    parser.add_argument("--training-output", default="kernels/training/lora_training.ipynb")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    if args.target in ("submission", "all"):
        out_sub = compile_submission_notebook(repo / args.config, repo / args.output)
        print(f"[Compiler] Successfully compiled submission notebook to {out_sub}")
    if args.target in ("training", "all"):
        out_trn = compile_training_notebook(repo / args.training_config, repo / args.training_output)
        print(f"[Compiler] Successfully compiled training notebook to {out_trn}")
