"""
Single-Source-of-Truth Kaggle Notebook Compiler.
Reads modular Python source files directly from `src/namethatllm/*.py` and
`configs/production_ensemble.json` to assemble `kernels/submission/kaggle_submission.ipynb`.
"""

import ast
import json
import re
import argparse
from pathlib import Path

MODULE_ORDER = [
    "config.py",
    "wpca_gaussian.py",
    "spectral_bank.py",
    "neural_extractor.py",
    "postprocess.py",
    "pipeline.py",
]

def strip_internal_imports(source: str) -> str:
    """Removes internal `from namethatllm... import ...` or `from .xxx import ...` lines, including multi-line."""
    lines = []
    in_paren_import = False
    for line in source.splitlines():
        if in_paren_import:
            if ")" in line:
                in_paren_import = False
            continue
        if re.match(r"^\s*(from\s+(\.|namethatllm)\S*\s+import\s+|import\s+namethatllm)", line):
            if "(" in line and ")" not in line:
                in_paren_import = True
            continue
        lines.append(line)
    return "\n".join(lines).strip() + "\n"

def compile_notebook(config_path: Path, output_path: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    pkg_dir = repo_root / "src" / "namethatllm"
    cfg_data = json.loads(config_path.read_text(encoding="utf-8"))

    # Cell 1: Write isolated subprocess script `/kaggle/working/run_lexical.py` directly from `lexical_engine.py`
    lex_src = strip_internal_imports((pkg_dir / "lexical_engine.py").read_text(encoding="utf-8"))
    ast.parse(lex_src, filename="lexical_engine.py")
    cell_lexical = f"%%writefile /kaggle/working/run_lexical.py\n{lex_src}"

    # Cell 2+: Read each package module in dependency order
    cells_code = [cell_lexical]
    for mod_name in MODULE_ORDER:
        mod_path = pkg_dir / mod_name
        raw_src = strip_internal_imports(mod_path.read_text(encoding="utf-8"))
        if mod_name == "config.py":
            raw_src = f"ENSEMBLE_CONFIG = {repr(cfg_data)}\n\n" + raw_src
        ast.parse(raw_src, filename=mod_name)
        cells_code.append(f"# === MODULE: src/namethatllm/{mod_name} ===\n{raw_src}")

    # Final Cell: Entrypoint call
    entry_code = "if __name__ == '__main__':\n    run_submission_pipeline(ENSEMBLE_CONFIG)\n"
    ast.parse(entry_code, filename="entrypoint")
    cells_code.append(entry_code)

    # Build Jupyter .ipynb JSON structure
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in code.splitlines()],
            }
            for code in cells_code
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile Kaggle submission notebook from src/namethatllm")
    parser.add_argument("--config", default="configs/production_ensemble.json", help="Path to ensemble config")
    parser.add_argument("--output", default="kernels/submission/kaggle_submission.ipynb", help="Output .ipynb path")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    cfg_p = Path(args.config) if Path(args.config).is_absolute() else repo / args.config
    out_p = Path(args.output) if Path(args.output).is_absolute() else repo / args.output
    out = compile_notebook(cfg_p, out_p)
    print(f"[Compiler] Successfully compiled notebook to {out}")
