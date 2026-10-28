"""
Automated Test -> Compile -> Kaggle Push CLI Runner
==================================================
1. Runs pytest to verify all unit/integration tests pass.
2. Compiles notebook from src/namethatllm/ + configs/v42_ensemble.json.
3. Pushes kernel to Kaggle via Kaggle CLI.
"""

import sys
import subprocess
from pathlib import Path
from compile_notebook import compile_notebook

def run_cmd(cmd: list):
    print(f"[RUNNING] {' '.join(cmd)}")
    res = subprocess.run(cmd, check=True)
    return res

def main():
    root_dir = Path(__file__).resolve().parent.parent
    config_path = root_dir / "configs" / "v42_ensemble.json"
    kernel_dir = root_dir / "kernels" / "submission"
    notebook_path = kernel_dir / "kaggle_submission.ipynb"

    print("=== Step 1: Running Pytest Verification Suite ===")
    subprocess.run([sys.executable, "-m", "pytest", "tests"], cwd=root_dir, check=True)

    print("\n=== Step 2: Compiling Single-Source-of-Truth Kaggle Notebook ===")
    compile_notebook(config_path, notebook_path)

    print(f"\n=== Step 3: Pushing Kernel to Kaggle ({kernel_dir}) ===")
    subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(kernel_dir)], cwd=root_dir, check=True)

    print("\n[SUCCESS] Kernel successfully pushed to Kaggle!")

if __name__ == "__main__":
    main()
