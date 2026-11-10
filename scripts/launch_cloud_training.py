"""
NameThatLLM - Detached Cloud Training Launcher (Modal NVIDIA A100-80GB)
======================================================================
Deploys 'ntl-inhouse-trainer' to Modal and asynchronously spawns the full
18-class champion Qwen3-14B training pipeline.

Because this launcher uses Modal's serverless .spawn() execution model,
the local process exits immediately with exit code 0.
The training job runs 100% cloud-side on Modal, completely immune to:
  - Laptop shutdown / sleep / power-off
  - Wi-Fi / DNS disconnection
  - Terminal closure
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    import modal
except ImportError:
    print("Error: 'modal' package is not installed. Run: pip install modal")
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch detached NameThatLLM training on Modal A100 GPU")
    parser.add_argument(
        "--config",
        default="configs/training_lora.json",
        help="Path to training config json (default: configs/training_lora.json)",
    )
    parser.add_argument(
        "--no-deploy",
        action="store_true",
        help="Skip deploying the latest code before spawning",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    modal_trainer_path = repo_root / "scripts" / "modal_trainer.py"

    if not args.no_deploy:
        print("[Launcher] Deploying latest code and configs to Modal ('ntl-inhouse-trainer')...")
        res = subprocess.run(
            [sys.executable, "-m", "modal", "deploy", str(modal_trainer_path)],
            check=False,
        )
        if res.returncode != 0:
            print("[Launcher Warning] Modal deploy exited with non-zero status. Attempting spawn anyway...")

    print(f"[Launcher] Looking up deployed function 'train_cloud_worker'...")
    try:
        f = modal.Function.from_name("ntl-inhouse-trainer", "train_cloud_worker")
    except Exception as exc:
        print(f"[Launcher Error] Could not connect to deployed function: {exc}")
        sys.exit(1)

    print(f"[Launcher] Spawning asynchronous training task on NVIDIA A100-80GB GPU...")
    print(f"           Config: {args.config}")
    call = f.spawn(config_path=args.config)

    print("\n" + "=" * 78)
    print("🚀 SUCCESS! Champion Qwen3-14B Training Job Dispatched to Cloud GPU!")
    print("=" * 78)
    print(f"  Modal Function Call ID : {call.object_id}")
    print(f"  App Name               : ntl-inhouse-trainer")
    print(f"  Target Hardware        : NVIDIA A100-80GB (Ampere, BF16 + FlashAttention-2)")
    print(f"  Execution Budget       : 12.0 Hours (43,200s timeout)")
    print(f"  Auto-Save Checkpoints  : Every 500 steps to 'ntl-checkpoints' volume")
    print("=" * 78)
    print("STATUS: DETACHED AND INDEPENDENT.")
    print("  You can now safely shut down your laptop, close your IDE, or disconnect Wi-Fi.")
    print("  The model training will continue uninterrupted in the cloud until finished.")
    print("\nTo inspect live logs from any terminal:")
    print("  python -m modal app logs ntl-inhouse-trainer")
    print("\nTo view GPU telemetry on the web:")
    print("  https://modal.com/apps")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
