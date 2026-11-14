"""
Automated Experiment Ledger and Benchmark Progression Tracker
============================================================
Logs multi-phase gate and ensemble experiments across Kaggle GPUs and local environments.
Maintains synchronized records in:
  - experiments/experiment_ledger.csv
  - experiments/experiment_ledger.md
"""

from __future__ import annotations

import csv
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


LEDGER_COLUMNS = [
    "exp_id",
    "timestamp",
    "phase",
    "target_family",
    "feature_set",
    "subspace_dim",
    "head_type",
    "gate_weight",
    "oof_family_bacc",
    "holdout_family_bacc",
    "hard_sister_acc",
    "holdout_18cls_raw_bacc",
    "holdout_18cls_sinkhorn_bacc",
    "delta_vs_baseline_bacc",
    "notes",
]


class ExperimentLedger:
    """
    Structured ledger for logging feature extraction, gate training,
    and ensemble optimization experiments.
    """

    def __init__(
        self,
        ledger_dir: Path | str = "experiments",
        baseline_bacc: float | None = None,
    ):
        self.ledger_dir = Path(ledger_dir)
        self.ledger_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.ledger_dir / "experiment_ledger.csv"
        self.md_path = self.ledger_dir / "experiment_ledger.md"
        self.baseline_bacc = baseline_bacc

        # Initialize CSV if not present
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(LEDGER_COLUMNS)

    def add_entry(self, record: dict[str, Any]) -> dict[str, Any]:
        """
        Appends an experiment record to the CSV and regenerates the Markdown table.
        """
        entry = dict(record)

        if "timestamp" not in entry or not entry["timestamp"]:
            entry["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Automatically record baseline if this is EXP-00 or F0_plain_solo
        if entry.get("feature_set") == "F0_plain_solo" or entry.get("exp_id") == "EXP-00":
            if self.baseline_bacc is None and "holdout_18cls_sinkhorn_bacc" in entry:
                self.baseline_bacc = float(entry["holdout_18cls_sinkhorn_bacc"])

        if "delta_vs_baseline_bacc" not in entry or entry["delta_vs_baseline_bacc"] is None:
            if self.baseline_bacc is not None and "holdout_18cls_sinkhorn_bacc" in entry:
                try:
                    entry["delta_vs_baseline_bacc"] = float(entry["holdout_18cls_sinkhorn_bacc"]) - float(self.baseline_bacc)
                except (TypeError, ValueError):
                    entry["delta_vs_baseline_bacc"] = 0.0
            else:
                entry["delta_vs_baseline_bacc"] = 0.0

        # Standardize missing columns
        for col in LEDGER_COLUMNS:
            if col not in entry or entry[col] is None:
                entry[col] = ""

        # Append to CSV
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([entry.get(c, "") for c in LEDGER_COLUMNS])

        # Regenerate Markdown
        self._regenerate_markdown()
        return entry

    def get_dataframe(self) -> pd.DataFrame:
        """Loads all logged experiments as a pandas DataFrame."""
        if not self.csv_path.exists():
            return pd.DataFrame(columns=LEDGER_COLUMNS)
        try:
            return pd.read_csv(self.csv_path)
        except Exception:
            return pd.DataFrame(columns=LEDGER_COLUMNS)

    def get_best_run(
        self,
        target_family: str | None = None,
        metric: str = "holdout_18cls_sinkhorn_bacc",
    ) -> dict[str, Any] | None:
        """Finds the best run matching criteria by given metric."""
        df = self.get_dataframe()
        if df.empty:
            return None
        if target_family is not None:
            df = df[df["target_family"].str.lower() == target_family.lower()]
        if df.empty or metric not in df.columns:
            return None

        # Convert metric column to numeric
        valid_rows = df.dropna(subset=[metric])
        if valid_rows.empty:
            return None
        best_idx = pd.to_numeric(valid_rows[metric], errors="coerce").idxmax()
        if pd.isna(best_idx):
            return None
        return df.loc[best_idx].to_dict()

    def export_copy(self, dest_dir: Path | str) -> None:
        """Copies the CSV and Markdown files to dest_dir (for Kaggle output artifacts)."""
        d = Path(dest_dir)
        d.mkdir(parents=True, exist_ok=True)
        if self.csv_path.exists():
            shutil.copy2(str(self.csv_path), str(d / "experiment_ledger.csv"))
        if self.md_path.exists():
            shutil.copy2(str(self.md_path), str(d / "experiment_ledger.md"))

    def _regenerate_markdown(self) -> None:
        """Formats the entire CSV history into a GitHub-Flavored Markdown table."""
        df = self.get_dataframe()
        if df.empty:
            self.md_path.write_text("# NameThatLLM Experiment Ledger\n\nNo experiments logged yet.\n", encoding="utf-8")
            return

        lines = [
            "# NameThatLLM Gate & Ensemble Experiment Ledger",
            f"*Last Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}*",
            "",
            "| Exp ID | Phase | Family | Feature Set | D_fish | Head | GW | Family OOF | Family Holdout | Sister Acc | 18c Raw | 18c Sinkhorn | Δ vs Base | Notes |",
            "|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]

        def _fmt_pct(v: Any) -> str:
            if v is None or v == "" or pd.isna(v):
                return "-"
            try:
                val = float(v)
                return f"{val * 100:.2f}%" if val <= 1.0 else f"{val:.2f}%"
            except (ValueError, TypeError):
                return str(v)

        def _fmt_delta(v: Any) -> str:
            if v is None or v == "" or pd.isna(v):
                return "+0.00%"
            try:
                val = float(v)
                pct = val * 100 if abs(val) <= 1.0 else val
                return f"{pct:+.2f}%"
            except (ValueError, TypeError):
                return str(v)

        for _, row in df.iterrows():
            exp_id = str(row.get("exp_id", ""))
            phase = str(row.get("phase", ""))
            fam = str(row.get("target_family", ""))
            feat = str(row.get("feature_set", ""))
            sub_d = str(row.get("subspace_dim", "")) or "-"
            head = str(row.get("head_type", ""))
            gw = f"{float(row['gate_weight']):.2f}" if row.get("gate_weight") != "" and not pd.isna(row.get("gate_weight")) else "-"
            oof_fam = _fmt_pct(row.get("oof_family_bacc"))
            hld_fam = _fmt_pct(row.get("holdout_family_bacc"))
            sis_acc = _fmt_pct(row.get("hard_sister_acc"))
            raw_18 = _fmt_pct(row.get("holdout_18cls_raw_bacc"))
            sk_18 = _fmt_pct(row.get("holdout_18cls_sinkhorn_bacc"))
            delta = _fmt_delta(row.get("delta_vs_baseline_bacc"))
            notes = str(row.get("notes", "")).replace("|", "\\|")

            line = (
                f"| `{exp_id}` | {phase} | {fam} | `{feat}` | {sub_d} | {head} | {gw} | "
                f"{oof_fam} | **{hld_fam}** | {sis_acc} | {raw_18} | **{sk_18}** | **{delta}** | {notes} |"
            )
            lines.append(line)

        lines.append("")
        self.md_path.write_text("\n".join(lines), encoding="utf-8")
