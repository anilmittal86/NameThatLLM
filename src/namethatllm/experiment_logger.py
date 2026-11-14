"""
Unified Multi-Backend Experiment Tracker (CSV/Markdown/SQLite + MLflow + W&B)
=============================================================================
Provides a unified experiment ledger that simultaneously logs runs to:
  1. Local storage & SQLite (Always Active, zero external dependencies):
     - experiments/experiment_ledger.csv
     - experiments/experiment_ledger.md
     - experiments/experiment_ledger.db (table: experiments)
  2. MLflow Backend (optional, auto-initialized if installed):
     - Logs parameters, accuracy metrics, and artifacts to SQLite/local URI.
  3. Weights & Biases Backend (optional, auto-enabled if WANDB_API_KEY is in env):
     - Logs parameters and metrics live to cloud W&B project dashboard.
     - Automatically guarded against offline Kaggle competition rerun environments.
"""

from __future__ import annotations

import csv
import os
import shutil
import sqlite3
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
    Unified Multi-Backend Experiment Tracker:
    - Zero external dependencies for CSV, Markdown, and SQLite storage.
    - Safe optional integration with MLflow and Weights & Biases.
    """

    def __init__(
        self,
        ledger_dir: Path | str = "experiments",
        baseline_bacc: float | None = None,
        use_mlflow: bool = True,
        use_wandb: bool | None = None,
    ):
        self.ledger_dir = Path(ledger_dir)
        self.ledger_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.ledger_dir / "experiment_ledger.csv"
        self.md_path = self.ledger_dir / "experiment_ledger.md"
        self.db_path = self.ledger_dir / "experiment_ledger.db"
        self.baseline_bacc = baseline_bacc

        # 1. Initialize CSV
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(LEDGER_COLUMNS)

        # 2. Initialize SQLite table
        self._init_sqlite()

        # 3. Optional MLflow backend initialization
        self.use_mlflow = use_mlflow
        if self.use_mlflow:
            self._init_mlflow()

        # 4. Optional W&B backend initialization
        if use_wandb is None:
            # Auto-enable if WANDB_API_KEY is present and not in competition rerun
            self.use_wandb = bool(os.environ.get("WANDB_API_KEY")) and (
                os.environ.get("KAGGLE_IS_COMPETITION_RERUN") != "1"
            )
        else:
            self.use_wandb = use_wandb and (os.environ.get("KAGGLE_IS_COMPETITION_RERUN") != "1")

    # -------------------------------------------------------------------------
    # Core Local Storage (CSV, Markdown, SQLite)
    # -------------------------------------------------------------------------

    def _init_sqlite(self) -> None:
        """Initializes the local SQLite database and table schema."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS experiments (
                        exp_id TEXT PRIMARY KEY,
                        timestamp TEXT,
                        phase TEXT,
                        target_family TEXT,
                        feature_set TEXT,
                        subspace_dim INTEGER,
                        head_type TEXT,
                        gate_weight REAL,
                        oof_family_bacc REAL,
                        holdout_family_bacc REAL,
                        hard_sister_acc REAL,
                        holdout_18cls_raw_bacc REAL,
                        holdout_18cls_sinkhorn_bacc REAL,
                        delta_vs_baseline_bacc REAL,
                        notes TEXT
                    )
                """)
                conn.commit()
        except Exception as e:
            print(f"[ExperimentLedger] SQLite init warning: {e}")

    def _log_sqlite(self, entry: dict[str, Any]) -> None:
        """Inserts or replaces an experiment entry in SQLite."""
        try:
            def _to_num(val: Any, typ: type) -> Any:
                if val is None or val == "" or pd.isna(val):
                    return None
                try:
                    return typ(val)
                except (ValueError, TypeError):
                    return None

            row = (
                str(entry.get("exp_id", "")),
                str(entry.get("timestamp", "")),
                str(entry.get("phase", "")),
                str(entry.get("target_family", "")),
                str(entry.get("feature_set", "")),
                _to_num(entry.get("subspace_dim"), int),
                str(entry.get("head_type", "")),
                _to_num(entry.get("gate_weight"), float),
                _to_num(entry.get("oof_family_bacc"), float),
                _to_num(entry.get("holdout_family_bacc"), float),
                _to_num(entry.get("hard_sister_acc"), float),
                _to_num(entry.get("holdout_18cls_raw_bacc"), float),
                _to_num(entry.get("holdout_18cls_sinkhorn_bacc"), float),
                _to_num(entry.get("delta_vs_baseline_bacc"), float),
                str(entry.get("notes", "")),
            )
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO experiments (
                        exp_id, timestamp, phase, target_family, feature_set,
                        subspace_dim, head_type, gate_weight, oof_family_bacc,
                        holdout_family_bacc, hard_sister_acc, holdout_18cls_raw_bacc,
                        holdout_18cls_sinkhorn_bacc, delta_vs_baseline_bacc, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, row)
                conn.commit()
        except Exception as e:
            print(f"[ExperimentLedger] SQLite log warning: {e}")

    # -------------------------------------------------------------------------
    # Optional Cloud & Local UI Backends (MLflow & W&B)
    # -------------------------------------------------------------------------

    def _init_mlflow(self) -> None:
        """Configures MLflow tracking URI and active experiment."""
        try:
            import mlflow

            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
            if not tracking_uri:
                mlflow_db = (self.ledger_dir / "mlflow.db").resolve()
                tracking_uri = f"sqlite:///{mlflow_db.as_posix()}"
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment("NameThatLLM_Gates_and_Ensemble")
        except Exception:
            self.use_mlflow = False

    def _log_mlflow(self, entry: dict[str, Any], artifacts: list[Path | str] | None = None) -> None:
        """Logs parameters, metrics, and artifacts to MLflow."""
        if not self.use_mlflow:
            return
        try:
            import mlflow

            exp_id = str(entry.get("exp_id", "EXP"))
            target_fam = str(entry.get("target_family", ""))
            feat = str(entry.get("feature_set", ""))
            head = str(entry.get("head_type", ""))
            run_name = f"{exp_id}_{target_fam}_{feat}_{head}".strip("_")

            with mlflow.start_run(run_name=run_name):
                # 1. Parameters
                param_keys = [
                    "exp_id", "phase", "target_family", "feature_set",
                    "subspace_dim", "head_type", "gate_weight",
                    "select_layers", "include_velocity", "layer_weighting"
                ]
                params = {}
                for k in param_keys:
                    if k in entry and entry[k] != "" and entry[k] is not None:
                        params[k] = str(entry[k])[:250]
                if params:
                    mlflow.log_params(params)

                # 2. Metrics
                metric_keys = [
                    "oof_family_bacc", "holdout_family_bacc", "hard_sister_acc",
                    "holdout_18cls_raw_bacc", "holdout_18cls_sinkhorn_bacc",
                    "delta_vs_baseline_bacc"
                ]
                metrics = {}
                for k in metric_keys:
                    v = entry.get(k)
                    if v is not None and v != "" and not pd.isna(v):
                        try:
                            metrics[k] = float(v)
                        except (ValueError, TypeError):
                            pass
                if metrics:
                    mlflow.log_metrics(metrics)

                # 3. Artifacts
                if artifacts:
                    for art in artifacts:
                        art_p = Path(art)
                        if art_p.exists():
                            if art_p.is_file():
                                mlflow.log_artifact(str(art_p))
                            elif art_p.is_dir():
                                mlflow.log_artifacts(str(art_p))
        except Exception as e:
            print(f"[ExperimentLedger] MLflow logging note: {e}")

    def _log_wandb(self, entry: dict[str, Any]) -> None:
        """Logs run to Weights & Biases if enabled."""
        if not self.use_wandb:
            return
        if os.environ.get("KAGGLE_IS_COMPETITION_RERUN") == "1":
            return
        try:
            import wandb

            exp_id = str(entry.get("exp_id", "EXP"))
            target_fam = str(entry.get("target_family", ""))
            feat = str(entry.get("feature_set", ""))
            head = str(entry.get("head_type", ""))
            run_name = f"{exp_id}_{target_fam}_{feat}_{head}".strip("_")

            param_keys = [
                "exp_id", "phase", "target_family", "feature_set",
                "subspace_dim", "head_type", "gate_weight",
                "select_layers", "include_velocity", "layer_weighting"
            ]
            params = {k: entry[k] for k in param_keys if k in entry and entry[k] != "" and entry[k] is not None}

            metric_keys = [
                "oof_family_bacc", "holdout_family_bacc", "hard_sister_acc",
                "holdout_18cls_raw_bacc", "holdout_18cls_sinkhorn_bacc",
                "delta_vs_baseline_bacc"
            ]
            metrics = {}
            for k in metric_keys:
                v = entry.get(k)
                if v is not None and v != "" and not pd.isna(v):
                    try:
                        metrics[k] = float(v)
                    except (ValueError, TypeError):
                        pass

            wandb.init(
                project="NameThatLLM",
                name=run_name,
                config=params,
                reinit=True,
            )
            if metrics:
                wandb.log(metrics)
            wandb.finish()
        except Exception as e:
            print(f"[ExperimentLedger] W&B logging note: {e}")

    # -------------------------------------------------------------------------
    # Public Entrypoints
    # -------------------------------------------------------------------------

    def add_entry(
        self,
        record: dict[str, Any],
        artifacts: list[Path | str] | None = None,
    ) -> dict[str, Any]:
        """
        Appends an experiment record to CSV, SQLite, and Markdown table,
        and optionally streams to MLflow and W&B backends.
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

        # Standardize missing columns for CSV
        for col in LEDGER_COLUMNS:
            if col not in entry or entry[col] is None:
                entry[col] = ""

        # 1. Append to CSV
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([entry.get(c, "") for c in LEDGER_COLUMNS])

        # 2. Log to SQLite
        self._log_sqlite(entry)

        # 3. Log to MLflow
        self._log_mlflow(entry, artifacts=artifacts)

        # 4. Log to W&B
        self._log_wandb(entry)

        # 5. Regenerate Markdown table
        self._regenerate_markdown()
        return entry

    def get_dataframe(self) -> pd.DataFrame:
        """Loads all logged experiments as a pandas DataFrame."""
        if self.db_path.exists():
            try:
                with sqlite3.connect(self.db_path) as conn:
                    return pd.read_sql_query("SELECT * FROM experiments ORDER BY timestamp ASC", conn)
            except Exception:
                pass

        if self.csv_path.exists():
            try:
                return pd.read_csv(self.csv_path)
            except Exception:
                pass
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

        valid_rows = df.dropna(subset=[metric])
        if valid_rows.empty:
            return None
        best_idx = pd.to_numeric(valid_rows[metric], errors="coerce").idxmax()
        if pd.isna(best_idx):
            return None
        return df.loc[best_idx].to_dict()

    def export_copy(self, dest_dir: Path | str) -> None:
        """Copies the CSV, SQLite DB, and Markdown files to dest_dir (for Kaggle output artifacts)."""
        d = Path(dest_dir)
        d.mkdir(parents=True, exist_ok=True)
        if self.csv_path.exists():
            shutil.copy2(str(self.csv_path), str(d / "experiment_ledger.csv"))
        if self.md_path.exists():
            shutil.copy2(str(self.md_path), str(d / "experiment_ledger.md"))
        if self.db_path.exists():
            shutil.copy2(str(self.db_path), str(d / "experiment_ledger.db"))

    def _regenerate_markdown(self) -> None:
        """Formats the entire experiment history into a GitHub-Flavored Markdown table."""
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
