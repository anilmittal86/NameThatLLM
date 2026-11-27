"""Tests for purified Clean Master Pool V2 assembly and prompt gap target generation."""

from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

WORKSPACE_DIR = Path(__file__).resolve().parents[1]
if str(WORKSPACE_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR / "scripts"))
if str(WORKSPACE_DIR / "src") not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR / "src"))

from data_preparation import build_clean_master_pool_v2  # noqa: E402
from namethatllm.config import LABELS  # noqa: E402


def _make_passage(seed: str, n_words: int = 50) -> str:
    tokens = [f"tok_{seed}_{i}" for i in range(n_words - 1)] + ["end."]
    return " ".join(tokens)


class TestDataPreparationV2(unittest.TestCase):
    def test_fidelity_filtering_blacklist_and_prompt_gap_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            crt_dir = data_dir / "external_datasets" / "crtproduction"
            prv_dir = data_dir / "external_datasets" / "prvsiyan"
            inst_dir = data_dir / "external_datasets" / "prvsiyan_instyle"
            crt_dir.mkdir(parents=True)
            prv_dir.mkdir(parents=True)
            inst_dir.mkdir(parents=True)

            bench_text = _make_passage("bench")
            train_text = _make_passage("train")
            pd.DataFrame({"text": [bench_text], "model": ["gpt-5.5"]}).to_parquet(
                data_dir / "clean_benchmark_1800_rows.parquet", index=False
            )
            pd.DataFrame({"text": [train_text], "model": ["gpt-5.5"], "prompt_id": ["p_tr"]}).to_csv(
                data_dir / "train.csv", index=False
            )

            good_gem_text = _make_passage("good_gem37")
            bad_gem_text = _make_passage("bad_gem37")
            low_fid_text = _make_passage("low_fid_kimi")
            newline_text = _make_passage("nl_gpt", 25) + "\n" + _make_passage("nl_part2", 25)
            good_claude_text = _make_passage("good_claude")

            # CRT v17 pool + fidelity table
            crt_rows = [
                {
                    "text": good_gem_text,
                    "model": "gemini-3.7-flash",
                    "source_dataset": "davidpiln/ntl-vertex-gemini-gen",
                    "prompt_canon": "p001",
                    "match_weight": 1.0,
                    "flag_gemini37_routing_risk": False,
                    "flag_train_prompt_regen": False,
                    "flag_near_train_prompt": False,
                    "flag_kb_proxy_after_ruling": False,
                },
                {
                    "text": good_claude_text,
                    "model": "claude-opus-4-8",
                    "source_dataset": "tadeuhayashida/ntl-gen003-claude-panel",
                    "prompt_canon": "p001",
                    "match_weight": 1.0,
                    "flag_gemini37_routing_risk": False,
                    "flag_train_prompt_regen": False,
                    "flag_near_train_prompt": False,
                    "flag_kb_proxy_after_ruling": False,
                },
                {
                    "text": low_fid_text,
                    "model": "kimi-k3",
                    "source_dataset": "elliotthankinson/ntl-tells-v4",
                    "prompt_canon": "p001",
                    "match_weight": 1.0,
                    "flag_gemini37_routing_risk": False,
                    "flag_train_prompt_regen": False,
                    "flag_near_train_prompt": False,
                    "flag_kb_proxy_after_ruling": False,
                },
                {
                    "text": bench_text,
                    "model": "gpt-5.5",
                    "source_dataset": "good_src",
                    "prompt_canon": "p001",
                    "match_weight": 1.0,
                    "flag_gemini37_routing_risk": False,
                    "flag_train_prompt_regen": False,
                    "flag_near_train_prompt": False,
                    "flag_kb_proxy_after_ruling": False,
                },
                {
                    "text": newline_text,
                    "model": "gpt-5.5",
                    "source_dataset": "good_src",
                    "prompt_canon": "p002",
                    "match_weight": 1.0,
                    "flag_gemini37_routing_risk": False,
                    "flag_train_prompt_regen": False,
                    "flag_near_train_prompt": False,
                    "flag_kb_proxy_after_ruling": False,
                },
            ]
            pd.DataFrame(crt_rows).to_parquet(crt_dir / "ntl_open_pool.parquet", index=False)
            pd.DataFrame({"text": [bad_gem_text]}).to_parquet(crt_dir / "gemini37_unverified.parquet", index=False)
            pd.DataFrame(
                [
                    {"source_dataset": "davidpiln/ntl-vertex-gemini-gen", "model": "gemini-3.7-flash", "fidelity_ratio": 0.969},
                    {"source_dataset": "tadeuhayashida/ntl-gen003-claude-panel", "model": "claude-opus-4-8", "fidelity_ratio": 0.935},
                    {"source_dataset": "elliotthankinson/ntl-tells-v4", "model": "kimi-k3", "fidelity_ratio": 0.232},
                    {"source_dataset": "good_src", "model": "gpt-5.5", "fidelity_ratio": 0.950},
                ]
            ).to_csv(crt_dir / "fidelity_by_source.csv", index=False)

            # prvsiyan passages: includes bad non-thinking Alibaba kimi-k3, good high Moonshot kimi-k3, bad gem37
            prv_good_kimi = _make_passage("prv_high_kimi")
            prv_nothink_kimi = _make_passage("prv_nothink_kimi")
            prv_bad_gem37 = _make_passage("prv_bad_gem37")
            pd.DataFrame(
                [
                    {
                        "text": prv_good_kimi,
                        "model": "kimi-k3",
                        "passes_funnel": True,
                        "reasoning_setting": 'variant: {"reasoning": {"effort": "high"}}',
                        "served_provider": "Moonshot AI",
                    },
                    {
                        "text": prv_nothink_kimi,
                        "model": "kimi-k3",
                        "passes_funnel": True,
                        "reasoning_setting": 'variant: {"reasoning": {"enabled": false}}',
                        "served_provider": "Alibaba",
                    },
                    {
                        "text": prv_bad_gem37,
                        "model": "gemini-3.7-flash",
                        "passes_funnel": True,
                        "reasoning_setting": "provider_default",
                        "served_provider": "Google Vertex AI",
                    },
                ]
            ).to_csv(prv_dir / "passages.csv", index=False)

            # prvsiyan_instyle attempts to re-inject low_fid_text, bad_gem_text, and prv_nothink_kimi
            instyle_clean_gpt = _make_passage("instyle_gpt")
            pd.DataFrame(
                [
                    {"text": low_fid_text, "model": "kimi-k3", "source": "elliotthankinson/ntl-tells-v4"},
                    {"text": bad_gem_text, "model": "gemini-3.7-flash", "source": "good_src"},
                    {"text": prv_nothink_kimi, "model": "kimi-k3", "source": "prvsiyan/prvsiyans-extended-name-that-llm-dataset"},
                    {"text": instyle_clean_gpt, "model": "gpt-5.5", "source": "good_src"},
                ]
            ).to_parquet(inst_dir / "pool_v1.parquet", index=False)

            # prompts.parquet for panel gap generation
            pd.DataFrame(
                [
                    {"prompt_canon": "p001", "prompt_text": "Describe alpha in detail.", "panel_labels": 10, "passages": 10},
                    {"prompt_canon": "p002", "prompt_text": "Explain beta (words 50-180).", "panel_labels": 5, "passages": 5},
                ]
            ).to_parquet(crt_dir / "prompts.parquet", index=False)

            df_v2, df_top, df_slots = build_clean_master_pool_v2(data_dir=data_dir, min_fidelity=0.70, n_gap_panels=10)
            texts = set(df_v2["text"])
            self.assertEqual(texts, {good_gem_text, good_claude_text, prv_good_kimi, instyle_clean_gpt})
            self.assertNotIn(low_fid_text, texts)
            self.assertNotIn(prv_nothink_kimi, texts)
            self.assertNotIn(prv_bad_gem37, texts)
            self.assertNotIn(bad_gem_text, texts)
            self.assertNotIn(bench_text, texts)
            self.assertNotIn(train_text, texts)
            self.assertNotIn(newline_text, texts)

            # p001 has 2 verified clean classes (gemini-3.7-flash, claude-opus-4-8) -> missing 16 incl kimi-k3
            self.assertEqual(len(df_top), 2)
            self.assertEqual(df_top.iloc[0]["prompt_canon"], "p001")
            self.assertEqual(df_top.iloc[0]["n_present_clean"], 2)
            self.assertEqual(df_top.iloc[0]["n_missing"], len(LABELS) - 2)
            self.assertTrue((df_slots["prompt_canon"] == "p001").any())


if __name__ == "__main__":
    unittest.main()
