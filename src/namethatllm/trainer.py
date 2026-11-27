"""
Name That LLM - Thin Backward-Compatible Training Facade & CLI Entrypoint.
Re-exports all modular training primitives from `namethatllm.training.*`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .layer_probing import build_supervised_family_subspace, engineer_multiview_features, run_all_layers_probe_sweep
from .training.backbones import (
    _BACKBONE_REGISTRY,
    BaseBackboneAdapter,
    BgeAdapter,
    DebertaAdapter,
    PrecisionSpec,
    QwenAdapter,
    TokenizationSpec,
    _disable_final_norm,
    _stabilize_fp16_backbone,
    package_encoder_ridge_checkpoint,
    register_backbone,
    resolve_backbone_adapter,
    run_deberta_specialist_suite,
)
from .training.data import (
    PROBLEM_WATCH_CLASSES,
    BucketedLengthBatchSampler,
    PromptGroupBatchSampler,
    TextSequenceDataset,
    _load_benchmark_holdout,
    _load_stage1_pool_df,
    _normalize_match_weights,
    _select_stage1_pool_files,
    build_targeted_epoch_indices,
    collate_text_batch,
    load_clean_stage1_pool,
    load_zero_leak_stage1_pool,
    norm_text_hash,
)
from .training.embeddings import extract_and_export_embeddings, save_extracted_features_bundle
from .training.engine import (
    _configure_training_environment,
    _setup_training_env,
    run_encoder_stage1,
    run_training_pipeline,
    train_prompt_group_step,
)
from .training.evaluation import (
    _blend_head,
    _evaluate_bench_checkpoint,
    _solve_centered_dual_ridge,
    compute_detailed_metrics,
    compute_grouped_inference_logits,
    evaluate_stage2_oof_cv,
    extract_features_and_logits,
    fit_ridge_heads,
    gpu_trace_ridge_eval,
    print_prediction_table,
    solve_trace_ridge_wb,
)
from .training.gates import FamilyMLPHead, fit_family_gate_head, fit_family_mlp_head, run_sweep, train_family_gate, train_family_gate_from_dataset
from .training.losses import _build_sister_mask, compute_supcon_loss, compute_two_stage_loss, family_margin_cross_entropy
from .training.modeling import (
    MultiLayerLoRAClassifier,
    _build_class_weight_tensor,
    _get_loraplus_param_groups,
    _ManualFP16LossScaler,
    _step_with_grad_guard,
    export_compatible_checkpoint,
    find_compatible_champion_lora,
    load_and_slice_champion_heads,
)

run_bge_v2_stage1 = run_encoder_stage1


def main() -> None:
    parser = argparse.ArgumentParser(description="NameThatLLM Training & Gate Pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON")
    parser.add_argument("--from-embeddings-dir", type=str, default=None, help="Offline training from exported embeddings")
    parser.add_argument("--target-family", type=str, default="gemini", help="Target family for gate training")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--head-type", type=str, default="mlp", choices=["mlp", "ridge", "hybrid"])
    parser.add_argument("--select-layers", type=str, default=None, help="'auto_top8' or comma-separated layer indices")
    parser.add_argument("--use-multiview", action="store_true", help="Add engineered multi-view features")
    parser.add_argument("--no-velocity", action="store_true", help="Disable inter-layer velocity directions")
    parser.add_argument("--no-layer-weighting", action="store_true", help="Disable layer-probing importance weighting")
    args, _ = parser.parse_known_args()
    if args.from_embeddings_dir:
        out_p = Path(args.output_dir) if args.output_dir else Path(f"export_{args.target_family}_gate_{args.head_type}")
        sel_l = args.select_layers
        if sel_l is not None and sel_l != "auto_top8":
            sel_l = [int(x.strip()) for x in sel_l.split(",") if x.strip()]
        train_family_gate_from_dataset(
            dataset_dir=Path(args.from_embeddings_dir),
            target_family=args.target_family,
            output_dir=out_p,
            head_type=args.head_type,
            select_layers=sel_l,
            use_multiview=args.use_multiview,
            include_velocity=not args.no_velocity,
            layer_weighting=not args.no_layer_weighting,
        )
    elif args.config:
        run_training_pipeline(json.loads(Path(args.config).read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
