"""End-to-end Notebook-1 orchestration (preprocess + audit).

This is the single implementation of the Section-5 "Notebook 1" steps.  The
notebook is a thin wrapper around :func:`run_preprocess_pipeline` so the logic
lives in one importable, testable place.

All CSV artifacts store **paths relative to the repository root**; use
:func:`sutureqc.utils.resolve_path` to turn them back into absolute paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from . import audit, config, crops, geometry, labels_io, manifest, points_io, proxy, ratings, splits
from .utils import ensure_dir, relpath, write_json


def verify_expected_folders(data_root: Path = config.DATA_ROOT) -> dict[str, bool]:
    """Step 1: confirm the expected dataset folders exist."""
    annot = data_root / "annotation_images"
    checks = {
        "data_root": data_root.exists(),
        "annotation_images": annot.exists(),
        "masks": (annot / "masks").exists(),
        "stitch_bboxes": (annot / "stitch_bboxes").exists(),
        "manifest_csv": (annot / "masks" / "manifest.csv").exists(),
    }
    for batch in config.ALL_BATCHES:
        checks[f"batch/{batch}"] = (data_root / batch / "dvrk_wound_closure").exists()
    return checks


def _to_relative(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].map(lambda v: relpath(v, config.REPO_ROOT) if isinstance(v, str) and v else v)
    return out


def run_preprocess_pipeline(
    data_root: Path = config.DATA_ROOT,
    artifacts_dir: Path = config.ARTIFACTS_DIR,
    figures_dir: Path = config.FIGURES_DIR,
    crop_pad: float = config.CROP_PAD,
    seed: int = config.SEED,
) -> dict[str, Any]:
    """Run the full Notebook-1 pipeline and write every artifact.

    Returns a dict with the in-memory DataFrames (absolute paths) plus the
    audit report and proxy thresholds.
    """
    ensure_dir(artifacts_dir)
    ensure_dir(figures_dir)
    crops_dir = ensure_dir(artifacts_dir / "crops")

    checks = verify_expected_folders(data_root)
    if not all(v for k, v in checks.items() if not k.startswith("batch/")):
        missing = [k for k, v in checks.items() if not v and not k.startswith("batch/")]
        raise FileNotFoundError(f"Missing required dataset folders: {missing}")

    # -- Steps 2-5: manifest + full frame inventory (keeps empty boxes) ------- #
    base_manifest = manifest.load_manifest(data_root)
    manifest_all = manifest.build_frame_inventory(data_root, base_manifest)

    # -- Step 6: one row per stitch instance (valid boxes only) -------------- #
    stitch = manifest.stitch_instance_rows(manifest_all)

    # -- Step 7: episode/action context + robot kinematics metadata ---------- #
    action_labels = labels_io.load_all_action_labels(config.ALL_BATCHES, data_root)
    stitch = labels_io.annotate_action_context(stitch, action_labels)

    # -- Step 8: geometry features ------------------------------------------- #
    insertion_exit = points_io.load_all_insertion_exit(config.POINT_LABEL_BATCHES, data_root)
    features = geometry.compute_geometry_features(stitch, insertion_exit)

    # -- Step 11: grouped splits (episode-level) ----------------------------- #
    splits_df = splits.assign_grouped_splits(features, group_col="group_id", seed=seed)
    leakage = splits.check_no_group_leakage(splits_df)

    # -- Step 3.4 fallback target: geometric-consistency proxy --------------- #
    train_mask = (splits_df["split"] == "train").to_numpy()
    features_with_proxy, proxy_thresholds = proxy.build_proxy_targets(features, train_mask)
    features_with_proxy["split"] = splits_df["split"].to_numpy()

    # -- Step 9: crops ------------------------------------------------------- #
    crop_paths = crops.generate_crops(stitch, crops_dir)

    # -- Step 10: expert rating template ------------------------------------- #
    rating_template = ratings.build_expert_rating_template(stitch, crop_paths, n_raters=3)

    # -- Step 12: audit report + overlay figure ------------------------------ #
    report = audit.build_audit_report(
        manifest_all=manifest_all,
        stitch_df=stitch,
        features_df=features_with_proxy,
        splits_df=splits_df,
        proxy_thresholds=proxy_thresholds,
    )
    report["folder_checks"] = checks
    report["leakage_check"] = leakage
    report["n_action_labels"] = int(len(action_labels))
    report["n_insertion_exit_rows"] = int(len(insertion_exit))
    report["insertion_exit_skipped_lines"] = int(insertion_exit.attrs.get("n_skipped_lines", 0))
    audit.write_audit(report, artifacts_dir / "audit_report.json")

    splice_df = splits_df[["stitch_id", "group_id", "batch", "episode", "split"]].copy()

    # -- Write artifacts (paths stored relative to repo root) ---------------- #
    path_cols = ("frame_path", "mask_path", "bbox_path", "crop_path")
    _to_relative(manifest_all, path_cols).to_csv(artifacts_dir / "manifest_all.csv", index=False)
    _to_relative(stitch, path_cols).to_csv(artifacts_dir / "stitch_instances.csv", index=False)
    _to_relative(features_with_proxy, path_cols).to_csv(artifacts_dir / "geometry_features.csv", index=False)
    rating_template.to_csv(artifacts_dir / "expert_rating_template.csv", index=False)
    splice_df.to_csv(artifacts_dir / "splits_grouped.csv", index=False)
    write_json(artifacts_dir / "proxy_thresholds.json", {"edges": proxy_thresholds, "weights": config.PROXY_WEIGHTS})

    audit.plot_sample_annotations(stitch, figures_dir / "sample_annotations.png")

    return {
        "manifest_all": manifest_all,
        "stitch_instances": stitch,
        "geometry_features": features_with_proxy,
        "expert_rating_template": rating_template,
        "splits": splits_df,
        "crops": crop_paths,
        "audit_report": report,
        "proxy_thresholds": proxy_thresholds,
        "insertion_exit": insertion_exit,
        "action_labels": action_labels,
        "leakage_check": leakage,
        "folder_checks": checks,
    }
