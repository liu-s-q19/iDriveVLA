from __future__ import annotations

import json
import hashlib
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


NavhardJob = Dict[str, str]
PREDICTION_DIAGNOSTIC_COLUMNS = [
    "protocol_valid",
    "protocol_reason",
    "action_tokens_count",
    "protocol_action_tokens_count",
    "generated_action_tokens_count",
    "raw_pose_count",
    "was_padded",
    "was_truncated",
    "used_zero_fallback",
]


def safe_token_list(tokens: Optional[Sequence[str]]) -> List[str]:
    return [] if tokens is None else list(tokens)


def intersect_tokens(lhs: Optional[Sequence[str]], rhs: Optional[Sequence[str]]) -> List[str]:
    rhs_set = set(safe_token_list(rhs))
    seen = set()
    ordered_tokens: List[str] = []
    for token in safe_token_list(lhs):
        if token in rhs_set and token not in seen:
            ordered_tokens.append(token)
            seen.add(token)
    return ordered_tokens


def select_stage_two_tokens(
    *,
    traffic_mode: str,
    available_synthetic_tokens: Optional[Sequence[str]],
    reactive_initial_tokens: Optional[Sequence[str]],
    non_reactive_initial_tokens: Optional[Sequence[str]],
    metric_cache_tokens: Optional[Sequence[str]],
) -> List[str]:
    mode = str(traffic_mode).lower()
    if mode == "reactive":
        scene_filter_tokens = reactive_initial_tokens
    elif mode == "non_reactive":
        scene_filter_tokens = non_reactive_initial_tokens
    else:
        raise ValueError(f"Unsupported traffic_mode={traffic_mode}")

    # Preserve scene_filter order while keeping only tokens backed by synthetic scenes and metric cache.
    available = intersect_tokens(scene_filter_tokens, available_synthetic_tokens)
    return intersect_tokens(available, metric_cache_tokens)


def stable_token_seed(token: str, seed_base: int = 0) -> int:
    token_bytes = f"{int(seed_base)}::{token}".encode("utf-8")
    digest = hashlib.sha256(token_bytes).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**31 - 1)


def build_eval_jobs(stage1_tokens: Sequence[str], stage2_tokens: Sequence[str]) -> List[NavhardJob]:
    jobs: List[NavhardJob] = []
    for stage_name, tokens in (("stage_one", stage1_tokens), ("stage_two", stage2_tokens)):
        for token in tokens:
            jobs.append(
                {
                    "job_id": f"{stage_name}:{token}",
                    "stage_name": stage_name,
                    "token": str(token),
                }
            )
    return jobs


def shard_eval_jobs(jobs: Sequence[NavhardJob], num_shards: int) -> List[List[NavhardJob]]:
    if num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {num_shards}")
    shards: List[List[NavhardJob]] = [[] for _ in range(num_shards)]
    for idx, job in enumerate(jobs):
        shards[idx % num_shards].append(dict(job))
    return shards


def collect_all_mappings(
    raw_mapping: Optional[Iterable[Sequence[Any]]],
    scene_tokens: Sequence[str],
) -> Dict[Tuple[str, str], List[Tuple[str, str]]]:
    all_mappings: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
    if raw_mapping is None:
        return all_mappings

    scene_tokens_set = set(scene_tokens)
    for item in raw_mapping:
        if len(item) != 3:
            continue
        orig_token, prev_token, two_stage_pairs = item
        if prev_token not in scene_tokens_set and orig_token not in scene_tokens_set:
            continue
        all_mappings[(str(orig_token), str(prev_token))] = [tuple(map(str, pair[:2])) for pair in two_stage_pairs if len(pair) >= 2]
    return all_mappings


def filter_mappings_by_available_tokens(
    all_mappings: Dict[Tuple[str, str], List[Tuple[str, str]]],
    available_tokens: Sequence[str],
) -> Dict[Tuple[str, str], List[Tuple[str, str]]]:
    available = set(available_tokens)
    filtered: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
    for (orig_token, prev_token), pairs in all_mappings.items():
        if orig_token not in available or prev_token not in available:
            continue
        valid_pairs = [pair for pair in pairs if len(pair) >= 2 and pair[0] in available and pair[1] in available]
        if valid_pairs:
            filtered[(orig_token, prev_token)] = valid_pairs
    return filtered


def validate_job_coverage(expected_jobs: Sequence[NavhardJob], combined_rows: pd.DataFrame) -> None:
    expected = {(job["stage_name"], job["token"]) for job in expected_jobs}
    if "stage_name" not in combined_rows.columns or "token" not in combined_rows.columns:
        raise RuntimeError("Combined shard rows must include 'stage_name' and 'token' columns for coverage validation.")

    actual_pairs = list(zip(combined_rows["stage_name"].astype(str), combined_rows["token"].astype(str)))
    actual = set(actual_pairs)
    missing = sorted(expected - actual)
    duplicates = sorted({pair for pair in actual if actual_pairs.count(pair) > 1})

    if missing:
        preview = ", ".join(f"{stage}:{token}" for stage, token in missing[:10])
        raise RuntimeError(f"Missing shard results for {len(missing)} jobs. First missing jobs: {preview}")
    if duplicates:
        preview = ", ".join(f"{stage}:{token}" for stage, token in duplicates[:10])
        raise RuntimeError(f"Duplicate shard results detected for {len(duplicates)} jobs. First duplicates: {preview}")


def summarize_prediction_diagnostics(rows: pd.DataFrame) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "num_protocol_invalid": 0,
        "num_padded": 0,
        "num_truncated": 0,
        "num_zero_fallback": 0,
        "action_tokens_count_mean": None,
        "action_tokens_count_min": None,
        "action_tokens_count_max": None,
        "protocol_action_tokens_count_mean": None,
        "protocol_action_tokens_count_min": None,
        "protocol_action_tokens_count_max": None,
        "generated_action_tokens_count_mean": None,
        "generated_action_tokens_count_min": None,
        "generated_action_tokens_count_max": None,
    }
    if rows.empty:
        return summary

    if "protocol_valid" in rows.columns:
        protocol_valid = pd.to_numeric(rows["protocol_valid"], errors="coerce").fillna(0)
        summary["num_protocol_invalid"] = int((protocol_valid == 0).sum())
    if "was_padded" in rows.columns:
        summary["num_padded"] = int(pd.to_numeric(rows["was_padded"], errors="coerce").fillna(0).sum())
    if "was_truncated" in rows.columns:
        summary["num_truncated"] = int(pd.to_numeric(rows["was_truncated"], errors="coerce").fillna(0).sum())
    if "used_zero_fallback" in rows.columns:
        summary["num_zero_fallback"] = int(pd.to_numeric(rows["used_zero_fallback"], errors="coerce").fillna(0).sum())
    if "action_tokens_count" in rows.columns:
        action_counts = pd.to_numeric(rows["action_tokens_count"], errors="coerce").dropna()
        if not action_counts.empty:
            summary["action_tokens_count_mean"] = float(action_counts.mean())
            summary["action_tokens_count_min"] = int(action_counts.min())
            summary["action_tokens_count_max"] = int(action_counts.max())
    if "protocol_action_tokens_count" in rows.columns:
        protocol_action_counts = pd.to_numeric(rows["protocol_action_tokens_count"], errors="coerce").dropna()
        if not protocol_action_counts.empty:
            summary["protocol_action_tokens_count_mean"] = float(protocol_action_counts.mean())
            summary["protocol_action_tokens_count_min"] = int(protocol_action_counts.min())
            summary["protocol_action_tokens_count_max"] = int(protocol_action_counts.max())
    if "generated_action_tokens_count" in rows.columns:
        generated_action_counts = pd.to_numeric(rows["generated_action_tokens_count"], errors="coerce").dropna()
        if not generated_action_counts.empty:
            summary["generated_action_tokens_count_mean"] = float(generated_action_counts.mean())
            summary["generated_action_tokens_count_min"] = int(generated_action_counts.min())
            summary["generated_action_tokens_count_max"] = int(generated_action_counts.max())
    return summary


def compute_final_scores(pdm_score_df: pd.DataFrame) -> pd.DataFrame:
    df = pdm_score_df.reset_index()
    assert not df["two_frame_extended_comfort"].isna().any(), (
        "Found NaN in 'two_frame_extended_comfort'. Please check aggregator completeness."
    )
    two_frame_scores = df["two_frame_extended_comfort"].to_numpy()
    weighted_metrics = np.stack(df["weighted_metrics"].to_numpy())
    weighted_metrics_array = np.stack(df["weighted_metrics_array"].to_numpy())

    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex

    two_frame_idx = WeightedMetricIndex.TWO_FRAME_EXTENDED_COMFORT
    weighted_metrics[:, two_frame_idx] = two_frame_scores
    weighted_sum = (weighted_metrics * weighted_metrics_array).sum(axis=1)
    total_weight = weighted_metrics_array.sum(axis=1)
    assert np.all(total_weight > 0), "Found total_weight == 0 during score computation."

    weighted_metric_scores = weighted_sum / total_weight
    df["score"] = df["multiplicative_metrics_prod"].to_numpy() * weighted_metric_scores
    df.drop(columns=["weighted_metrics", "weighted_metrics_array", "multiplicative_metrics_prod"], inplace=True)
    return df


def calculate_weighted_average_score(df: pd.DataFrame) -> pd.Series:
    score_cols = [c for c in df.columns if c not in {"weight", "token"}]
    if df.empty:
        return pd.Series([np.nan] * len(score_cols), index=score_cols)

    weights = df["weight"].to_numpy()
    scores = df[score_cols].to_numpy()
    total_weight = weights.sum()
    if total_weight == 0:
        return pd.Series([np.nan] * len(score_cols), index=score_cols)
    weighted_avg = (scores * weights[:, None]).sum(axis=0) / total_weight
    return pd.Series(weighted_avg, index=score_cols)


def calculate_individual_mapping_scores(
    pdm_score_df: pd.DataFrame,
    all_mappings: Dict[Tuple[str, str], List[Tuple[str, str]]],
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    all_group_scores = []
    stage1_group_scores = []
    stage2_group_scores = []

    for (orig_token, prev_token), second_stage_pairs in all_mappings.items():
        first_tokens = [pair[0] for pair in second_stage_pairs if len(pair) > 0]
        second_tokens = [pair[1] for pair in second_stage_pairs if len(pair) > 1]

        group1_stage1_df = pdm_score_df[pdm_score_df["token"] == orig_token]
        group1_stage2_df = pdm_score_df[pdm_score_df["token"].isin(first_tokens)]

        group2_stage1_df = pdm_score_df[pdm_score_df["token"] == prev_token]
        group2_stage2_df = pdm_score_df[pdm_score_df["token"].isin(second_tokens)]

        group1_stage1_scores = calculate_weighted_average_score(group1_stage1_df)
        group1_stage2_scores = calculate_weighted_average_score(group1_stage2_df)
        group2_stage1_scores = calculate_weighted_average_score(group2_stage1_df)
        group2_stage2_scores = calculate_weighted_average_score(group2_stage2_df)

        stage1_group_scores.append(group1_stage1_scores)
        stage1_group_scores.append(group2_stage1_scores)
        stage2_group_scores.append(group1_stage2_scores)
        stage2_group_scores.append(group2_stage2_scores)

        group1_scores = group1_stage1_scores * group1_stage2_scores
        group2_scores = group2_stage1_scores * group2_stage2_scores
        all_group_scores.append((group1_scores + group2_scores) / 2)

    if not all_group_scores:
        nan_series = pd.Series(dtype=float)
        return nan_series, nan_series, nan_series

    return (
        pd.DataFrame(all_group_scores).mean(),
        pd.DataFrame(stage1_group_scores).mean(),
        pd.DataFrame(stage2_group_scores).mean(),
    )


def create_scene_aggregators(
    all_mappings: Dict[Tuple[str, str], List[Tuple[str, str]]],
    full_score_df: pd.DataFrame,
    proposal_sampling: Any,
) -> pd.DataFrame:
    from navsim.planning.simulation.planner.pdm_planner.scoring.scene_aggregator import SceneAggregator

    full_score_df = full_score_df.copy()
    full_score_df["two_frame_extended_comfort"] = np.nan
    full_score_df["weight"] = np.nan
    full_score_df = full_score_df.set_index("token")

    all_updates = []
    for (now_frame, previous_frame), second_stage in all_mappings.items():
        aggregator = SceneAggregator(
            now_frame=now_frame,
            previous_frame=previous_frame,
            second_stage=second_stage,
            score_df=full_score_df,
            proposal_sampling=proposal_sampling,
        )
        updated_rows = aggregator.aggregate_scores()
        all_updates.append(updated_rows)

    all_updates_df = pd.concat(all_updates, ignore_index=True).set_index("token")
    full_score_df.update(all_updates_df)
    full_score_df.reset_index(inplace=True)
    full_score_df = full_score_df.drop(columns=["ego_simulated_states"])
    return full_score_df


def finalize_merged_results(
    *,
    combined_rows: pd.DataFrame,
    all_mappings: Dict[Tuple[str, str], List[Tuple[str, str]]],
    proposal_sampling: Any,
    scene_frame_type_original: Any,
    scene_frame_type_synthetic: Any,
    pdm_result_field_names: Sequence[str],
    output_dir: Optional[Path],
    write_artifacts: bool = True,
    logger: Optional[Any] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    pdm_score_df = combined_rows.copy()
    available_tokens = [str(x) for x in pdm_score_df["token"].tolist()]
    eval_mappings = filter_mappings_by_available_tokens(all_mappings, available_tokens)

    pseudo_closed_loop_valid = False
    try:
        if eval_mappings:
            pdm_score_df = create_scene_aggregators(eval_mappings, pdm_score_df, proposal_sampling)
            pdm_score_df = compute_final_scores(pdm_score_df)
            pseudo_closed_loop_valid = True
        else:
            pdm_score_df["weight"] = 1.0
    except Exception:
        if logger is not None:
            logger.warning("Failed to calculate pseudo closed-loop weights or comfort.")
            logger.warning(traceback.format_exc())
        pdm_score_df["weight"] = 1.0
        pseudo_closed_loop_valid = False

    num_successful = int(pdm_score_df["valid"].sum())
    num_failed = int(len(pdm_score_df) - num_successful)

    score_cols = [
        c
        for c in pdm_score_df.columns
        if ((any(name in c for name in pdm_result_field_names) or c == "two_frame_extended_comfort" or c == "score") and c != "pdm_score")
    ]
    if "score" not in score_cols:
        if "pdm_score" in pdm_score_df.columns:
            pdm_score_df["score"] = pdm_score_df["pdm_score"]
        else:
            pdm_score_df["score"] = np.nan
        score_cols.append("score")

    if "frame_type" not in pdm_score_df.columns:
        pdm_score_df["frame_type"] = np.nan

    if eval_mappings:
        pcl_group_score, pcl_stage1_score, pcl_stage2_score = calculate_individual_mapping_scores(
            pdm_score_df[score_cols + ["token", "weight"]],
            eval_mappings,
        )
    else:
        pcl_group_score = pd.Series({"score": pdm_score_df[pdm_score_df["valid"]]["score"].mean()})
        stage1_mean = pdm_score_df[pdm_score_df["frame_type"] == scene_frame_type_original]["score"].mean()
        stage2_mean = pdm_score_df[pdm_score_df["frame_type"] == scene_frame_type_synthetic]["score"].mean()
        pcl_stage1_score = pd.Series({"score": stage1_mean})
        pcl_stage2_score = pd.Series({"score": stage2_mean})

    for col in score_cols:
        stage_one_mask = pdm_score_df["frame_type"] == scene_frame_type_original
        stage_two_mask = pdm_score_df["frame_type"] == scene_frame_type_synthetic
        pdm_score_df.loc[stage_one_mask, f"{col}_stage_one"] = pdm_score_df.loc[stage_one_mask, col]
        pdm_score_df.loc[stage_two_mask, f"{col}_stage_two"] = pdm_score_df.loc[stage_two_mask, col]

    pdm_score_df.drop(columns=score_cols, inplace=True)
    pdm_score_df["score"] = pdm_score_df["score_stage_one"].combine_first(pdm_score_df["score_stage_two"])
    pdm_score_df.drop(columns=["score_stage_one", "score_stage_two"], inplace=True)

    stage1_cols = [f"{col}_stage_one" for col in score_cols if col != "score"]
    stage2_cols = [f"{col}_stage_two" for col in score_cols if col != "score"]
    score_cols = stage1_cols + stage2_cols + ["score"]

    diagnostic_cols = [col for col in PREDICTION_DIAGNOSTIC_COLUMNS if col in pdm_score_df.columns]
    keep_cols = ["token", "valid"] + diagnostic_cols + score_cols
    pdm_score_df = pdm_score_df[keep_cols]

    summary_rows = []

    stage1_row = pd.Series(index=pdm_score_df.columns, dtype=object)
    stage1_row["token"] = "extended_pdm_score_stage_one"
    stage1_row["valid"] = pseudo_closed_loop_valid
    stage1_row["score"] = pcl_stage1_score.get("score", np.nan)
    for col in pcl_stage1_score.index:
        if col not in ["token", "valid", "score"]:
            stage1_row[f"{col}_stage_one"] = pcl_stage1_score[col]
    summary_rows.append(stage1_row)

    stage2_row = pd.Series(index=pdm_score_df.columns, dtype=object)
    stage2_row["token"] = "extended_pdm_score_stage_two"
    stage2_row["valid"] = pseudo_closed_loop_valid
    stage2_row["score"] = pcl_stage2_score.get("score", np.nan)
    for col in pcl_stage2_score.index:
        if col not in ["token", "valid", "score"]:
            stage2_row[f"{col}_stage_two"] = pcl_stage2_score[col]
    summary_rows.append(stage2_row)

    combined_row = pd.Series(index=pdm_score_df.columns, dtype=object)
    combined_row["token"] = "extended_pdm_score_combined"
    combined_row["valid"] = pseudo_closed_loop_valid
    combined_row["score"] = pcl_group_score.get("score", np.nan)
    for col in pcl_stage1_score.index:
        if col not in ["token", "valid", "score"]:
            combined_row[f"{col}_stage_one"] = pcl_stage1_score[col]
    for col in pcl_stage2_score.index:
        if col not in ["token", "valid", "score"]:
            combined_row[f"{col}_stage_two"] = pcl_stage2_score[col]
    summary_rows.append(combined_row)

    pdm_score_df = pd.concat([pdm_score_df, pd.DataFrame(summary_rows)], ignore_index=True)

    csv_path: Optional[Path] = None
    if write_artifacts:
        if output_dir is None:
            raise ValueError("output_dir is required when write_artifacts=True")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"{datetime.utcnow().strftime('%Y.%m.%d.%H.%M.%S')}.csv"
        pdm_score_df.to_csv(csv_path, index=False)

    summary = {
        "num_successful_scenarios": num_successful,
        "num_failed_scenarios": num_failed,
        "final_extended_pdm_score": float(
            pdm_score_df[pdm_score_df["token"] == "extended_pdm_score_combined"]["score"].iloc[0]
        ),
        "csv_path": str(csv_path) if csv_path is not None else None,
    }
    summary.update(summarize_prediction_diagnostics(combined_rows))

    if write_artifacts:
        with open(Path(output_dir) / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    return pdm_score_df, summary
