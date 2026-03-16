import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from omegaconf import OmegaConf

from tools.eval.navsim_eval_audit import extract_failed_tokens_from_log, load_csv_rows, summarize_csv_rows


SHARD_DIR_RE = re.compile(r"^shard\d+$")


class GPUsNotReady(RuntimeError):
    pass


@dataclass(frozen=True)
class LaunchPaths:
    run_root: Path
    shards_root: Path
    tmp_root: Path
    master_log_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wait for GPUs and launch 8-shard navsim eval in tmux.")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--sensor-data-path", required=True)
    parser.add_argument("--metric-cache-path", required=True)
    parser.add_argument("--json-data-path", required=True)
    parser.add_argument("--scene-filter-config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--session-name", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--required-gpu-count", type=int, default=8)
    parser.add_argument("--max-used-memory-mb", type=int, default=2048)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--startup-wait-seconds", type=int, default=3)
    parser.add_argument("--gpu-indices", default="")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def query_gpu_stats() -> List[Dict[str, int]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stats: List[Dict[str, int]] = []
    for line in result.stdout.strip().splitlines():
        index, memory_used_mb, memory_total_mb, utilization_gpu = [part.strip() for part in line.split(",")]
        stats.append(
            {
                "index": int(index),
                "memory_used_mb": int(memory_used_mb),
                "memory_total_mb": int(memory_total_mb),
                "utilization_gpu": int(utilization_gpu),
            }
        )
    return stats


def select_ready_gpus(
    gpu_stats: Sequence[Dict[str, int]],
    required_gpu_count: int,
    max_used_memory_mb: int,
) -> List[int]:
    ready = [
        int(gpu["index"])
        for gpu in sorted(gpu_stats, key=lambda item: int(item["index"]))
        if int(gpu["memory_used_mb"]) <= max_used_memory_mb
    ]
    if len(ready) < required_gpu_count:
        raise GPUsNotReady(
            f"Not enough ready GPUs: required={required_gpu_count}, ready={len(ready)}, "
            f"max_used_memory_mb={max_used_memory_mb}, ready_indices={ready}"
        )
    return ready[:required_gpu_count]


def load_log_names(scene_filter_config_path: Path) -> List[str]:
    cfg = OmegaConf.load(scene_filter_config_path)
    log_names = list(cfg.log_names or [])
    if not log_names:
        raise ValueError(f"No log_names found in scene filter config: {scene_filter_config_path}")
    return log_names


def plan_log_shards(log_names: Sequence[str], num_shards: int) -> List[List[str]]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    shards: List[List[str]] = [[] for _ in range(num_shards)]
    for idx, log_name in enumerate(log_names):
        shards[idx % num_shards].append(str(log_name))
    return shards


def make_launch_paths(output_root: Path, run_tag: str) -> LaunchPaths:
    run_root = output_root / run_tag
    shards_root = run_root / "shards"
    tmp_root = run_root / "tmp"
    return LaunchPaths(
        run_root=run_root,
        shards_root=shards_root,
        tmp_root=tmp_root,
        master_log_path=run_root / "master.log",
    )


def ensure_dirs(paths: LaunchPaths) -> None:
    paths.run_root.mkdir(parents=True, exist_ok=True)
    paths.shards_root.mkdir(parents=True, exist_ok=True)
    paths.tmp_root.mkdir(parents=True, exist_ok=True)


def shell_join(items: Iterable[str]) -> str:
    return " ".join(shlex.quote(item) for item in items)


def hydra_string(value: str) -> str:
    return json.dumps(str(value), ensure_ascii=True)


def shard_command(
    repo_root: Path,
    shard_index: int,
    gpu_index: int,
    shard_logs: Sequence[str],
    args: argparse.Namespace,
    paths: LaunchPaths,
) -> str:
    shard_output_dir = paths.run_root / f"shard{shard_index}"
    shard_output_dir.mkdir(parents=True, exist_ok=True)
    shard_log_path = paths.shards_root / f"shard{shard_index}.log"
    shard_log_names = json.dumps(list(shard_logs), ensure_ascii=True, separators=(",", ":"))
    py_exe = "/data/miniconda/envs/autovla_codeclean/bin/python"
    navsim_devkit_root = repo_root / "navsim"
    env_exports = [
        f"export NAVSIM_DEVKIT_ROOT={shlex.quote(str(navsim_devkit_root))}",
        "export NAVSIM_DATA_ROOT=/data/dataset/navsim",
        "export NUPLAN_DATA_ROOT=/data/dataset/navsim",
        "export NUPLAN_MAPS_ROOT=/data/dataset/navsim/maps",
        "export NAVSIM_EXP_ROOT=/data/dataset/navsim/navsim_logs",
        "export OPENSCENE_DATA_ROOT=/data/dataset/navsim",
        "export NUPLAN_MAP_VERSION=nuplan-maps-v1.0",
        f"export PYTHONPATH={shlex.quote(str(navsim_devkit_root))}:$PYTHONPATH",
        f"cd {shlex.quote(str(repo_root))}",
        (
            "CUDA_VISIBLE_DEVICES="
            f"{gpu_index} {shlex.quote(py_exe)} navsim/navsim/planning/script/run_pdm_score_cot.py "
            "agent=autovla_agent "
            f"+agent.config_path={shlex.quote(hydra_string(args.config_path))} "
            f"+agent.checkpoint_path={shlex.quote(hydra_string(args.checkpoint_path))} "
            f"+agent.sensor_data_path={shlex.quote(hydra_string(args.sensor_data_path))} "
            f"+agent.lora_conf.use_lora={'true' if args.use_lora else 'false'} "
            f"metric_cache_path={shlex.quote(hydra_string(args.metric_cache_path))} "
            f"json_data_path={shlex.quote(hydra_string(args.json_data_path))} "
            "+split_filter.navtest=navtest "
            "worker=sequential "
            f"+seed={args.seed} "
            f"output_dir={shlex.quote(hydra_string(str(shard_output_dir)))} "
            f"train_test_split.scene_filter.log_names={shlex.quote(shard_log_names)} "
            "train_test_split.scene_filter.tokens=null"
        ),
    ]
    command = "bash -lc " + shlex.quote(" && ".join(env_exports) + f" >{shlex.quote(str(shard_log_path))} 2>&1")
    return command


def start_tmux_session(session_name: str, command: str) -> None:
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session_name, command],
        check=True,
    )


def session_pid(session_name: str) -> str:
    result = subprocess.run(
        ["tmux", "list-panes", "-t", session_name, "-F", "#{pane_pid}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().splitlines()[0]


def shard_done(path: Path) -> bool:
    return (path / "summary.json").exists()


def wait_for_shards(shard_output_dirs: Sequence[Path], poll_seconds: int, master_log_path: Path) -> None:
    pending = set(shard_output_dirs)
    while pending:
        completed = {path for path in pending if shard_done(path)}
        pending -= completed
        with master_log_path.open("a", encoding="utf-8") as f:
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            f.write(f"[{timestamp}] shard_progress completed={len(shard_output_dirs) - len(pending)} total={len(shard_output_dirs)}\n")
        if pending:
            time.sleep(poll_seconds)


def merge_csv_rows(csv_paths: Sequence[Path], merged_csv_path: Path) -> None:
    fieldnames: List[str] = []
    merged_rows: List[Dict[str, str]] = []
    for csv_path in csv_paths:
        rows = load_csv_rows(csv_path)
        scenario_rows = [row for row in rows if str(row.get("token", "")).strip().lower() != "average"]
        if scenario_rows and not fieldnames:
            fieldnames = list(scenario_rows[0].keys())
        merged_rows.extend(scenario_rows)
    if not fieldnames:
        fieldnames = ["token", "valid"]
    with merged_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)


def collect_summary(run_root: Path) -> Dict[str, object]:
    shard_dirs = sorted(path for path in run_root.iterdir() if path.is_dir() and SHARD_DIR_RE.fullmatch(path.name))
    csv_paths: List[Path] = []
    failed_tokens: List[str] = []
    for shard_dir in shard_dirs:
        shard_log_path = run_root / "shards" / f"{shard_dir.name}.log"
        failed_tokens.extend(extract_failed_tokens_from_log(shard_log_path))
        summary_path = shard_dir / "summary.json"
        if not summary_path.exists():
            shard_csvs = sorted(shard_dir.glob("*.csv"))
            if not shard_csvs:
                raise FileNotFoundError(f"No shard summary or CSV found in {shard_dir}")
            csv_paths.append(shard_csvs[-1])
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        csv_paths.append(Path(summary["csv_path"]))

    merged_csv_path = run_root / "merged_rows.csv"
    merge_csv_rows(csv_paths, merged_csv_path)
    summary = summarize_csv_rows(load_csv_rows(merged_csv_path), failed_tokens=failed_tokens)
    summary["csv_path"] = str(merged_csv_path)
    summary["shard_count"] = len(shard_dirs)
    return summary


def wait_for_ready_gpus(required_gpu_count: int, max_used_memory_mb: int, poll_seconds: int, master_log_path: Path) -> List[int]:
    while True:
        gpu_stats = query_gpu_stats()
        try:
            ready = select_ready_gpus(
                gpu_stats,
                required_gpu_count=required_gpu_count,
                max_used_memory_mb=max_used_memory_mb,
            )
        except GPUsNotReady as exc:
            with master_log_path.open("a", encoding="utf-8") as f:
                timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                f.write(f"[{timestamp}] waiting_for_gpus {exc}\n")
            time.sleep(poll_seconds)
            continue
        with master_log_path.open("a", encoding="utf-8") as f:
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            f.write(f"[{timestamp}] ready_gpus={ready}\n")
        return ready


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_root = Path(args.output_root)
    paths = make_launch_paths(output_root, args.run_tag)
    ensure_dirs(paths)

    log_names = load_log_names(Path(args.scene_filter_config))
    shard_logs = plan_log_shards(log_names, args.num_shards)

    with paths.master_log_path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "event": "launcher_start",
                    "session_name": args.session_name,
                    "run_tag": args.run_tag,
                    "checkpoint_path": args.checkpoint_path,
                    "config_path": args.config_path,
                    "required_gpu_count": args.required_gpu_count,
                    "max_used_memory_mb": args.max_used_memory_mb,
                    "num_shards": args.num_shards,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    if args.gpu_indices.strip():
        ready_gpus = [int(part.strip()) for part in args.gpu_indices.split(",") if part.strip()]
        if len(ready_gpus) < args.required_gpu_count:
            raise GPUsNotReady(
                f"Explicit gpu_indices provided but insufficient: required={args.required_gpu_count}, got={ready_gpus}"
            )
        with paths.master_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"event": "explicit_ready_gpus", "gpu_indices": ready_gpus}) + "\n")
    else:
        ready_gpus = wait_for_ready_gpus(
            required_gpu_count=args.required_gpu_count,
            max_used_memory_mb=args.max_used_memory_mb,
            poll_seconds=args.poll_seconds,
            master_log_path=paths.master_log_path,
        )

    if args.dry_run:
        return

    shard_output_dirs: List[Path] = []
    shard_sessions: List[Dict[str, str]] = []
    for shard_index, gpu_index in enumerate(ready_gpus[: args.num_shards]):
        shard_session = f"{args.session_name}_shard{shard_index}"
        shard_output_dir = paths.run_root / f"shard{shard_index}"
        shard_output_dirs.append(shard_output_dir)
        command = shard_command(
            repo_root=repo_root,
            shard_index=shard_index,
            gpu_index=gpu_index,
            shard_logs=shard_logs[shard_index],
            args=args,
            paths=paths,
        )
        start_tmux_session(shard_session, command)
        time.sleep(args.startup_wait_seconds)
        shard_sessions.append(
            {
                "session_name": shard_session,
                "pid": session_pid(shard_session),
                "gpu_index": str(gpu_index),
                "log_path": str(paths.shards_root / f"shard{shard_index}.log"),
            }
        )

    (paths.run_root / "launched_shards.json").write_text(
        json.dumps(shard_sessions, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    wait_for_shards(
        shard_output_dirs=shard_output_dirs,
        poll_seconds=args.poll_seconds,
        master_log_path=paths.master_log_path,
    )
    summary = collect_summary(paths.run_root)
    (paths.run_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
