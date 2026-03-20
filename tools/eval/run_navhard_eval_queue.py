#!/usr/bin/env python3
import argparse
import json
import os
import socket
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml


def _now_ts() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")


def build_default_tasks(root_dir: str) -> List[Dict[str, Any]]:
    root = Path(root_dir).resolve()
    tasks: List[Dict[str, Any]] = [
        {
            "id": "sft10_epoch4_pose8eval",
            "label": "历史10点 SFT epoch4",
            "checkpoint_path": str(root / "runs/sft/2026-03-12_06-03-13/epoch=4-loss=0.9352.ckpt"),
            "train_num_poses": 10,
            "status": "pending",
        },
        {
            "id": "rft10_baseline_step6000_pose8eval",
            "label": "历史10点 baseline step6000",
            "checkpoint_path": str(
                root / "runs/grpo/grpo_navsimv2_sft20260312e4_ip190_2026-03-16_13-23-06/ckpt/rft-step6000-reward6.2188.ckpt"
            ),
            "train_num_poses": 10,
            "status": "pending",
        },
        {
            "id": "rft10_answer_step6000_pose8eval",
            "label": "历史10点 answer-format step6000",
            "checkpoint_path": str(
                root
                / ".worktrees/navsimv2-answer-block/runs/grpo/grpo_navsimv2_answer_protocol_12000_2026-03-17/ckpt/rft-step6000-reward6.2500.ckpt"
            ),
            "train_num_poses": 10,
            "status": "pending",
        },
        {
            "id": "rft10_answer_step12000_pose8eval",
            "label": "历史10点 answer-format step12000",
            "checkpoint_path": str(
                root
                / ".worktrees/navsimv2-answer-block/runs/grpo/grpo_navsimv2_answer_protocol_12000_2026-03-17/ckpt/rft-step12000-reward6.6875.ckpt"
            ),
            "train_num_poses": 10,
            "status": "pending",
        },
        {
            "id": "sft8_epoch4_pose8eval",
            "label": "当前8点 SFT epoch4",
            "checkpoint_path": str(root / "runs/sft/navsimv2_sft8_local_8gpu_retry3_2026-03-18_09-14-56/epoch=4-loss=0.8492.ckpt"),
            "train_num_poses": 8,
            "status": "pending",
        },
        {
            "id": "rft8_default_step6000_pose8eval",
            "label": "当前8点 default-format step6000",
            "checkpoint_path": str(
                root / "runs/grpo/grpo_navsimv2_sft8_default_format_ip33_restart1_2026-03-19_09-16-45/ckpt/rft-step6000-reward6.2500.ckpt"
            ),
            "train_num_poses": 8,
            "status": "pending",
        },
        {
            "id": "rft8_default_step12000_pose8eval",
            "label": "当前8点 default-format step12000",
            "checkpoint_path": str(
                root / "runs/grpo/grpo_navsimv2_sft8_default_format_ip33_restart1_2026-03-19_09-16-45/ckpt/rft-step12000-reward0.ckpt"
            ),
            "train_num_poses": 8,
            "status": "waiting_ckpt",
        },
        {
            "id": "rft8_answer_step6000_pose8eval",
            "label": "当前8点 answer-format step6000",
            "checkpoint_path": str(
                root / "runs/grpo/grpo_navsimv2_sft8_answer_format_local_restart1_2026-03-19_09-16-46/ckpt/rft-step6000-reward5.9375.ckpt"
            ),
            "train_num_poses": 8,
            "status": "pending",
        },
        {
            "id": "rft8_answer_step12000_pose8eval",
            "label": "当前8点 answer-format step12000",
            "checkpoint_path": str(
                root / "runs/grpo/grpo_navsimv2_sft8_answer_format_local_restart1_2026-03-19_09-16-46/ckpt/rft-step12000-reward0.ckpt"
            ),
            "train_num_poses": 8,
            "status": "waiting_ckpt",
        },
    ]

    return [refresh_task_status(task) for task in tasks]


def refresh_task_status(task: Dict[str, Any]) -> Dict[str, Any]:
    refreshed = deepcopy(task)
    ckpt_exists = Path(str(refreshed["checkpoint_path"])).exists()
    refreshed["ready"] = ckpt_exists
    if ckpt_exists:
        if refreshed.get("status") in {"waiting_ckpt", "pending"}:
            refreshed["status"] = "pending"
    else:
        refreshed["status"] = "waiting_ckpt"
    return refreshed


def filter_tasks(tasks: List[Dict[str, Any]], selected_ids: List[str]) -> List[Dict[str, Any]]:
    if not selected_ids:
        return list(tasks)
    by_id = {task["id"]: task for task in tasks}
    filtered = []
    for task_id in selected_ids:
        if task_id not in by_id:
            raise KeyError(f"Unknown task id: {task_id}")
        filtered.append(by_id[task_id])
    return filtered


def build_eval_command(
    root_dir: str,
    task: Dict[str, Any],
    run_dir: Path,
    smoke_stage_one: int,
    smoke_stage_two: int,
    execution_mode: str,
) -> List[str]:
    root_path = Path(root_dir).resolve()
    if execution_mode == "single":
        script_path = root_path / "scripts/eval/run_navhard_two_stage_autovla_single.sh"
        output_key = "OUTPUT_DIR"
    elif execution_mode == "8gpu":
        script_path = root_path / "scripts/eval/run_navhard_two_stage_autovla_current_8gpu.sh"
        output_key = "PLAN_DIR"
    else:
        raise ValueError(f"Unsupported execution_mode={execution_mode}")
    env_parts = [
        f"CKPT_PATH={task['checkpoint_path']}",
        f"{output_key}={run_dir}",
    ]
    if smoke_stage_one > 0:
        env_parts.append(f"MAX_STAGE_ONE={smoke_stage_one}")
    if smoke_stage_two > 0:
        env_parts.append(f"MAX_STAGE_TWO={smoke_stage_two}")
    env_prefix = " ".join(env_parts)
    return ["bash", str(script_path), env_prefix]


def _command_as_shell(command: List[str]) -> str:
    assert len(command) == 3 and command[0] == "bash"
    return f"{command[2]} {command[1]}"


def _load_queue_manifest(queue_dir: Path) -> Dict[str, Any]:
    manifest_path = queue_dir / "manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _write_queue_manifest(queue_dir: Path, manifest: Dict[str, Any]) -> None:
    manifest_path = queue_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _config_is_pose8(config_path: Path) -> bool:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return int(cfg["model"]["trajectory_sampling"]["num_poses"]) == 8


def init_queue(queue_dir: Path, root_dir: str) -> Dict[str, Any]:
    queue_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = queue_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "queue_dir": str(queue_dir),
        "root_dir": str(Path(root_dir).resolve()),
        "host": socket.gethostname(),
        "created_at_utc": _now_ts(),
        "config_path": str(Path(root_dir).resolve() / "config/eval/navhard_two_stage_autovla_rft20260312_step6000.yaml"),
        "tasks": build_default_tasks(root_dir),
    }
    _write_queue_manifest(queue_dir, manifest)
    return manifest


def _run_one_task(
    queue_dir: Path,
    manifest: Dict[str, Any],
    task: Dict[str, Any],
    smoke_stage_one: int,
    smoke_stage_two: int,
    execution_mode: str,
) -> Dict[str, Any]:
    task = refresh_task_status(task)
    root_dir = manifest["root_dir"]
    config_path = Path(manifest["config_path"])

    if not task["ready"]:
        return task
    if not _config_is_pose8(config_path):
        task["status"] = "blocked_bad_config"
        task["error"] = f"{config_path} is not using 8 poses"
        return task

    run_dir = queue_dir / "runs" / task["id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    task["last_run_dir"] = str(run_dir)

    command = build_eval_command(
        root_dir=root_dir,
        task=task,
        run_dir=run_dir,
        smoke_stage_one=smoke_stage_one,
        smoke_stage_two=smoke_stage_two,
        execution_mode=execution_mode,
    )
    shell_cmd = _command_as_shell(command)
    stdout_path = run_dir / "worker_stdout.log"
    stderr_path = run_dir / "worker_stderr.log"

    task["status"] = "running"
    task["started_at_utc"] = _now_ts()
    task["smoke_stage_one"] = smoke_stage_one
    task["smoke_stage_two"] = smoke_stage_two

    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        proc = subprocess.run(
            shell_cmd,
            cwd=root_dir,
            shell=True,
            stdout=out,
            stderr=err,
            text=True,
        )

    task["finished_at_utc"] = _now_ts()
    task["return_code"] = proc.returncode
    task["stdout_log"] = str(stdout_path)
    task["stderr_log"] = str(stderr_path)

    summary_candidates = sorted(run_dir.glob("**/summary.json"))
    resolved_candidates = sorted(run_dir.glob("**/resolved_config.yaml"))
    if summary_candidates:
        task["summary_path"] = str(summary_candidates[-1])
    if resolved_candidates:
        task["resolved_config_path"] = str(resolved_candidates[-1])

    if proc.returncode == 0 and summary_candidates:
        task["status"] = "done"
        task["result"] = json.loads(summary_candidates[-1].read_text(encoding="utf-8"))
    else:
        task["status"] = "failed"

    return task


def run_queue(
    queue_dir: Path,
    smoke_stage_one: int,
    smoke_stage_two: int,
    task_ids: List[str],
    execution_mode: str,
    poll_missing: bool,
    poll_interval_sec: int,
    max_wait_cycles: int,
) -> Dict[str, Any]:
    selected_ids = list(task_ids)
    selected_set = set(selected_ids)
    cycles = 0

    while True:
        manifest = _load_queue_manifest(queue_dir)
        results_path = queue_dir / "results.jsonl"
        if selected_ids:
            filter_tasks(manifest["tasks"], selected_ids)

        tasks: List[Dict[str, Any]] = []
        did_work = False
        has_waiting = False

        for task in manifest["tasks"]:
            current = refresh_task_status(task)
            if selected_set and current["id"] not in selected_set:
                tasks.append(current)
                continue
            if current["status"] == "pending":
                did_work = True
                current = _run_one_task(
                    queue_dir=queue_dir,
                    manifest=manifest,
                    task=current,
                    smoke_stage_one=smoke_stage_one,
                    smoke_stage_two=smoke_stage_two,
                    execution_mode=execution_mode,
                )
                _append_jsonl(results_path, {"task_id": current["id"], "status": current["status"], "ts": _now_ts()})
            elif current["status"] == "waiting_ckpt":
                has_waiting = True
            tasks.append(current)

        manifest["tasks"] = tasks
        manifest["updated_at_utc"] = _now_ts()
        _write_queue_manifest(queue_dir, manifest)

        if did_work:
            cycles = 0
            continue
        if not poll_missing or not has_waiting:
            return manifest
        cycles += 1
        if max_wait_cycles > 0 and cycles >= max_wait_cycles:
            return manifest
        time.sleep(poll_interval_sec)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential navhard evaluation queue runner.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create queue manifest.")
    init_parser.add_argument("--queue-dir", type=str, required=True)
    init_parser.add_argument("--root-dir", type=str, default="/data/liushiqi/AutoVLA")

    run_parser = subparsers.add_parser("run", help="Run pending tasks sequentially.")
    run_parser.add_argument("--queue-dir", type=str, required=True)
    run_parser.add_argument("--smoke-stage-one", type=int, default=1)
    run_parser.add_argument("--smoke-stage-two", type=int, default=1)
    run_parser.add_argument("--task-id", action="append", default=[], help="Optional task id filter.")
    run_parser.add_argument("--execution-mode", choices=["single", "8gpu"], default="single")
    run_parser.add_argument("--poll-missing", action="store_true", help="Keep polling waiting_ckpt tasks.")
    run_parser.add_argument("--poll-interval-sec", type=int, default=600)
    run_parser.add_argument("--max-wait-cycles", type=int, default=0, help="0 means no cycle limit.")

    args = parser.parse_args()

    if args.command == "init":
        manifest = init_queue(Path(args.queue_dir).resolve(), args.root_dir)
        print(json.dumps({"queue_dir": manifest["queue_dir"], "num_tasks": len(manifest["tasks"])}, ensure_ascii=False))
        return

    if args.command == "run":
        manifest = run_queue(
            queue_dir=Path(args.queue_dir).resolve(),
            smoke_stage_one=int(args.smoke_stage_one),
            smoke_stage_two=int(args.smoke_stage_two),
            task_ids=list(args.task_id),
            execution_mode=str(args.execution_mode),
            poll_missing=bool(args.poll_missing),
            poll_interval_sec=int(args.poll_interval_sec),
            max_wait_cycles=int(args.max_wait_cycles),
        )
        done = sum(1 for task in manifest["tasks"] if task["status"] == "done")
        waiting = sum(1 for task in manifest["tasks"] if task["status"] == "waiting_ckpt")
        failed = sum(1 for task in manifest["tasks"] if task["status"] == "failed")
        print(json.dumps({"queue_dir": manifest["queue_dir"], "done": done, "waiting_ckpt": waiting, "failed": failed}, ensure_ascii=False))
        return

    raise RuntimeError(f"unsupported command={args.command}")


if __name__ == "__main__":
    main()
