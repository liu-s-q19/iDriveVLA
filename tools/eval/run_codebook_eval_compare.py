from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _stable_fingerprint(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _resolve_flags(args: argparse.Namespace) -> tuple[bool, bool, bool]:
    explicitly_set = args.run_smoke or args.run_full or args.run_final_rerun
    if not explicitly_set:
        return True, True, True
    return args.run_smoke, args.run_full, args.run_final_rerun


def _score_key(stage: str) -> str:
    if stage == "smoke":
        return "final_extended_pdm_score"
    if stage in {"full", "final_rerun"}:
        return "score_mean"
    raise ValueError(f"Unsupported stage: {stage}")


def _read_score(summary_path: Path, stage: str) -> float:
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary not found: {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    key = _score_key(stage)
    if key not in payload:
        raise KeyError(f"Missing score key '{key}' in {summary_path}")
    return float(payload[key])


def _run_eval(
    *,
    root_dir: Path,
    python_bin: str,
    gpu_list: str,
    ckpt_path: str,
    model_config_path: str,
    eval_config_path: str,
    stage: str,
    variant: str,
    run_dir: Path,
    smoke_max_stage_one: Optional[int],
    smoke_max_stage_two: Optional[int],
) -> Dict[str, Any]:
    if stage == "smoke":
        script_path = root_dir / "scripts/eval/run_navhard_two_stage_autovla_current_8gpu.sh"
    elif stage in {"full", "final_rerun"}:
        script_path = root_dir / "scripts/eval/run_navtest_epdms_standard_8gpu.sh"
    else:
        raise ValueError(f"Unsupported stage={stage}")

    cmd: List[str] = [
        "bash",
        str(script_path),
        "--set",
        f"model.config_path={model_config_path}",
    ]
    env = dict()
    env.update(**{
        "ROOT_DIR": str(root_dir),
        "PYTHON_BIN": python_bin,
        "GPU_LIST": gpu_list,
        "CKPT_PATH": ckpt_path,
        "CONFIG_PATH": eval_config_path,
        "PLAN_DIR": str(run_dir),
    })
    if stage == "smoke":
        if smoke_max_stage_one is not None:
            env["MAX_STAGE_ONE"] = str(smoke_max_stage_one)
        if smoke_max_stage_two is not None:
            env["MAX_STAGE_TWO"] = str(smoke_max_stage_two)

    completed = subprocess.run(
        cmd,
        check=True,
        cwd=str(root_dir),
        env={**os.environ, **env},
        capture_output=True,
        text=True,
    )

    summary_path = run_dir / "merged" / "summary.json"
    score = _read_score(summary_path, stage=stage)
    fingerprint = _stable_fingerprint(
        {
            "stage": stage,
            "variant": variant,
            "ckpt_path": ckpt_path,
            "model_config_path": model_config_path,
            "eval_config_path": eval_config_path,
            "summary_path": str(summary_path.resolve()),
            "score": score,
        }
    )
    return {
        "stage": stage,
        "variant": variant,
        "score": score,
        "summary_path": str(summary_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "fingerprint": fingerprint,
        "stdout_tail": completed.stdout[-1200:],
    }


def _render_report_md(report: Dict[str, Any]) -> str:
    lines = [
        "# NavSim v2 Codebook Compare Report",
        "",
        f"- created_at_utc: `{report['created_at_utc']}`",
        f"- status: `{report['status']}`",
        f"- ckpt_path: `{report['inputs']['ckpt_path']}`",
        f"- old_model_config_path: `{report['inputs']['old_model_config_path']}`",
        f"- new_model_config_path: `{report['inputs']['new_model_config_path']}`",
        "",
    ]

    smoke = report.get("smoke", {})
    if smoke:
        lines.extend(
            [
                "## Smoke (navhard 40/40)",
                f"- improved (new > old): `{smoke.get('improved')}`",
                f"- old score: `{smoke.get('old', {}).get('score')}`",
                f"- new score: `{smoke.get('new', {}).get('score')}`",
                f"- old summary: `{smoke.get('old', {}).get('summary_path')}`",
                f"- new summary: `{smoke.get('new', {}).get('summary_path')}`",
                f"- old fingerprint: `{smoke.get('old', {}).get('fingerprint')}`",
                f"- new fingerprint: `{smoke.get('new', {}).get('fingerprint')}`",
                "",
            ]
        )

    full = report.get("full", {})
    if full:
        lines.extend(
            [
                "## Full (navtest old/new)",
                f"- executed: `{full.get('executed')}`",
                f"- winner: `{full.get('winner')}`",
                f"- old score: `{full.get('old', {}).get('score')}`",
                f"- new score: `{full.get('new', {}).get('score')}`",
                f"- old summary: `{full.get('old', {}).get('summary_path')}`",
                f"- new summary: `{full.get('new', {}).get('summary_path')}`",
                f"- old fingerprint: `{full.get('old', {}).get('fingerprint')}`",
                f"- new fingerprint: `{full.get('new', {}).get('fingerprint')}`",
                "",
            ]
        )

    final_rerun = report.get("final_rerun", {})
    if final_rerun:
        lines.extend(
            [
                "## Final Rerun",
                f"- executed: `{final_rerun.get('executed')}`",
                f"- winner: `{final_rerun.get('winner')}`",
                f"- score: `{final_rerun.get('score')}`",
                f"- summary: `{final_rerun.get('summary_path')}`",
                f"- fingerprint: `{final_rerun.get('fingerprint')}`",
                "",
            ]
        )

    if report.get("error"):
        lines.extend(
            [
                "## Error",
                f"- stage: `{report['error'].get('stage')}`",
                f"- variant: `{report['error'].get('variant')}`",
                f"- returncode: `{report['error'].get('returncode')}`",
                f"- stderr: `{report['error'].get('stderr')}`",
            ]
        )

    return "\n".join(lines).strip() + "\n"


def _write_reports(plan_dir: Path, report: Dict[str, Any]) -> None:
    plan_dir.mkdir(parents=True, exist_ok=True)
    json_path = plan_dir / "compare_report.json"
    md_path = plan_dir / "compare_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(_render_report_md(report), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare old/new codebook model.config_path through existing navhard/navtest pipelines.")
    parser.add_argument("--root-dir", type=Path, default=Path("/data/liushiqi/AutoVLA"))
    parser.add_argument("--plan-dir", type=Path, default=None)
    parser.add_argument("--python-bin", default="/data/miniconda/envs/autolsqv2/bin/python")
    parser.add_argument("--gpu-list", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--old-model-config-path", required=True)
    parser.add_argument("--new-model-config-path", required=True)
    parser.add_argument(
        "--smoke-config-template",
        default="config/eval/navhard_two_stage_autovla_recogdrive_vlm2b_sft8_epoch4.yaml",
    )
    parser.add_argument(
        "--full-config-template",
        default="config/eval/navsimv2_epdms_standard_autovla_recogdrive_vlm2b_sft8_epoch4.yaml",
    )
    parser.add_argument("--smoke-max-stage-one", type=int, default=40)
    parser.add_argument("--smoke-max-stage-two", type=int, default=40)
    parser.add_argument("--run-smoke", action="store_true")
    parser.add_argument("--run-full", action="store_true")
    parser.add_argument("--run-final-rerun", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    run_smoke, run_full, run_final_rerun = _resolve_flags(args)
    root_dir = args.root_dir.resolve()
    plan_dir = args.plan_dir.resolve() if args.plan_dir else (
        root_dir / "logs/eval" / f"codebook_cmp_e9_{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')}"
    )

    smoke_config_path = (root_dir / args.smoke_config_template).resolve()
    full_config_path = (root_dir / args.full_config_template).resolve()

    report: Dict[str, Any] = {
        "created_at_utc": _utc_now(),
        "status": "running",
        "inputs": {
            "root_dir": str(root_dir),
            "plan_dir": str(plan_dir),
            "python_bin": args.python_bin,
            "gpu_list": args.gpu_list,
            "ckpt_path": args.ckpt_path,
            "old_model_config_path": args.old_model_config_path,
            "new_model_config_path": args.new_model_config_path,
            "smoke_config_template": str(smoke_config_path),
            "full_config_template": str(full_config_path),
            "smoke_max_stage_one": args.smoke_max_stage_one,
            "smoke_max_stage_two": args.smoke_max_stage_two,
        },
        "policy": {
            "smoke_gate": "new > old",
            "full_primary_metric": "score_mean",
            "smoke_primary_metric": "final_extended_pdm_score",
        },
    }
    _write_reports(plan_dir, report)

    try:
        smoke_passed_gate = True
        if run_smoke:
            smoke_old = _run_eval(
                root_dir=root_dir,
                python_bin=args.python_bin,
                gpu_list=args.gpu_list,
                ckpt_path=args.ckpt_path,
                model_config_path=args.old_model_config_path,
                eval_config_path=str(smoke_config_path),
                stage="smoke",
                variant="old",
                run_dir=plan_dir / "smoke_old",
                smoke_max_stage_one=args.smoke_max_stage_one,
                smoke_max_stage_two=args.smoke_max_stage_two,
            )
            smoke_new = _run_eval(
                root_dir=root_dir,
                python_bin=args.python_bin,
                gpu_list=args.gpu_list,
                ckpt_path=args.ckpt_path,
                model_config_path=args.new_model_config_path,
                eval_config_path=str(smoke_config_path),
                stage="smoke",
                variant="new",
                run_dir=plan_dir / "smoke_new",
                smoke_max_stage_one=args.smoke_max_stage_one,
                smoke_max_stage_two=args.smoke_max_stage_two,
            )
            smoke_passed_gate = bool(smoke_new["score"] > smoke_old["score"])
            report["smoke"] = {
                "executed": True,
                "old": smoke_old,
                "new": smoke_new,
                "improved": smoke_passed_gate,
            }
        else:
            report["smoke"] = {
                "executed": False,
                "improved": None,
                "skip_reason": "run_smoke disabled",
            }

        can_run_full = run_full and ((not run_smoke) or smoke_passed_gate)
        if can_run_full:
            full_old = _run_eval(
                root_dir=root_dir,
                python_bin=args.python_bin,
                gpu_list=args.gpu_list,
                ckpt_path=args.ckpt_path,
                model_config_path=args.old_model_config_path,
                eval_config_path=str(full_config_path),
                stage="full",
                variant="old",
                run_dir=plan_dir / "full_old",
                smoke_max_stage_one=None,
                smoke_max_stage_two=None,
            )
            full_new = _run_eval(
                root_dir=root_dir,
                python_bin=args.python_bin,
                gpu_list=args.gpu_list,
                ckpt_path=args.ckpt_path,
                model_config_path=args.new_model_config_path,
                eval_config_path=str(full_config_path),
                stage="full",
                variant="new",
                run_dir=plan_dir / "full_new",
                smoke_max_stage_one=None,
                smoke_max_stage_two=None,
            )
            winner = "new" if full_new["score"] > full_old["score"] else "old"
            report["full"] = {
                "executed": True,
                "old": full_old,
                "new": full_new,
                "winner": winner,
            }
        else:
            report["full"] = {
                "executed": False,
                "winner": None,
                "skip_reason": "smoke gate blocked or run_full disabled",
            }

        if run_final_rerun and report["full"].get("executed"):
            winner = report["full"]["winner"]
            winner_model_cfg = args.new_model_config_path if winner == "new" else args.old_model_config_path
            rerun = _run_eval(
                root_dir=root_dir,
                python_bin=args.python_bin,
                gpu_list=args.gpu_list,
                ckpt_path=args.ckpt_path,
                model_config_path=winner_model_cfg,
                eval_config_path=str(full_config_path),
                stage="final_rerun",
                variant=winner,
                run_dir=plan_dir / f"final_rerun_{winner}",
                smoke_max_stage_one=None,
                smoke_max_stage_two=None,
            )
            report["final_rerun"] = {
                "executed": True,
                "winner": winner,
                **rerun,
            }
        else:
            report["final_rerun"] = {
                "executed": False,
                "winner": None,
                "skip_reason": "run_final_rerun disabled or full stage not executed",
            }

        report["status"] = "success"
        _write_reports(plan_dir, report)
        return 0
    except subprocess.CalledProcessError as exc:
        stage = "unknown"
        variant = "unknown"
        cmd = [str(x) for x in (exc.cmd or [])]
        if "run_navhard_two_stage_autovla_current_8gpu.sh" in " ".join(cmd):
            stage = "smoke"
        elif "run_navtest_epdms_standard_8gpu.sh" in " ".join(cmd):
            stage = "full_or_final_rerun"
        if any("model.config_path=" in x and "old" in x for x in cmd):
            variant = "old"
        elif any("model.config_path=" in x and "new" in x for x in cmd):
            variant = "new"

        report["status"] = "failed"
        report["error"] = {
            "stage": stage,
            "variant": variant,
            "returncode": int(exc.returncode),
            "cmd": cmd,
            "stderr": exc.stderr or "",
            "stdout": exc.stdout or "",
        }
        _write_reports(plan_dir, report)
        return 1
    except Exception as exc:  # pragma: no cover - defensive fallback
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        _write_reports(plan_dir, report)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
