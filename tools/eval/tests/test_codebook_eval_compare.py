import json
import subprocess
from pathlib import Path
from typing import Optional

from tools.eval import run_codebook_eval_compare as mod


def _invoke_main(tmp_path: Path, monkeypatch, fail_on_call: Optional[int] = None) -> int:
    call_state = {"count": 0}

    def fake_run(cmd, check, cwd, env, capture_output, text):
        del check, cwd, capture_output, text
        call_state["count"] += 1
        if fail_on_call is not None and call_state["count"] == fail_on_call:
            raise subprocess.CalledProcessError(returncode=9, cmd=cmd, stderr="boom")

        plan_dir = Path(env["PLAN_DIR"])
        merged = plan_dir / "merged"
        merged.mkdir(parents=True, exist_ok=True)

        is_smoke = "run_navhard_two_stage_autovla_current_8gpu.sh" in cmd[1]
        is_old = any("model.config_path=/tmp/old.yaml" in str(x) for x in cmd)

        if is_smoke:
            score = 0.10 if is_old else 0.25
            payload = {"final_extended_pdm_score": score}
        else:
            if is_old:
                score = 0.71
            else:
                score = 0.73 if call_state["count"] == 4 else 0.731
            payload = {"score_mean": score}

        (merged / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    plan_dir = tmp_path / "cmp_plan"
    return mod.main(
        [
            "--root-dir",
            str(tmp_path),
            "--plan-dir",
            str(plan_dir),
            "--ckpt-path",
            "/tmp/demo.ckpt",
            "--old-model-config-path",
            "/tmp/old.yaml",
            "--new-model-config-path",
            "/tmp/new.yaml",
            "--run-smoke",
            "--run-full",
            "--run-final-rerun",
        ]
    )


def test_main_generates_compare_reports_and_records_paths(tmp_path, monkeypatch):
    exit_code = _invoke_main(tmp_path, monkeypatch)
    assert exit_code == 0

    report_json = tmp_path / "cmp_plan" / "compare_report.json"
    report_md = tmp_path / "cmp_plan" / "compare_report.md"

    assert report_json.exists()
    assert report_md.exists()

    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["status"] == "success"
    assert payload["smoke"]["improved"] is True
    assert payload["full"]["winner"] == "new"
    assert payload["final_rerun"]["winner"] == "new"
    assert Path(payload["smoke"]["old"]["summary_path"]).exists()
    assert Path(payload["smoke"]["new"]["summary_path"]).exists()
    assert Path(payload["full"]["old"]["summary_path"]).exists()
    assert Path(payload["full"]["new"]["summary_path"]).exists()
    assert Path(payload["final_rerun"]["summary_path"]).exists()


def test_main_writes_error_report_and_returns_non_zero_on_subtask_failure(tmp_path, monkeypatch):
    exit_code = _invoke_main(tmp_path, monkeypatch, fail_on_call=2)
    assert exit_code == 1

    report_json = tmp_path / "cmp_plan" / "compare_report.json"
    assert report_json.exists()
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert "boom" in payload["error"]["stderr"]
