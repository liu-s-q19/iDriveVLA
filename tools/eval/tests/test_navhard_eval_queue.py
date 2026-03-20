from pathlib import Path

from tools.eval import run_navhard_eval_queue as mod


def test_default_tasks_include_expected_entries_and_waiting_step12000():
    tasks = mod.build_default_tasks("/data/liushiqi/AutoVLA")
    task_ids = [task["id"] for task in tasks]

    assert "sft10_epoch4_pose8eval" in task_ids
    assert "rft10_baseline_step6000_pose8eval" in task_ids
    assert "rft10_answer_step12000_pose8eval" in task_ids
    assert "sft8_epoch4_pose8eval" in task_ids
    assert "rft8_default_step6000_pose8eval" in task_ids
    assert "rft8_default_step12000_pose8eval" in task_ids
    assert "rft8_answer_step12000_pose8eval" in task_ids

    waiting_ids = {task["id"] for task in tasks if task["status"] == "waiting_ckpt"}
    assert "rft8_default_step12000_pose8eval" in waiting_ids
    assert "rft8_answer_step12000_pose8eval" in waiting_ids


def test_build_eval_command_uses_current_pose8_single_machine_script(tmp_path):
    task = {
        "id": "demo",
        "checkpoint_path": "/tmp/demo.ckpt",
        "status": "pending",
    }

    run_dir = tmp_path / "demo"
    command = mod.build_eval_command(
        root_dir="/data/liushiqi/AutoVLA",
        task=task,
        run_dir=run_dir,
        smoke_stage_one=1,
        smoke_stage_two=1,
        execution_mode="single",
    )

    assert command[0] == "bash"
    assert command[1].endswith("scripts/eval/run_navhard_two_stage_autovla_single.sh")
    assert "CKPT_PATH=/tmp/demo.ckpt" in command[2]
    assert "MAX_STAGE_ONE=1" in command[2]
    assert "MAX_STAGE_TWO=1" in command[2]
    assert f"OUTPUT_DIR={run_dir}" in command[2]


def test_build_eval_command_omits_stage_caps_in_full_mode(tmp_path):
    task = {
        "id": "demo",
        "checkpoint_path": "/tmp/demo.ckpt",
        "status": "pending",
    }

    run_dir = tmp_path / "demo"
    command = mod.build_eval_command(
        root_dir="/data/liushiqi/AutoVLA",
        task=task,
        run_dir=run_dir,
        smoke_stage_one=0,
        smoke_stage_two=0,
        execution_mode="single",
    )

    assert "MAX_STAGE_ONE=" not in command[2]
    assert "MAX_STAGE_TWO=" not in command[2]


def test_refresh_task_status_marks_missing_ckpt_as_waiting():
    task = {
        "id": "missing-demo",
        "checkpoint_path": "/tmp/definitely_missing_navhard_eval_queue.ckpt",
        "status": "pending",
    }

    refreshed = mod.refresh_task_status(task)

    assert refreshed["status"] == "waiting_ckpt"
    assert refreshed["ready"] is False


def test_filter_task_ids_keeps_requested_order():
    tasks = mod.build_default_tasks("/data/liushiqi/AutoVLA")

    filtered = mod.filter_tasks(tasks, ["rft8_answer_step6000_pose8eval", "sft8_epoch4_pose8eval"])

    assert [task["id"] for task in filtered] == [
        "rft8_answer_step6000_pose8eval",
        "sft8_epoch4_pose8eval",
    ]


def test_build_eval_command_uses_current_pose8_8gpu_script(tmp_path):
    task = {
        "id": "demo",
        "checkpoint_path": "/tmp/demo.ckpt",
        "status": "pending",
    }

    run_dir = tmp_path / "demo"
    command = mod.build_eval_command(
        root_dir="/data/liushiqi/AutoVLA",
        task=task,
        run_dir=run_dir,
        smoke_stage_one=0,
        smoke_stage_two=0,
        execution_mode="8gpu",
    )

    assert command[0] == "bash"
    assert command[1].endswith("scripts/eval/run_navhard_two_stage_autovla_current_8gpu.sh")
    assert "PLAN_DIR=" in command[2]
    assert f"PLAN_DIR={run_dir}" in command[2]
    assert "CKPT_PATH=/tmp/demo.ckpt" in command[2]


def test_run_queue_can_promote_waiting_ckpt_when_file_appears(tmp_path, monkeypatch):
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    ckpt_path = tmp_path / "appeared.ckpt"

    manifest = {
        "queue_dir": str(queue_dir),
        "root_dir": "/data/liushiqi/AutoVLA",
        "config_path": "/data/liushiqi/AutoVLA/config/eval/navhard_two_stage_autovla_rft20260312_step6000.yaml",
        "tasks": [
            {
                "id": "appeared",
                "label": "appeared",
                "checkpoint_path": str(ckpt_path),
                "train_num_poses": 8,
                "status": "waiting_ckpt",
            }
        ],
    }
    mod._write_queue_manifest(queue_dir, manifest)

    def fake_run_one_task(**kwargs):
        task = dict(kwargs["task"])
        task["status"] = "done"
        return task

    monkeypatch.setattr(mod, "_run_one_task", fake_run_one_task)
    monkeypatch.setattr(mod, "_config_is_pose8", lambda _: True)

    ckpt_path.write_text("x", encoding="utf-8")
    updated = mod.run_queue(
        queue_dir=queue_dir,
        smoke_stage_one=1,
        smoke_stage_two=1,
        task_ids=[],
        execution_mode="single",
        poll_missing=False,
        poll_interval_sec=1,
        max_wait_cycles=1,
    )

    assert updated["tasks"][0]["status"] == "done"
