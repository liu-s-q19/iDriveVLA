from pathlib import Path
import subprocess


def test_navhard_recogdrive_multi_gpu_script_exists_and_points_to_recogdrive_config():
    script_path = Path("/data/liushiqi/AutoVLA/scripts/eval/run_navhard_two_stage_autovla_recogdrive_vlm2b_8gpu.sh")

    assert script_path.exists(), "recogdrive navhard 8GPU eval script should exist"

    content = script_path.read_text(encoding="utf-8")
    assert "config/eval/navhard_two_stage_autovla_recogdrive_vlm2b_sft8_epoch4.yaml" in content
    assert "scripts/eval/run_navhard_two_stage_autovla_current_8gpu.sh" in content
    assert "GPU_LIST" in content
    assert "PLAN_DIR" in content

    subprocess.run(["bash", "-n", str(script_path)], check=True)


def test_navhard_recogdrive_single_machine_script_exists_and_points_to_recogdrive_config():
    script_path = Path("/data/liushiqi/AutoVLA/scripts/eval/run_navhard_two_stage_autovla_recogdrive_vlm2b_single.sh")

    assert script_path.exists(), "recogdrive navhard single-machine eval script should exist"

    content = script_path.read_text(encoding="utf-8")
    assert "config/eval/navhard_two_stage_autovla_recogdrive_vlm2b_sft8_epoch4.yaml" in content
    assert "scripts/eval/run_navhard_two_stage_autovla_single.sh" in content
    assert "MAX_STAGE_ONE" in content
    assert "MAX_STAGE_TWO" in content

    subprocess.run(["bash", "-n", str(script_path)], check=True)
