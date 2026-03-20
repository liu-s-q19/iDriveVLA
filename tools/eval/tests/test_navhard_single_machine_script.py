from pathlib import Path
import subprocess


def test_navhard_single_machine_script_exists_and_points_to_current_evaluator():
    script_path = Path("/data/liushiqi/AutoVLA/scripts/eval/run_navhard_two_stage_autovla_single.sh")

    assert script_path.exists(), "single-machine navhard eval script should exist"

    content = script_path.read_text(encoding="utf-8")
    assert "tools/eval/run_navhard_two_stage_autovla.py" in content
    assert "NUPLAN_MAPS_ROOT" in content
    assert "OPENSCENE_DATA_ROOT" in content
    assert "NUPLAN_MAP_VERSION" in content

    subprocess.run(["bash", "-n", str(script_path)], check=True)
