from pathlib import Path
import subprocess


def test_navhard_multi_gpu_script_exists_and_points_to_current_sharded_pipeline():
    script_path = Path("/data/liushiqi/AutoVLA/scripts/eval/run_navhard_two_stage_autovla_current_8gpu.sh")

    assert script_path.exists(), "current 8GPU navhard eval script should exist"

    content = script_path.read_text(encoding="utf-8")
    assert "tools/eval/prepare_navhard_two_stage_shards.py" in content
    assert "tools/eval/run_navhard_two_stage_autovla_shard.py" in content
    assert "tools/eval/merge_navhard_two_stage_shards.py" in content
    assert "NUPLAN_MAPS_ROOT" in content
    assert "OPENSCENE_DATA_ROOT" in content
    assert "NUPLAN_MAP_VERSION" in content
    assert "CKPT_PATH" in content
    assert "MAX_STAGE_ONE" in content
    assert "MAX_STAGE_TWO" in content

    subprocess.run(["bash", "-n", str(script_path)], check=True)
