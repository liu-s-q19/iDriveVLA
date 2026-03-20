from pathlib import Path
import subprocess


def test_navhard_queue_ip190_launcher_script_points_to_queue_runner():
    script_path = Path("/data/liushiqi/AutoVLA/scripts/eval/start_navhard_eval_queue_ip190.sh")

    assert script_path.exists(), "ip190 navhard queue launcher should exist"

    content = script_path.read_text(encoding="utf-8")
    assert "tools/eval/run_navhard_eval_queue.py init" in content
    assert "tools/eval/run_navhard_eval_queue.py run" in content
    assert "nohup" in content
    assert "SSH_HOST" in content
    assert "10.199.7.190" in content
    assert "EXECUTION_MODE" in content
    assert "POLL_MISSING" in content

    subprocess.run(["bash", "-n", str(script_path)], check=True)
