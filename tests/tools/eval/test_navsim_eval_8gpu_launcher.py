from tools.eval.navsim_eval_8gpu_launcher import (
    GPUsNotReady,
    collect_summary,
    plan_log_shards,
    shard_done,
    select_ready_gpus,
)


def test_plan_log_shards_balances_evenly():
    log_names = [f"log_{idx}" for idx in range(17)]

    shards = plan_log_shards(log_names, num_shards=8)

    assert len(shards) == 8
    assert sum(len(shard) for shard in shards) == 17
    assert max(len(shard) for shard in shards) - min(len(shard) for shard in shards) <= 1
    assert shards[0] == ["log_0", "log_8", "log_16"]
    assert shards[7] == ["log_7", "log_15"]


def test_select_ready_gpus_returns_first_ready_devices():
    gpu_stats = [
        {"index": 0, "memory_used_mb": 512},
        {"index": 1, "memory_used_mb": 1024},
        {"index": 2, "memory_used_mb": 2048},
        {"index": 3, "memory_used_mb": 128},
    ]

    selected = select_ready_gpus(
        gpu_stats,
        required_gpu_count=3,
        max_used_memory_mb=2048,
    )

    assert selected == [0, 1, 2]


def test_select_ready_gpus_raises_when_not_enough_devices():
    gpu_stats = [
        {"index": 0, "memory_used_mb": 4096},
        {"index": 1, "memory_used_mb": 1024},
    ]

    try:
        select_ready_gpus(
            gpu_stats,
            required_gpu_count=2,
            max_used_memory_mb=2048,
        )
    except GPUsNotReady as exc:
        assert "required=2" in str(exc)
        assert "ready=1" in str(exc)
    else:
        raise AssertionError("Expected GPUsNotReady to be raised")


def test_collect_summary_ignores_shards_log_dir(tmp_path):
    run_root = tmp_path / "run"
    (run_root / "shards").mkdir(parents=True)

    for shard_idx in range(2):
        shard_dir = run_root / f"shard{shard_idx}"
        shard_dir.mkdir()
        csv_path = shard_dir / f"rows_{shard_idx}.csv"
        csv_path.write_text("token,valid,score\n"
                            f"tok_{shard_idx},True,0.{shard_idx + 5}\n",
                            encoding="utf-8")
        (shard_dir / "summary.json").write_text(
            f'{{"csv_path": "{csv_path}"}}',
            encoding="utf-8",
        )
        (run_root / "shards" / f"shard{shard_idx}.log").write_text("", encoding="utf-8")

    summary = collect_summary(run_root)

    assert summary["num_rows_scenarios"] == 2
    assert summary["valid_rows"] == 2
    assert summary["invalid_rows"] == 0


def test_shard_done_requires_summary_json(tmp_path):
    shard_dir = tmp_path / "shard0"
    shard_dir.mkdir()
    (shard_dir / "partial.csv").write_text("token,valid,score\n", encoding="utf-8")

    assert shard_done(shard_dir) is False

    (shard_dir / "summary.json").write_text('{"csv_path":"partial.csv"}', encoding="utf-8")

    assert shard_done(shard_dir) is True
