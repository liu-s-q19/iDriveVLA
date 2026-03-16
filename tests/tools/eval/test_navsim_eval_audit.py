from pathlib import Path

from tools.eval.navsim_eval_audit import (
    extract_failed_tokens_from_log,
    find_results_csv_from_log,
    summarize_csv_rows,
)


def test_summarize_csv_rows_excludes_average_row():
    rows = [
        {"token": "tok_a", "valid": "True", "score": "0.8"},
        {"token": "tok_b", "valid": "False", "score": ""},
        {"token": "tok_c", "valid": "True", "score": "0.2"},
        {
            "token": "average",
            "valid": "False",
            "score": "0.5",
        },
    ]

    summary = summarize_csv_rows(rows)

    assert summary["num_rows_scenarios"] == 3
    assert summary["valid_rows"] == 2
    assert summary["invalid_rows"] == 1
    assert summary["failed_token_count"] == 1
    assert summary["score_mean_valid"] == 0.5
    assert summary["score_mean_all_rows"] == 0.5


def test_extract_failed_tokens_from_log_deduplicates_and_preserves_order(tmp_path: Path):
    log_path = tmp_path / "eval.log"
    log_path.write_text(
        "\n".join(
            [
                "WARNING ----------- Agent failed for token tok_b:",
                "Traceback ...",
                "WARNING ----------- Agent failed for token tok_a:",
                "WARNING ----------- Agent failed for token tok_b:",
            ]
        )
    )

    assert extract_failed_tokens_from_log(log_path) == ["tok_b", "tok_a"]


def test_find_results_csv_from_log_extracts_logged_path(tmp_path: Path):
    log_path = tmp_path / "eval.log"
    csv_path = tmp_path / "result.csv"
    log_path.write_text(
        f"Finished.\nResults are stored in: {csv_path}.\n"
    )

    assert find_results_csv_from_log(log_path) == csv_path


def test_summarize_csv_rows_handles_all_invalid_rows_without_score_column():
    rows = [
        {"token": "tok_a", "valid": "False"},
        {"token": "tok_b", "valid": "False"},
    ]

    summary = summarize_csv_rows(rows)

    assert summary["invalid_rows"] == 2
    assert summary["score_mean_valid"] is None
    assert summary["score_mean_all_rows"] is None
