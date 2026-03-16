import csv
import re
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Mapping, Sequence


FAILED_TOKEN_RE = re.compile(r"Agent failed for token ([^:]+):")
RESULTS_PATH_RE = re.compile(r"Results are stored in:\s+(.+?)(?:\.)?\s*$")


def is_aggregate_row(row: Mapping[str, str]) -> bool:
    token = str(row.get("token", "")).strip().lower()
    return token in {"", "average"}


def scenario_rows(rows: Iterable[Mapping[str, str]]) -> List[Dict[str, str]]:
    return [dict(row) for row in rows if not is_aggregate_row(row)]


def extract_failed_tokens_from_log(log_path: Path) -> List[str]:
    failed_tokens: List[str] = []
    seen = set()
    for line in Path(log_path).read_text().splitlines():
        match = FAILED_TOKEN_RE.search(line)
        if not match:
            continue
        token = match.group(1)
        if token in seen:
            continue
        seen.add(token)
        failed_tokens.append(token)
    return failed_tokens


def find_results_csv_from_log(log_path: Path) -> Path:
    for line in Path(log_path).read_text().splitlines():
        match = RESULTS_PATH_RE.search(line)
        if match:
            return Path(match.group(1).rstrip("."))
    raise ValueError(f"Results csv path not found in log: {log_path}")


def summarize_csv_rows(
    rows: Sequence[Mapping[str, str]],
    failed_tokens: Sequence[str] = (),
) -> Dict[str, object]:
    filtered_rows = scenario_rows(rows)
    valid_rows = [row for row in filtered_rows if str(row.get("valid", "")).lower() == "true"]
    invalid_rows = [row for row in filtered_rows if str(row.get("valid", "")).lower() == "false"]

    valid_scores = [float(row["score"]) for row in valid_rows if str(row.get("score", "")).strip()]
    all_scores = [float(row["score"]) for row in filtered_rows if str(row.get("score", "")).strip()]
    failed_token_count = len(failed_tokens) if failed_tokens else len(invalid_rows)

    return {
        "num_rows_scenarios": len(filtered_rows),
        "valid_rows": len(valid_rows),
        "invalid_rows": len(invalid_rows),
        "score_mean_valid": mean(valid_scores) if valid_scores else None,
        "score_mean_all_rows": mean(all_scores) if all_scores else None,
        "failed_token_count": failed_token_count,
    }


def load_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    with Path(csv_path).open(newline="") as f:
        return list(csv.DictReader(f))
