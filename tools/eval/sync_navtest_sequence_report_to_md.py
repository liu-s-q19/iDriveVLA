#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


START_MARKER = "<!-- NAVTEST_FULLSET_AUTO_START -->"
END_MARKER = "<!-- NAVTEST_FULLSET_AUTO_END -->"


def _read_report(report_path: Path) -> list[dict[str, str]]:
    if not report_path.exists():
        return []
    with report_path.open("r", encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle, delimiter="\t")
        rows = [
            {
                "order": (row.get("order") or "").strip(),
                "name": (row.get("name") or "").strip(),
                "status": (row.get("status") or "").strip(),
                "summary": (row.get("summary") or "").strip(),
                "score_mean": "-",
            }
            for row in reader
        ]
    return rows


def _read_score_mean(summary_path_str: str) -> str:
    summary_path = Path(summary_path_str)
    if not summary_path_str or summary_path_str == "-" or not summary_path.exists():
        return "-"
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return "-"
    score = payload.get("score_mean")
    return str(score) if score is not None else "-"


def _sort_key(item: dict[str, str]) -> tuple[int, int | str]:
    order = item.get("order", "")
    if order.isdigit():
        return (0, int(order))
    return (1, order)


def _parse_extra_row(extra_row: str) -> dict[str, str]:
    # format: order|name|status|summary_path
    parts = [part.strip() for part in extra_row.split("|")]
    if len(parts) != 4:
        raise ValueError(f"Invalid --extra-row: {extra_row!r}. Expected: order|name|status|summary_path")
    return {
        "order": parts[0],
        "name": parts[1],
        "status": parts[2],
        "summary": parts[3],
        "score_mean": _read_score_mean(parts[3]),
    }


def _render_block(report_paths: list[Path], rows: list[dict[str, str]]) -> str:
    lines = [
        START_MARKER,
        "## Navtest Fullset 自动同步",
        "",
        "来源：",
    ]
    for report_path in report_paths:
        lines.append(f"- `{report_path}`")
    lines.extend(
        [
            "",
            "| 顺序 | 项目 | 状态 | score_mean | summary |",
            "| --- | --- | --- | --- | --- |",
        ]
    )

    if not rows:
        lines.append("| - | - | - | - | - |")
    else:
        for row in sorted(rows, key=_sort_key):
            summary = row["summary"] if row["summary"] else "-"
            score_mean = row["score_mean"] if row["score_mean"] else "-"
            lines.append(
                f"| {row['order'] or '-'} | `{row['name'] or '-'}` | `{row['status'] or '-'}` | `{score_mean}` | `{summary}` |"
            )

    lines.extend(["", END_MARKER])
    return "\n".join(lines)


def _upsert_block(md_text: str, block_text: str) -> str:
    start_idx = md_text.find(START_MARKER)
    end_idx = md_text.find(END_MARKER)

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        end_idx = end_idx + len(END_MARKER)
        return md_text[:start_idx].rstrip() + "\n\n" + block_text + "\n"

    return md_text.rstrip() + "\n\n" + block_text + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync navtest sequence report tsv to markdown section.")
    parser.add_argument(
        "--report",
        action="append",
        required=True,
        help="Path to result_index.tsv. Can be specified multiple times.",
    )
    parser.add_argument(
        "--extra-row",
        action="append",
        default=[],
        help="Append a row in format: order|name|status|summary_path. Can be specified multiple times.",
    )
    parser.add_argument("--md", type=str, required=True, help="Path to markdown file")
    args = parser.parse_args()

    report_paths = [Path(item).resolve() for item in args.report]
    md_path = Path(args.md).resolve()

    rows: list[dict[str, str]] = []
    for report_path in report_paths:
        report_rows = _read_report(report_path)
        for row in report_rows:
            row["score_mean"] = _read_score_mean(row["summary"])
        rows.extend(report_rows)

    for extra_row in args.extra_row:
        rows.append(_parse_extra_row(extra_row))

    block_text = _render_block(report_paths, rows)

    old_text = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
    new_text = _upsert_block(old_text, block_text)
    md_path.write_text(new_text, encoding="utf-8")
    print(f"[synced] reports={len(report_paths)} rows={len(rows)} md={md_path}")


if __name__ == "__main__":
    main()
