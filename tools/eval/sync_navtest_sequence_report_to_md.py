#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
            }
            for row in reader
        ]
    return rows


def _render_block(report_path: Path, rows: list[dict[str, str]]) -> str:
    lines = [
        START_MARKER,
        "## Navtest Fullset 自动同步",
        "",
        f"来源：`{report_path}`",
        "",
        "| 顺序 | 项目 | 状态 | summary |",
        "| --- | --- | --- | --- |",
    ]

    if not rows:
        lines.append("| - | - | - | - |")
    else:
        for row in sorted(rows, key=lambda item: item["order"]):
            summary = row["summary"] if row["summary"] else "-"
            lines.append(
                f"| {row['order'] or '-'} | `{row['name'] or '-'}` | `{row['status'] or '-'}` | `{summary}` |"
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
    parser.add_argument("--report", type=str, required=True, help="Path to result_index.tsv")
    parser.add_argument("--md", type=str, required=True, help="Path to markdown file")
    args = parser.parse_args()

    report_path = Path(args.report).resolve()
    md_path = Path(args.md).resolve()

    rows = _read_report(report_path)
    block_text = _render_block(report_path, rows)

    old_text = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
    new_text = _upsert_block(old_text, block_text)
    md_path.write_text(new_text, encoding="utf-8")
    print(f"[synced] rows={len(rows)} md={md_path}")


if __name__ == "__main__":
    main()
