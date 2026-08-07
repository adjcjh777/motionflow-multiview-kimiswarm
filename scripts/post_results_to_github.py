#!/usr/bin/env python3
"""Post a result summary comment to a GitHub issue.

Example (dry-run):
    python scripts/post_results_to_github.py \
        --issue 76 \
        --json outputs/eval_jsons/pp_best_mpi.json \
        --csv outputs/robustness_matrix_bayesian_tri_v2_pp.csv \
        --title "v4 smoke eval" \
        --dry-run

Example (post):
    export GITHUB_TOKEN=ghp_xxx
    python scripts/post_results_to_github.py \
        --issue 76 \
        --json outputs/eval_jsons/pp_best_mpi.json \
        --title "v4 A800 eval results"
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError


DEFAULT_REPO = "adjcjh777/motionflow-multiview-kimiswarm"
ISSUE_API = "https://api.github.com/repos/{repo}/issues/{issue}/comments"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Post a formatted result summary to a GitHub issue."
    )
    parser.add_argument("--issue", type=int, required=True, help="GitHub issue number")
    parser.add_argument(
        "--json",
        dest="json_paths",
        action="append",
        default=[],
        help="Path to a result JSON file (may be repeated)",
    )
    parser.add_argument(
        "--csv",
        dest="csv_paths",
        action="append",
        default=[],
        help="Path to a result CSV file (may be repeated)",
    )
    parser.add_argument(
        "--title",
        default="MotionFlow-MultiView v4 result summary",
        help="Title for the comment summary",
    )
    parser.add_argument(
        "--repo",
        default=os.getenv("GITHUB_REPO", DEFAULT_REPO),
        help=f"GitHub repo slug (default: {DEFAULT_REPO})",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("GITHUB_TOKEN"),
        help="GitHub personal access token (default: $GITHUB_TOKEN)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the comment body instead of posting",
    )
    parser.add_argument(
        "--max-csv-rows",
        type=int,
        default=40,
        help="Maximum number of CSV rows to inline in the comment",
    )
    return parser.parse_args(argv)


def load_json_metrics(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON file {path} does not contain a top-level object")
    return data


def format_json_metrics(path: Path, data: dict) -> str:
    lines = [f"### JSON: `{path.name}`"]
    metrics = [
        ("MPJPE (mm)", "mpjpe_mm"),
        ("PA-MPJPE (mm)", "pa_mpjpe_mm"),
        ("PCK@50", "pck_50"),
        ("PCK@100", "pck_100"),
        ("PCK@150", "pck_150"),
        ("AUC", "auc"),
    ]
    for label, key in metrics:
        value = data.get(key)
        if value is None:
            continue
        if isinstance(value, float):
            lines.append(f"- **{label}**: {value:.4f}")
        else:
            lines.append(f"- **{label}**: {value}")
    extra = {k: v for k, v in data.items() if k not in {m[1] for m in metrics} and v is not None}
    if extra:
        lines.append("- **Other keys**: " + ", ".join(sorted(extra.keys())))
    return "\n".join(lines)


def load_csv_table(path: Path) -> list:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
    return rows


def format_csv_table(path: Path, rows: list, max_rows: int = 40) -> str:
    if not rows:
        return f"### CSV: `{path.name}`\n(empty file)"

    header = rows[0]
    body = rows[1:]
    truncated = False
    if len(body) > max_rows:
        body = body[:max_rows]
        truncated = True

    # Build a GitHub-flavored markdown table
    md = [f"### CSV: `{path.name}`"]
    md.append(" | ".join(header))
    md.append(" | ".join(["---"] * len(header)))
    for row in body:
        md.append(" | ".join(row))
    if truncated:
        md.append(f"\n_Truncated to first {max_rows} data rows; full file: `{path}`_")
    return "\n".join(md)


def build_comment_body(args, json_sections, csv_sections) -> str:
    lines = [
        f"## {args.title}",
        "",
        f"**Repository:** `{args.repo}`  ",
        f"**Issue:** #{args.issue}  ",
        f"**Posted by:** `scripts/post_results_to_github.py`  ",
        "",
    ]

    if json_sections:
        lines.append("## JSON metrics")
        lines.append("")
        lines.extend(json_sections)
        lines.append("")

    if csv_sections:
        if json_sections:
            lines.append("---")
        lines.append("## Robustness / result tables")
        lines.append("")
        lines.extend(csv_sections)
        lines.append("")

    if not json_sections and not csv_sections:
        lines.append("_No result files were attached to this summary._")
        lines.append("")

    lines.append("---")
    lines.append("_This comment was generated automatically by T20 automation._")
    return "\n".join(lines)


def post_comment(token: str, repo: str, issue: int, body: str, dry_run: bool = False):
    if dry_run:
        print("=== DRY-RUN: comment body ===")
        print(body)
        print("=== END DRY-RUN ===")
        return

    if not token:
        raise RuntimeError(
            "GITHUB_TOKEN is not set. Pass --token or set the environment variable."
        )

    url = ISSUE_API.format(repo=repo, issue=issue)
    payload = json.dumps({"body": body}).encode("utf-8")
    req = Request(
        url,
        data=payload,
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "User-Agent": "motionflow-multiview-post-results/1.0",
        },
        method="POST",
    )

    try:
        with urlopen(req) as resp:
            response_data = json.load(resp)
            print(f"Comment posted: {response_data.get('html_url')}")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"GitHub API error {exc.code}: {body}") from exc


def main(argv=None):
    args = parse_args(argv)

    json_sections = []
    for p in args.json_paths:
        path = Path(p)
        if not path.exists():
            print(f"Warning: JSON file not found: {path}", file=sys.stderr)
            continue
        try:
            data = load_json_metrics(path)
            json_sections.append(format_json_metrics(path, data))
        except Exception as exc:
            print(f"Warning: failed to parse {path}: {exc}", file=sys.stderr)

    csv_sections = []
    for p in args.csv_paths:
        path = Path(p)
        if not path.exists():
            print(f"Warning: CSV file not found: {path}", file=sys.stderr)
            continue
        try:
            rows = load_csv_table(path)
            csv_sections.append(format_csv_table(path, rows, args.max_csv_rows))
        except Exception as exc:
            print(f"Warning: failed to parse {path}: {exc}", file=sys.stderr)

    body = build_comment_body(args, json_sections, csv_sections)
    post_comment(args.token, args.repo, args.issue, body, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
