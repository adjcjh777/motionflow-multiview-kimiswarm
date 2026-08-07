#!/usr/bin/env python3
"""Create GitHub issues and pull requests from experiment results via REST API.

This script does **not** require the ``gh`` CLI; it talks directly to the
GitHub REST API using a personal access token.  It is intended to be used by
T20 swarm agents to publish experiment results back to the repository.

Authentication
--------------
The script looks for a GitHub token in the following order:

1. ``--token <TOKEN>`` command-line argument
2. ``GITHUB_TOKEN`` environment variable
3. ``github.token`` key in ``~/.github_token`` (fallback for local runs)

For CI/automation, set ``GITHUB_TOKEN`` in the environment, for example::

    export GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

The repository used by the swarm is::

    adjcjh777/motionflow-multiview-kimiswarm

Examples
--------
Dry-run an issue from experiment results::

    python scripts/create_pr_and_issue.py \
        --type issue \
        --title "v13 temporal baseline smoke results" \
        --body "Initial smoke run for v13_temporal on GPU 0." \
        --json outputs/eval_jsons/v13_smoke_h36m.json \
        --csv outputs/robustness_v13_temporal.csv \
        --label experiment-results \
        --dry-run

Create a real issue::

    export GITHUB_TOKEN=ghp_xxx
    python scripts/create_pr_and_issue.py \
        --type issue \
        --title "v13 temporal baseline smoke results" \
        --body "Initial smoke run for v13_temporal on GPU 0." \
        --json outputs/eval_jsons/v13_smoke_h36m.json \
        --label experiment-results

Create a pull request from a feature branch::

    export GITHUB_TOKEN=ghp_xxx
    python scripts/create_pr_and_issue.py \
        --type pr \
        --title "Add cross-view transformer v17" \
        --body "See linked issue for design details." \
        --head swarm/v17_crossview_transformer \
        --base main

Exit codes
----------
* 0 - success (or dry-run completed)
* 1 - runtime error / API error
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError
from urllib.request import Request, urlopen


DEFAULT_REPO = "adjcjh777/motionflow-multiview-kimiswarm"
API_BASE = "https://api.github.com"
PLACEHOLDER_TOKEN = "ghp_PLACEHOLDER_TOKEN_SET_YOUR_OWN"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create GitHub issues and pull requests from experiment results."
    )
    parser.add_argument(
        "--type",
        choices=["issue", "pr"],
        required=True,
        help="Whether to create an issue or a pull request.",
    )
    parser.add_argument("--title", required=True, help="Title of the issue or PR.")
    parser.add_argument(
        "--body",
        default="",
        help="Initial body text. Result files will be appended if provided.",
    )
    parser.add_argument(
        "--json",
        dest="json_paths",
        action="append",
        default=[],
        help="Path to a JSON metrics file (may be repeated).",
    )
    parser.add_argument(
        "--csv",
        dest="csv_paths",
        action="append",
        default=[],
        help="Path to a CSV results file (may be repeated).",
    )
    parser.add_argument(
        "--label",
        dest="labels",
        action="append",
        default=[],
        help="Label(s) to apply to the issue/PR (may be repeated).",
    )
    parser.add_argument(
        "--head",
        help="Head branch for a pull request (required when --type=pr).",
    )
    parser.add_argument(
        "--base",
        default="main",
        help="Base branch for a pull request (default: main).",
    )
    parser.add_argument(
        "--repo",
        default=os.getenv("GITHUB_REPO", DEFAULT_REPO),
        help=f"GitHub repo slug (default: {DEFAULT_REPO}).",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("GITHUB_TOKEN"),
        help="GitHub personal access token (default: $GITHUB_TOKEN).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the payload and print it without sending anything to GitHub.",
    )
    parser.add_argument(
        "--draft",
        action="store_true",
        help="Create a draft pull request (only used when --type=pr).",
    )
    return parser.parse_args(argv)


def _load_token_file() -> Optional[str]:
    token_path = Path.home() / ".github_token"
    if token_path.exists():
        try:
            return token_path.read_text(encoding="utf-8").strip().splitlines()[0]
        except Exception:
            return None
    return None


def get_token(token: Optional[str]) -> Optional[str]:
    if token:
        return token
    env_token = os.getenv("GITHUB_TOKEN")
    if env_token:
        return env_token
    return _load_token_file()


def load_json_metrics(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON file {path} does not contain a top-level object")
    return data


def load_csv_table(path: Path) -> List[List[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        return [row for row in reader]


def format_json_section(path: Path, data: Dict[str, Any]) -> str:
    lines = [f"### JSON: `{path.name}`"]
    priority_keys = [
        ("MPJPE (mm)", "mpjpe_mm"),
        ("PA-MPJPE (mm)", "pa_mpjpe_mm"),
        ("PCK@50", "pck_50"),
        ("PCK@100", "pck_100"),
        ("PCK@150", "pck_150"),
        ("AUC", "auc"),
        ("step", "step"),
        ("epoch", "epoch"),
    ]
    seen = set()
    for label, key in priority_keys:
        if key in data:
            value = data[key]
            seen.add(key)
            if isinstance(value, float):
                lines.append(f"- **{label}**: {value:.4f}")
            else:
                lines.append(f"- **{label}**: {value}")
    for key, value in sorted(data.items()):
        if key not in seen:
            if isinstance(value, float):
                lines.append(f"- `{key}`: {value:.4f}")
            else:
                lines.append(f"- `{key}`: {value}")
    return "\n".join(lines)


def format_csv_section(path: Path, rows: List[List[str]], max_rows: int = 30) -> str:
    if not rows:
        return f"### CSV: `{path.name}`\n(empty file)"

    header = rows[0]
    body = rows[1:]
    truncated = False
    if len(body) > max_rows:
        body = body[:max_rows]
        truncated = True

    md = [f"### CSV: `{path.name}`"]
    md.append(" | ".join(header))
    md.append(" | ".join(["---"] * len(header)))
    for row in body:
        md.append(" | ".join(row))
    if truncated:
        md.append(f"\n_Truncated to first {max_rows} data rows; full file: `{path}`_")
    return "\n".join(md)


def build_body(args, json_sections: List[str], csv_sections: List[str]) -> str:
    parts = [args.body.strip()]
    if json_sections:
        if parts and parts[0]:
            parts.append("")
        parts.append("## Experiment metrics")
        parts.append("")
        parts.extend(json_sections)
    if csv_sections:
        if json_sections:
            parts.append("")
            parts.append("---")
            parts.append("")
        parts.append("## Result tables")
        parts.append("")
        parts.extend(csv_sections)
    return "\n\n".join(part for part in parts if part or part == "")


def _request(method: str, url: str, payload: Dict[str, Any], token: str) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        url,
        data=data,
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "User-Agent": "motionflow-multiview-github-automation/1.0",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method=method,
    )
    try:
        with urlopen(req, timeout=60) as resp:
            return json.load(resp)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"GitHub API error {exc.code}: {body}") from exc


def create_issue(
    repo: str,
    title: str,
    body: str,
    labels: List[str],
    token: str,
    dry_run: bool = False,
) -> Optional[str]:
    payload: Dict[str, Any] = {"title": title, "body": body}
    if labels:
        payload["labels"] = labels

    if dry_run:
        print("=== DRY-RUN: issue payload ===")
        print(json.dumps(payload, indent=2))
        return None

    url = f"{API_BASE}/repos/{repo}/issues"
    result = _request("POST", url, payload, token)
    return result.get("html_url")


def create_pull_request(
    repo: str,
    title: str,
    body: str,
    head: str,
    base: str,
    token: str,
    draft: bool = False,
    labels: List[str] = None,
    dry_run: bool = False,
) -> Optional[str]:
    payload: Dict[str, Any] = {
        "title": title,
        "body": body,
        "head": head,
        "base": base,
        "draft": draft,
    }

    if dry_run:
        print("=== DRY-RUN: PR payload ===")
        print(json.dumps(payload, indent=2))
        return None

    url = f"{API_BASE}/repos/{repo}/pulls"
    result = _request("POST", url, payload, token)
    pr_url = result.get("html_url")

    if labels:
        try:
            pr_number = result.get("number")
            if pr_number:
                _request("POST", f"{API_BASE}/repos/{repo}/issues/{pr_number}", {"labels": labels}, token)
        except Exception as exc:
            print(f"Warning: could not apply labels to PR: {exc}", file=sys.stderr)

    return pr_url


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.type == "pr" and not args.head:
        print("Error: --head is required when --type=pr", file=sys.stderr)
        return 1

    token = get_token(args.token)
    if not token:
        print(
            "Error: GitHub token not found. Set GITHUB_TOKEN, pass --token, "
            "or create ~/.github_token with a valid token.",
            file=sys.stderr,
        )
        return 1

    if token == PLACEHOLDER_TOKEN:
        print(
            "Warning: the placeholder token is being used. Set a real token via GITHUB_TOKEN.",
            file=sys.stderr,
        )
        if not args.dry_run:
            print("Refusing to send a request with the placeholder token.", file=sys.stderr)
            return 1

    json_sections: List[str] = []
    for p in args.json_paths:
        path = Path(p)
        if not path.exists():
            print(f"Warning: JSON file not found: {path}", file=sys.stderr)
            continue
        try:
            data = load_json_metrics(path)
            json_sections.append(format_json_section(path, data))
        except Exception as exc:
            print(f"Warning: failed to parse {path}: {exc}", file=sys.stderr)

    csv_sections: List[str] = []
    for p in args.csv_paths:
        path = Path(p)
        if not path.exists():
            print(f"Warning: CSV file not found: {path}", file=sys.stderr)
            continue
        try:
            rows = load_csv_table(path)
            csv_sections.append(format_csv_section(path, rows))
        except Exception as exc:
            print(f"Warning: failed to parse {path}: {exc}", file=sys.stderr)

    body = build_body(args, json_sections, csv_sections)

    if args.type == "issue":
        url = create_issue(
            args.repo,
            args.title,
            body,
            args.labels,
            token,
            dry_run=args.dry_run,
        )
    else:
        url = create_pull_request(
            args.repo,
            args.title,
            body,
            args.head,
            args.base,
            token,
            draft=args.draft,
            labels=args.labels,
            dry_run=args.dry_run,
        )

    if url:
        print(f"Created {args.type}: {url}")
    else:
        print(f"Dry-run completed for {args.type}.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
