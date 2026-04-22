"""
runner.py
─────────
CLI entry point for the Voice Agent Eval Pipeline.

Usage:
    python runner.py --project dental_booking_bot
    python runner.py --project dental_booking_bot --level 1
    python runner.py --project dental_booking_bot --prompt v2_strict.xml
    python runner.py --project dental_booking_bot --prompt v2_strict.xml --level 2
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich import box

from engine.config_loader import (
    load_config, load_prompt, list_prompts,
    ConfigValidationError, PromptNotFoundError,
)
from engine.level1_engine import Level1Engine, RuleResult
from engine.level2_engine import Level2Engine, RubricResult
from engine.level3_engine import Level3Engine, Level3Result

console = Console()
BASE = Path(__file__).parent


# ── Args ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Voice Agent Eval Pipeline")
    parser.add_argument("--project", required=True,
                        help="Project folder name under projects/")
    parser.add_argument("--level", type=int, choices=[1, 2, 3], default=None,
                        help="Run only this level (default: all levels)")
    parser.add_argument("--prompt", default="default.xml",
                        help="Prompt filename from prompts/ directory (default: default.xml)")
    return parser.parse_args()


# ── Level 1 ────────────────────────────────────────────────────────────────────

def run_level1(project_path: Path, config: dict) -> tuple[int, int]:
    cases_file = project_path / "test_cases" / "level1_cases.json"
    if not cases_file.exists():
        console.print(f"[yellow]⚠  No level1_cases.json — skipping L1[/]")
        return 0, 0

    cases = json.loads(cases_file.read_text("utf-8"))
    if not cases:
        console.print("[yellow]⚠  No L1 cases defined — skipping[/]")
        return 0, 0

    engine = Level1Engine(config)
    passed_total = failed_total = 0

    console.rule("[bold cyan]LEVEL 1 — Syntactic & Heuristic Checks[/]")

    for case in cases:
        case_id = case["case_id"]
        response = case["response"]
        expected_failures: list[str] = case.get("expected_failures", [])
        results: list[RuleResult] = engine.run(response)
        actual_failures = {r.rule_id for r in results if not r.passed}
        expected_set = set(expected_failures)

        table = Table(
            title=f"[bold]{case_id}[/] — {case.get('description', '')}",
            box=box.SIMPLE_HEAD, show_header=True,
        )
        table.add_column("Rule", style="cyan", no_wrap=True)
        table.add_column("Status", justify="center", width=8)
        table.add_column("Detail")

        case_ok = True
        for r in results:
            if r.passed:
                status, detail = "[green]PASS[/]", ""
            else:
                status = "[red]FAIL[/]"
                detail = f"{r.detail}  {r.excerpt}"
                case_ok = False
            table.add_row(r.rule_id, status, detail)

        console.print(table)

        unexpected_passes   = expected_set - actual_failures
        unexpected_failures = actual_failures - expected_set
        if unexpected_passes:
            console.print(f"  [yellow]⚠  Expected failures did not trigger: {unexpected_passes}[/]")
        if unexpected_failures:
            console.print(f"  [yellow]⚠  Unexpected failures: {unexpected_failures}[/]")

        if actual_failures == expected_set:
            passed_total += 1
        else:
            failed_total += 1

    return passed_total, failed_total


# ── Level 2 ────────────────────────────────────────────────────────────────────

def run_level2(project_path: Path, config: dict) -> tuple[int, int]:
    cases_file = project_path / "test_cases" / "level2_cases.json"
    if not cases_file.exists():
        console.print("[yellow]⚠  No level2_cases.json — skipping L2[/]")
        return 0, 0

    cases = json.loads(cases_file.read_text("utf-8"))
    if not cases:
        console.print("[yellow]⚠  No L2 cases defined — skipping[/]")
        return 0, 0

    engine = Level2Engine(config)
    console.rule("[bold cyan]LEVEL 2 — Semantic / LLM-as-a-Judge[/]")

    if not engine.is_available():
        console.print(f"[red]✗  Ollama not reachable — skipping L2[/]")
        return 0, 0

    passed_total = failed_total = 0

    for case in cases:
        case_id = case["case_id"]
        results: list[RubricResult] = engine.run(
            agent_response=case["response"],
            user_message=case.get("user_message", ""),
        )
        actual_failures = {r.rubric_id for r in results if not r.passed}
        expected_set    = set(case.get("expected_failures", []))

        table = Table(
            title=f"[bold]{case_id}[/] — {case.get('description', '')}",
            box=box.SIMPLE_HEAD, show_header=True,
        )
        table.add_column("Rubric", style="cyan", no_wrap=True)
        table.add_column("Name", style="dim")
        table.add_column("Status", justify="center", width=8)
        table.add_column("Reason")

        for r in results:
            status = "[green]PASS[/]" if r.passed else "[red]FAIL[/]"
            table.add_row(r.rubric_id, r.name, status, r.reason)

        console.print(table)

        if actual_failures == expected_set:
            passed_total += 1
        else:
            failed_total += 1

    return passed_total, failed_total


# ── Level 3 ────────────────────────────────────────────────────────────────────

def run_level3(project_path: Path, config: dict) -> tuple[int, int]:
    cases_file = project_path / "test_cases" / "level3_cases.json"
    if not cases_file.exists():
        console.print("[yellow]⚠  No level3_cases.json — skipping L3[/]")
        return 0, 0

    cases = json.loads(cases_file.read_text("utf-8"))
    if not cases:
        console.print("[yellow]⚠  No L3 cases defined — skipping[/]")
        return 0, 0

    engine = Level3Engine(config)
    console.rule("[bold cyan]LEVEL 3 — Multi-turn Simulation[/]")

    if not engine.is_available():
        console.print("[red]✗  Ollama not reachable — skipping L3[/]")
        return 0, 0

    passed_total = failed_total = 0

    for case in cases:
        result: Level3Result = engine.run(case)
        console.print(f"\n[bold]{result.case_id}[/]")

        if result.error:
            console.print(f"  [red]ERROR: {result.error}[/]")
            failed_total += 1
            continue

        for t in result.turns:
            console.print(f"  [dim]Turn {t.turn}[/]")
            console.print(f"    [blue]USER:[/]  {t.user_message}")
            console.print(f"    [cyan]AGENT:[/] {t.agent_response}")
            for a in t.assertions:
                icon = "[green]✓[/]" if a["passed"] else "[red]✗[/]"
                console.print(f"    {icon} {a['assertion']}: {a['detail']}")

        goal_icon = "[green]✓ GOAL ACHIEVED[/]" if result.goal_achieved else "[red]✗ GOAL NOT MET[/]"
        console.print(f"\n  {goal_icon} — {result.goal_reason}")

        if result.passed:
            passed_total += 1
        else:
            failed_total += 1

    return passed_total, failed_total


# ── Reporting & Ledger ─────────────────────────────────────────────────────────

def save_report(
    project_path: Path,
    prompt_stem: str,
    summary: dict[str, tuple[int, int]],
    config: dict,
) -> Path:
    """Save JSON run report to reports/<prompt_stem>/ and return the file path."""
    run_dir = project_path / "reports" / prompt_stem
    run_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    report_file = run_dir / f"{ts}_run.json"

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "project": config["meta"]["name"],
        "prompt_file": config["meta"].get("_prompt_file", "default.xml"),
        "prompt_stem": prompt_stem,
        "ollama_model": config["meta"]["ollama_model"],
        "results": {
            lvl: {"passed": p, "failed": f, "total": p + f}
            for lvl, (p, f) in summary.items()
        },
    }

    report_file.write_text(json.dumps(report, indent=2), "utf-8")
    return report_file


def append_ledger(project_path: Path, summary: dict[str, tuple[int, int]], config: dict) -> None:
    """Append this run's summary to the project-level history_ledger.json."""
    ledger_file = project_path / "reports" / "history_ledger.json"
    ledger_file.parent.mkdir(parents=True, exist_ok=True)

    ledger: list[dict] = []
    if ledger_file.exists():
        try:
            ledger = json.loads(ledger_file.read_text("utf-8"))
        except json.JSONDecodeError:
            ledger = []

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt_file": config["meta"].get("_prompt_file", "default.xml"),
        "prompt_stem": config["meta"].get("_prompt_stem", "default"),
        "ollama_model": config["meta"]["ollama_model"],
    }
    for lvl, (p, f) in summary.items():
        entry[lvl] = {"passed": p, "failed": f, "total": p + f}

    ledger.append(entry)
    ledger_file.write_text(json.dumps(ledger, indent=2), "utf-8")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    project_path = BASE / "projects" / args.project

    if not project_path.exists():
        console.print(f"[red]Project not found: {project_path}[/]")
        sys.exit(1)

    try:
        config = load_config(project_path)
    except (FileNotFoundError, ConfigValidationError) as exc:
        console.print(f"[red bold]Config error:[/] {exc}")
        sys.exit(1)

    try:
        config = load_prompt(config, project_path, args.prompt)
    except PromptNotFoundError as exc:
        console.print(f"[red bold]Prompt error:[/] {exc}")
        available = list_prompts(project_path)
        if available:
            console.print(f"[dim]Available prompts: {', '.join(available)}[/]")
        sys.exit(1)

    prompt_stem = config["meta"]["_prompt_stem"]

    console.print(f"\n[bold]Project:[/] {config['meta']['name']}")
    console.print(f"[bold]Model:[/]   {config['meta']['ollama_model']}")
    console.print(f"[bold]Prompt:[/]  {args.prompt}  [dim]({prompt_stem})[/]\n")

    summary: dict[str, tuple[int, int]] = {}
    run_all = args.level is None

    if run_all or args.level == 1:
        summary["L1"] = run_level1(project_path, config)

    if run_all or args.level == 2:
        summary["L2"] = run_level2(project_path, config)

    if run_all or args.level == 3:
        summary["L3"] = run_level3(project_path, config)

    # Summary table
    console.rule("[bold]Summary[/]")
    t = Table(box=box.ROUNDED)
    t.add_column("Level", style="bold")
    t.add_column("Passed", justify="right", style="green")
    t.add_column("Failed", justify="right", style="red")
    for level, (p, f) in summary.items():
        t.add_row(level, str(p), str(f))
    console.print(t)

    # Save report + ledger
    report_path = save_report(project_path, prompt_stem, summary, config)
    append_ledger(project_path, summary, config)
    console.print(f"\n[dim]Report saved → {report_path.relative_to(BASE)}[/]")
    console.print(f"[dim]Ledger updated → projects/{args.project}/reports/history_ledger.json[/]")

    total_failed = sum(f for _, f in summary.values())
    sys.exit(0 if total_failed == 0 else 1)


if __name__ == "__main__":
    main()
