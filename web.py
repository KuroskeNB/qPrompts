"""
web.py  —  Voice Agent Eval Pipeline  —  Localhost UI
python web.py  →  http://localhost:7860
"""

from __future__ import annotations

import json
import logging
import queue as _queue
import shutil
import time
import webbrowser
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Generator

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from engine.config_loader import (
    load_config, load_prompt, list_prompts,
    ConfigValidationError, PromptNotFoundError,
)
from engine.level1_engine import Level1Engine
from engine.level2_engine import Level2Engine
from engine.level3_engine import Level3Engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("eval")

BASE = Path(__file__).parent
SETTINGS_FILE = BASE / "settings.json"
LAST_RUN_FILE  = BASE / "last_run.json"
app = FastAPI()

_cancel_event = Event()  # set() to abort the running eval stream

# ── Live eval state (survives page refresh) ────────────────────────────────────
_eval_lock     = Lock()
_eval_status   = "idle"   # "idle" | "running" | "done"
_eval_events:  list[dict]           = []
_eval_listeners: list[_queue.Queue] = []
_EVAL_DONE = object()  # sentinel pushed to listeners when eval finishes


def _broadcast(event: dict) -> None:
    """Append event to history and push to every live SSE listener."""
    with _eval_lock:
        _eval_events.append(event)
        for q in _eval_listeners:
            q.put(event)


def _finish_eval() -> None:
    """Mark eval done and signal all listeners."""
    global _eval_status
    with _eval_lock:
        _eval_status = "done"
        for q in _eval_listeners:
            q.put(_EVAL_DONE)


def _load_settings() -> dict:
    if SETTINGS_FILE.exists():
        try:
            return json.loads(SETTINGS_FILE.read_text("utf-8"))
        except Exception:
            pass
    return {}

def _save_settings(data: dict) -> None:
    SETTINGS_FILE.write_text(json.dumps(data, indent=2), "utf-8")

# ── Default config template ────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "meta": {
        "name": "",
        "description": "",
    },
    "level1": {"rules": []},
    "level2": {
        "judge_model": "gemini-3.1-flash-lite-preview",
        "temperature": 0.0,
        "verdict_format": '{"pass": true/false, "reason": "one sentence"}',
        "rubrics": [],
    },
    "level3": {
        "default_turns": 4,
        "goal_check_model": "gemini-3.1-flash-lite-preview",
        "turn_assertions": [
            {"type": "ends_with_question",
             "description": "Agent must close each turn with exactly one clarifying question",
             "params": {"max_questions": 1}}
        ],
        "goal_verification_prompt": (
            'Given the conversation transcript, did the agent successfully achieve '
            'the stated goal: "{goal_state}"? Reply ONLY with '
            '{"achieved": bool, "reason": "one sentence"}.'
        ),
    },
}

EMPTY_CASES: dict[str, list] = {"1": [], "2": [], "3": []}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"

def _list_projects() -> list[str]:
    d = BASE / "projects"
    return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.exists() else []

def _cases_file(project: str, level: str) -> Path:
    return BASE / "projects" / project / "test_cases" / f"level{level}_cases.json"

def _load_cases(project: str, level: str) -> list:
    f = _cases_file(project, level)
    return json.loads(f.read_text("utf-8")) if f.exists() else []

def _save_cases(project: str, level: str, cases: list) -> None:
    f = _cases_file(project, level)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(cases, indent=2, ensure_ascii=False), "utf-8")

def _load_raw_config(project: str) -> dict:
    f = BASE / "projects" / project / "project_config.yaml"
    if not f.exists():
        raise HTTPException(404, "project_config.yaml not found")
    return yaml.safe_load(f.read_text("utf-8"))

def _save_raw_config(project: str, cfg: dict) -> None:
    f = BASE / "projects" / project / "project_config.yaml"
    f.write_text(yaml.dump(cfg, allow_unicode=True, sort_keys=False), "utf-8")


# ── API routes ─────────────────────────────────────────────────────────────────

@app.get("/projects")
def get_projects():
    return {"projects": _list_projects()}

@app.post("/projects")
async def create_project(req: Request):
    body = await req.json()
    name = body.get("name", "").strip().replace(" ", "_")
    if not name:
        raise HTTPException(400, "Project name required")
    path = BASE / "projects" / name
    if path.exists():
        raise HTTPException(409, "Project already exists")
    path.mkdir(parents=True)
    (path / "test_cases").mkdir()
    cfg = dict(DEFAULT_CONFIG)
    cfg["meta"] = dict(cfg["meta"])
    cfg["meta"]["name"] = name
    _save_raw_config(name, cfg)
    for lvl in ["1", "2", "3"]:
        _save_cases(name, lvl, [])
    return {"ok": True, "name": name}

@app.delete("/projects/{name}")
def delete_project(name: str):
    path = BASE / "projects" / name
    if not path.exists():
        raise HTTPException(404, "Project not found")
    shutil.rmtree(path)
    return {"ok": True}

@app.get("/projects/{name}/config")
def get_config(name: str):
    return _load_raw_config(name)

@app.put("/projects/{name}/config")
async def save_config(name: str, req: Request):
    body = await req.json()
    path = BASE / "projects" / name
    if not path.exists():
        raise HTTPException(404, "Project not found")
    # Support raw YAML import from clipboard
    if "_raw_yaml" in body:
        try:
            parsed = yaml.safe_load(body["_raw_yaml"])
        except yaml.YAMLError as e:
            raise HTTPException(422, f"YAML parse error: {e}")
        if not isinstance(parsed, dict):
            raise HTTPException(422, "YAML must be a mapping")
        body = parsed
    _save_raw_config(name, body)
    try:
        load_config(path)
    except ConfigValidationError as e:
        raise HTTPException(422, str(e))
    return {"ok": True}

@app.get("/projects/{name}/cases/{level}")
def get_cases(name: str, level: str):
    if level not in ("1", "2", "3"):
        raise HTTPException(400, "level must be 1, 2, or 3")
    return {"cases": _load_cases(name, level)}

@app.put("/projects/{name}/cases/{level}")
async def save_cases(name: str, level: str, req: Request):
    if level not in ("1", "2", "3"):
        raise HTTPException(400, "level must be 1, 2, or 3")
    body = await req.json()
    cases = body.get("cases", [])
    _save_cases(name, level, cases)
    return {"ok": True}


@app.get("/projects/{name}/config/yaml")
def get_config_yaml(name: str):
    cfg = _load_raw_config(name)
    return JSONResponse({"yaml": yaml.dump(cfg, allow_unicode=True, sort_keys=False)})

# ── Prompts ────────────────────────────────────────────────────────────────────

@app.get("/projects/{name}/prompts")
def get_prompts(name: str):
    path = BASE / "projects" / name
    return {"prompts": list_prompts(path)}

@app.get("/projects/{name}/prompts/{filename}")
def get_prompt(name: str, filename: str):
    f = BASE / "projects" / name / "prompts" / filename
    if not f.exists():
        raise HTTPException(404, "Prompt file not found")
    return {"filename": filename, "content": f.read_text("utf-8")}

@app.put("/projects/{name}/prompts/{filename}")
async def save_prompt(name: str, filename: str, req: Request):
    body = await req.json()
    d = BASE / "projects" / name / "prompts"
    d.mkdir(parents=True, exist_ok=True)
    (d / filename).write_text(body.get("content", ""), "utf-8")
    return {"ok": True}

@app.post("/projects/{name}/prompts")
async def create_prompt(name: str, req: Request):
    body = await req.json()
    filename = body.get("filename", "").strip()
    if not filename:
        raise HTTPException(400, "filename required")
    if not filename.endswith((".xml", ".txt", ".md")):
        filename += ".xml"
    d = BASE / "projects" / name / "prompts"
    d.mkdir(parents=True, exist_ok=True)
    f = d / filename
    if f.exists():
        raise HTTPException(409, "Prompt file already exists")
    f.write_text(body.get("content", ""), "utf-8")
    return {"ok": True, "filename": filename}

@app.delete("/projects/{name}/prompts/{filename}")
def delete_prompt(name: str, filename: str):
    f = BASE / "projects" / name / "prompts" / filename
    if not f.exists():
        raise HTTPException(404, "Prompt file not found")
    if filename == "default.xml":
        raise HTTPException(400, "Cannot delete default.xml")
    f.unlink()
    return {"ok": True}

# ── History ledger ─────────────────────────────────────────────────────────────

@app.get("/projects/{name}/ledger")
def get_ledger(name: str):
    f = BASE / "projects" / name / "reports" / "history_ledger.json"
    if not f.exists():
        return {"entries": []}
    try:
        return {"entries": json.loads(f.read_text("utf-8"))}
    except Exception:
        return {"entries": []}


@app.delete("/projects/{name}/ledger/{index}")
def delete_ledger_entry(name: str, index: int):
    f = BASE / "projects" / name / "reports" / "history_ledger.json"
    if not f.exists():
        raise HTTPException(status_code=404, detail="Ledger not found")
    entries = json.loads(f.read_text("utf-8"))
    if index < 0 or index >= len(entries):
        raise HTTPException(status_code=404, detail="Entry index out of range")
    entries.pop(index)
    f.write_text(json.dumps(entries, indent=2), "utf-8")
    return {"ok": True}

@app.get("/settings")
def get_settings():
    s = _load_settings()
    return {
        "gemini_api_key":      s.get("gemini_api_key", ""),
        "langsmith_api_key":   s.get("langsmith_api_key", ""),
        "langsmith_project":   s.get("langsmith_project", "voice-agent-evals"),
        "langsmith_endpoint":   s.get("langsmith_endpoint", "https://api.smith.langchain.com"),
        "langsmith_tenant_id":  s.get("langsmith_tenant_id", ""),
        "langsmith_session_id": s.get("langsmith_session_id", ""),
    }

@app.post("/settings")
async def save_settings_endpoint(req: Request):
    body = await req.json()
    s = _load_settings()
    for field in ("gemini_api_key", "langsmith_api_key", "langsmith_project", "langsmith_endpoint", "langsmith_tenant_id", "langsmith_session_id"):
        if field in body:
            s[field] = body[field]
    _save_settings(s)
    return {"ok": True}


@app.post("/test-langsmith")
async def test_langsmith(req: Request):
    """
    1. Auto-resolve tenant_id and session_id from the API key + project name.
    2. Post a test trace to confirm write access.
    3. Return resolved IDs so the frontend can save them without manual copy-paste.
    """
    import urllib.request, urllib.error, uuid as _uuid, json as _json
    from datetime import datetime, timezone

    body      = await req.json()
    api_key   = body.get("api_key", "").strip()
    project   = body.get("project", "voice-agent-evals").strip() or "voice-agent-evals"
    endpoint  = body.get("endpoint", "https://api.smith.langchain.com").rstrip("/")

    if not api_key:
        return JSONResponse({"ok": False, "error": "No API key provided"})

    def _get(url: str, extra_hdrs: dict = {}) -> dict | list:
        hdrs = {"x-api-key": api_key, **extra_hdrs}
        r = urllib.request.Request(url, headers=hdrs)
        with urllib.request.urlopen(r, timeout=10) as resp:
            return _json.loads(resp.read())

    # ── Step 1: resolve tenant_id from sessions list (no tenant header needed) ──
    tenant_id  = ""
    session_id = ""
    try:
        sessions = _get(f"{endpoint}/sessions")
        if isinstance(sessions, list) and sessions:
            # All sessions share the same tenant; grab it from the first one
            tenant_id = sessions[0].get("tenant_id", "")
            # Find session matching the requested project name (case-insensitive)
            for s in sessions:
                if s.get("name", "").lower() == project.lower():
                    session_id = s.get("id", "")
                    break
            # If no match, create/find by posting — LangSmith auto-creates on first run
    except Exception as exc:
        return JSONResponse({"ok": False, "error": f"Could not list sessions: {exc}"})

    # ── Step 2: post a test trace using session_id if found ───────────────────
    now    = datetime.now(timezone.utc)
    run_id = str(_uuid.uuid4())
    run = {
        "id": run_id,
        "name": "connection-test",
        "run_type": "llm",
        "inputs":  {"messages": [{"role": "user", "content": "ping"}]},
        "outputs": {"text": "pong"},
        "start_time": now.isoformat(),
        "end_time":   now.isoformat(),
        "trace_id": run_id,
        "dotted_order": now.strftime("%Y%m%dT%H%M%S%f") + "Z" + run_id,
        "tags": ["connection-test"],
        **({"session_id": session_id} if session_id else {"project_name": project}),
    }
    hdrs = {"Content-Type": "application/json", "x-api-key": api_key}
    if tenant_id:
        hdrs["X-Tenant-ID"] = tenant_id

    try:
        request = urllib.request.Request(
            f"{endpoint}/runs", data=_json.dumps(run).encode(), headers=hdrs, method="POST")
        with urllib.request.urlopen(request, timeout=10) as resp:
            status = resp.status

        # ── Step 3: auto-save resolved IDs back to settings ───────────────────
        if tenant_id or session_id:
            s = _load_settings()
            if tenant_id:  s["langsmith_tenant_id"]  = tenant_id
            if session_id: s["langsmith_session_id"] = session_id
            _save_settings(s)

        return {
            "ok": True, "status": status,
            "tenant_id": tenant_id, "session_id": session_id,
            "project_matched": bool(session_id),
        }
    except urllib.error.HTTPError as exc:
        err = exc.read().decode("utf-8", errors="replace")[:300]
        return JSONResponse({"ok": False, "error": f"HTTP {exc.code}: {err}"})
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)})


@app.post("/cancel")
def cancel_eval():
    _cancel_event.set()
    log.info("Eval cancelled by user")
    return {"ok": True}


@app.get("/debug-langsmith")
def debug_langsmith():
    """Synchronously fire one trace and return exactly what LangSmith responds."""
    import urllib.request as _ur, urllib.error as _ue
    import uuid as _uuid
    from datetime import datetime, timezone as _tz

    s = _load_settings()
    key       = s.get("langsmith_api_key", "").strip()
    endpoint  = s.get("langsmith_endpoint", "https://api.smith.langchain.com").rstrip("/")
    tenant    = s.get("langsmith_tenant_id", "").strip()
    project   = s.get("langsmith_project", "voice-agent-evals").strip() or "voice-agent-evals"
    session   = s.get("langsmith_session_id", "").strip()

    if not key:
        return {"ok": False, "error": "No langsmith_api_key in settings"}

    now    = datetime.now(_tz.utc)
    run_id = str(_uuid.uuid4())
    run = {
        "id": run_id,
        "name": "debug-trace",
        "run_type": "llm",
        "inputs":  {"messages": [{"role": "user", "content": "debug ping"}]},
        "outputs": {"text": "debug pong"},
        "start_time": now.isoformat(),
        "end_time":   now.isoformat(),
        "trace_id":   run_id,
        "dotted_order": now.strftime("%Y%m%dT%H%M%S%f") + "Z" + run_id,
        **({"session_id": session} if session else {"project_name": project}),
        "tags": ["debug"],
    }
    hdrs = {"Content-Type": "application/json", "x-api-key": key}
    if tenant:
        hdrs["X-Tenant-ID"] = tenant

    try:
        req = _ur.Request(f"{endpoint}/runs",
                          data=json.dumps(run).encode(), headers=hdrs, method="POST")
        with _ur.urlopen(req, timeout=10) as resp:
            body = resp.read().decode()
            return {"ok": True, "status": resp.status, "body": body,
                    "endpoint": endpoint, "tenant": tenant, "project": project,
                    "session_id": session, "run_id": run_id}
    except _ue.HTTPError as exc:
        err = exc.read().decode()
        return {"ok": False, "status": exc.code, "error": err,
                "endpoint": endpoint, "tenant": tenant}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@app.get("/last-run")
def get_last_run():
    if LAST_RUN_FILE.exists():
        try:
            events = json.loads(LAST_RUN_FILE.read_text("utf-8"))
            return {"events": events}
        except Exception:
            pass
    return {"events": []}


_FIXED_MODEL = "gemini-3.1-flash-lite-preview"


def _eval_worker(project: str, level: str, prompt: str, config: dict, project_path: Path) -> None:
    """
    Runs the full evaluation in a background thread.
    Emits events via _broadcast() so SSE clients can attach/re-attach at any time.
    The eval continues running even if all SSE clients disconnect (page refresh).
    """
    global _eval_status

    summary: dict[str, dict] = {}

    def emit(event: dict) -> None:
        _broadcast(event)

    def emit_log(msg: str) -> None:
        log.info(msg)
        _broadcast({"type": "log", "msg": msg})

    try:
        emit({"type": "meta", "name": config["meta"]["name"],
              "model": _FIXED_MODEL, "provider": "gemini", "prompt": prompt})
        run_all = level == "all"

        # Level 1
        if run_all or level == "1":
            emit({"type": "level_start", "level": 1, "title": "LEVEL 1 — Syntactic & Heuristic Checks"})
            cases_file = project_path / "test_cases" / "level1_cases.json"
            if not cases_file.exists():
                emit({"type": "level_skip", "level": 1, "msg": "No level1_cases.json found"})
            else:
                cases = json.loads(cases_file.read_text("utf-8"))
                if not cases:
                    emit({"type": "level_skip", "level": 1, "msg": "No test cases defined"})
                else:
                    engine = Level1Engine(config)
                    l1p = l1f = 0
                    for case in cases:
                        if _cancel_event.is_set():
                            emit_log(f"L1 cancelled after {l1p + l1f} cases")
                            emit({"type": "cancelled"}); return
                        emit_log(f"L1 running: {case['case_id']}")
                        results = engine.run(case["response"])
                        actual_failures = {r.rule_id for r in results if not r.passed}
                        expected_set = set(case.get("expected_failures", []))
                        case_ok = actual_failures == expected_set
                        if case_ok: l1p += 1
                        else: l1f += 1
                        emit_log(f"L1 {case['case_id']}: {'PASS' if case_ok else 'FAIL'}")
                        emit({"type": "l1_case", "case_id": case["case_id"],
                            "description": case.get("description", ""),
                            "response": case["response"],
                            "rules": [{"rule_id": r.rule_id, "passed": r.passed,
                                       "detail": r.detail, "excerpt": r.excerpt} for r in results],
                            "passed": case_ok,
                            "unexpected_passes": list(expected_set - actual_failures),
                            "unexpected_failures": list(actual_failures - expected_set)})
                    summary["L1"] = {"passed": l1p, "failed": l1f, "total": l1p+l1f}
                    emit({"type": "level_summary", "level": 1, "passed": l1p, "failed": l1f})

        # Level 2
        if run_all or level == "2":
            emit({"type": "level_start", "level": 2, "title": "LEVEL 2 — Semantic / LLM-as-a-Judge"})
            cases_file = project_path / "test_cases" / "level2_cases.json"
            if not cases_file.exists():
                emit({"type": "level_skip", "level": 2, "msg": "No level2_cases.json found"})
            else:
                engine2 = Level2Engine(config)
                if not engine2.is_available():
                    emit({"type": "level_skip", "level": 2, "msg": "Gemini API key missing or invalid"})
                else:
                    cases = json.loads(cases_file.read_text("utf-8"))
                    if not cases:
                        emit({"type": "level_skip", "level": 2, "msg": "No test cases defined"})
                    else:
                        l2p = l2f = 0
                        for case in cases:
                            if _cancel_event.is_set():
                                emit_log(f"L2 cancelled after {l2p + l2f} cases")
                                emit({"type": "cancelled"}); return
                            emit_log(f"L2 running: {case['case_id']}")
                            emit({"type": "l2_thinking", "case_id": case["case_id"]})
                            results = engine2.run(agent_response=case["response"],
                                                  user_message=case.get("user_message", ""),
                                                  cancel_event=_cancel_event)
                            if _cancel_event.is_set():
                                emit_log(f"L2 cancelled mid-case: {case['case_id']}")
                                emit({"type": "cancelled"}); return
                            actual_failures = {r.rubric_id for r in results if not r.passed}
                            expected_set = set(case.get("expected_failures", []))
                            case_ok = actual_failures == expected_set
                            if case_ok: l2p += 1
                            else: l2f += 1
                            emit_log(f"L2 {case['case_id']}: {'PASS' if case_ok else 'FAIL'}")
                            emit({"type": "l2_case", "case_id": case["case_id"],
                                "description": case.get("description", ""),
                                "user_message": case.get("user_message", ""),
                                "response": case["response"],
                                "rubrics": [{"rubric_id": r.rubric_id, "name": r.name,
                                             "passed": r.passed, "reason": r.reason} for r in results],
                                "passed": case_ok,
                                "unexpected_passes": list(expected_set - actual_failures),
                                "unexpected_failures": list(actual_failures - expected_set)})
                        summary["L2"] = {"passed": l2p, "failed": l2f, "total": l2p+l2f}
                        emit({"type": "level_summary", "level": 2, "passed": l2p, "failed": l2f})

        # Level 3
        if run_all or level == "3":
            emit({"type": "level_start", "level": 3, "title": "LEVEL 3 — Multi-turn Simulation"})
            cases_file = project_path / "test_cases" / "level3_cases.json"
            if not cases_file.exists():
                emit({"type": "level_skip", "level": 3, "msg": "No level3_cases.json found"})
            else:
                engine3 = Level3Engine(config)
                if not engine3.is_available():
                    emit({"type": "level_skip", "level": 3, "msg": "Gemini API key missing or invalid"})
                else:
                    cases = json.loads(cases_file.read_text("utf-8"))
                    if not cases:
                        emit({"type": "level_skip", "level": 3, "msg": "No test cases defined"})
                    else:
                        l3p = l3f = 0
                        for case in cases:
                            if _cancel_event.is_set():
                                emit_log(f"L3 cancelled after {l3p + l3f} cases")
                                emit({"type": "cancelled"}); return
                            emit_log(f"L3 running: {case['case_id']}")
                            emit({"type": "l3_thinking", "case_id": case["case_id"],
                                  "turns": case.get("turns", 4)})
                            result = engine3.run(case, cancel_event=_cancel_event)
                            if result.passed: l3p += 1
                            else: l3f += 1
                            emit_log(f"L3 {case['case_id']}: {'PASS' if result.passed else 'FAIL'}")
                            emit({"type": "l3_case", "case_id": result.case_id,
                                "description": case.get("description", ""),
                                "passed": result.passed,
                                "goal_achieved": result.goal_achieved,
                                "goal_reason": result.goal_reason,
                                "error": result.error,
                                "turns": [{"turn": t.turn, "user": t.user_message,
                                           "agent": t.agent_response, "assertions": t.assertions,
                                           "ok": t.all_assertions_passed,
                                           "latency_ms": t.latency_ms} for t in result.turns]})
                        summary["L3"] = {"passed": l3p, "failed": l3f, "total": l3p+l3f}
                        emit({"type": "level_summary", "level": 3, "passed": l3p, "failed": l3f})

        emit({"type": "done"})

    except Exception as exc:
        log.exception("Eval worker crashed")
        emit({"type": "error", "msg": f"Internal error: {exc}"})

    finally:
        # Save last-run snapshot
        with _eval_lock:
            events_snapshot = list(_eval_events)
        if events_snapshot:
            LAST_RUN_FILE.write_text(json.dumps(events_snapshot, indent=2), "utf-8")

        # Append to history ledger
        if summary:
            ledger_file = project_path / "reports" / "history_ledger.json"
            ledger_file.parent.mkdir(parents=True, exist_ok=True)
            ledger: list = []
            if ledger_file.exists():
                try: ledger = json.loads(ledger_file.read_text("utf-8"))
                except Exception: pass
            from datetime import datetime, timezone
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "prompt_file": prompt,
                "prompt_stem": Path(prompt).stem,
                "model": _FIXED_MODEL,
                "provider": "gemini",
            }
            entry.update(summary)
            ledger.append(entry)
            ledger_file.write_text(json.dumps(ledger, indent=2), "utf-8")

        _finish_eval()


@app.get("/run-status")
def get_run_status():
    return {"status": _eval_status, "events_count": len(_eval_events)}


@app.get("/run")
def run_eval(project: str = "", level: str = "all", prompt: str = "default.xml", api_key: str = ""):
    """
    SSE endpoint. Behaviour:
    - If eval is idle and project given  → start new eval thread, attach as listener.
    - If eval is running                 → attach as listener (replay past events + stream live).
    - If eval is done                    → replay completed events and close.
    Eval thread keeps running even when all SSE clients disconnect (page refresh).
    """
    global _eval_status, _eval_events

    listener_q: _queue.Queue = _queue.Queue()

    with _eval_lock:
        if _eval_status == "running":
            # Attach to existing run: snapshot past events then register listener.
            # Both inside the lock so _broadcast() can't slip an event between them.
            past = list(_eval_events)
            _eval_listeners.append(listener_q)
            live = True

        elif _eval_status == "done":
            # Replay the completed run then close — no live streaming needed.
            past = list(_eval_events)
            live = False

        else:  # idle — start a new run
            if not project:
                def _err():
                    yield _sse({"type": "error", "msg": "Project name required"})
                return StreamingResponse(_err(), media_type="text/event-stream",
                                         headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

            project_path = BASE / "projects" / project
            if not project_path.exists():
                def _err():
                    yield _sse({"type": "error", "msg": f"Project not found: {project}"})
                return StreamingResponse(_err(), media_type="text/event-stream",
                                         headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

            try:
                config = load_config(project_path)
                config = load_prompt(config, project_path, prompt)
            except (FileNotFoundError, ConfigValidationError, PromptNotFoundError) as exc:
                msg = str(exc)
                def _err():
                    yield _sse({"type": "error", "msg": msg})
                return StreamingResponse(_err(), media_type="text/event-stream",
                                         headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

            s = _load_settings()
            resolved_key = api_key or s.get("gemini_api_key", "")
            config["meta"]["provider"]           = "gemini"
            config["meta"]["gemini_api_key"]     = resolved_key
            config["meta"]["gemini_model"]       = _FIXED_MODEL
            config["meta"]["langsmith_api_key"]  = s.get("langsmith_api_key", "")
            config["meta"]["langsmith_project"]  = s.get("langsmith_project", "voice-agent-evals")
            config["meta"]["langsmith_endpoint"] = s.get("langsmith_endpoint", "https://api.smith.langchain.com")
            config["meta"]["langsmith_tenant_id"]  = s.get("langsmith_tenant_id", "")
            config["meta"]["langsmith_session_id"] = s.get("langsmith_session_id", "")

            _cancel_event.clear()
            _eval_status = "running"
            _eval_events = []
            _eval_listeners.append(listener_q)
            past = []
            live = True

            Thread(
                target=_eval_worker,
                args=(project, level, prompt, config, project_path),
                daemon=False,
                name="eval-worker",
            ).start()

    def stream() -> Generator[str, None, None]:
        try:
            # Replay events emitted before this client connected
            for event in past:
                yield _sse(event)

            if not live:
                return  # completed run — replay only, no queue

            # Stream live events until the worker signals done
            while True:
                try:
                    event = listener_q.get(timeout=30)
                except _queue.Empty:
                    yield ": keepalive\n\n"  # prevent proxy timeouts
                    continue
                if event is _EVAL_DONE:
                    break
                yield _sse(event)
        finally:
            with _eval_lock:
                if listener_q in _eval_listeners:
                    _eval_listeners.remove(listener_q)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── HTML ───────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Voice Agent Eval Pipeline</title>
<style>
:root{--bg:linear-gradient(135deg, #090a0f 0%, #13111c 100%);--sur:rgba(30,33,43,0.4);--sur2:rgba(43,47,60,0.5);--bor:rgba(255,255,255,0.08);--text:#f8fafc;--dim:#94a3b8;--green:#10b981;--red:#f43f5e;--yellow:#f59e0b;--blue:#3b82f6;--violet:#8b5cf6;--cyan:#06b6d4;--pink:#ec4899}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);background-attachment:fixed;color:var(--text);font-family:'Inter',system-ui,sans-serif;font-size:14px;height:100vh;display:flex;flex-direction:column;overflow:hidden}

/* ── Header ── */
header{background:rgba(15,17,23,0.5);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);padding:14px 24px;border-bottom:1px solid var(--bor);display:flex;align-items:center;gap:16px;flex-shrink:0;box-shadow:0 4px 30px rgba(0,0,0,0.1)}
header h1{font-size:16px;font-weight:700;background:-webkit-linear-gradient(0deg,#3b82f6,#8b5cf6);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.dot{width:8px;height:8px;border-radius:50%;background:var(--dim);flex-shrink:0;transition:background .3s}
.dot.up{background:var(--green);box-shadow:0 0 8px rgba(16,185,129,0.5)}
.dot.down{background:var(--red);box-shadow:0 0 8px rgba(244,63,94,0.5)}
.hdr-right{margin-left:auto;display:flex;align-items:center;gap:12px;font-size:12px;color:var(--dim)}

/* ── Layout ── */
.body{display:flex;flex:1;overflow:hidden}
.sidebar{width:220px;flex-shrink:0;background:rgba(20,22,31,0.3);backdrop-filter:blur(8px);border-right:1px solid var(--bor);display:flex;flex-direction:column;overflow:hidden}
.sidebar-top{padding:14px 12px;border-bottom:1px solid var(--bor);flex-shrink:0}
.proj-list{flex:1;overflow-y:auto;padding:8px}
.proj-item{padding:8px 10px;border-radius:8px;cursor:pointer;display:flex;align-items:center;gap:8px;margin-bottom:4px;transition:all .2s ease;font-size:13px;border:1px solid transparent}
.proj-item:hover{background:var(--sur2);border-color:rgba(255,255,255,0.05);transform:translateX(2px)}
.proj-item.active{background:rgba(59,130,246,0.15);border:1px solid rgba(59,130,246,0.3);box-shadow:inset 4px 0 0 var(--blue)}
.proj-item .pname{flex:1;font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.proj-item .del-btn{opacity:0;font-size:11px;color:var(--red);padding:2px 5px;border-radius:4px;transition:opacity .15s}
.proj-item:hover .del-btn{opacity:1}

/* ── Main panel ── */
.panel{flex:1;display:flex;flex-direction:column;overflow:hidden}
.tabs{display:flex;border-bottom:1px solid var(--bor);flex-shrink:0;padding:0 20px;background:rgba(15,17,23,0.3)}
.tab{padding:12px 16px;cursor:pointer;font-size:13px;font-weight:600;color:var(--dim);border-bottom:2px solid transparent;transition:all .2s ease}
.tab:hover{color:var(--text)}
.tab.active{color:#fff;border-bottom-color:var(--blue);text-shadow:0 0 12px rgba(59,130,246,0.5)}
.tab-content{flex:1;overflow-y:auto;padding:20px 24px;display:none;position:relative}
.tab-content.active{display:block;animation:fadeIn 0.3s}
@keyframes fadeIn{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}

/* ── Controls (run tab) ── */
.run-controls{display:flex;gap:12px;align-items:flex-end;margin-bottom:20px;flex-wrap:wrap}
.field{display:flex;flex-direction:column;gap:5px}
.field label{font-size:11px;text-transform:uppercase;letter-spacing:.7px;color:var(--dim)}
select,input[type=text],input[type=number],input[type=password],textarea{height:36px;border-radius:8px;border:1px solid rgba(255,255,255,0.12);background:rgba(15,17,23,0.5);color:var(--text);padding:0 12px;font-size:13px;font-family:inherit;transition:all .2s ease;box-shadow:inset 0 1px 3px rgba(0,0,0,0.2)}
select:focus,input:focus,textarea:focus{outline:none;border-color:var(--blue);box-shadow:0 0 0 2px rgba(59,130,246,0.3)}
textarea{height:auto;padding:8px 12px;resize:vertical;min-height:70px}
button{height:36px;border-radius:8px;border:1px solid var(--bor);background:var(--sur);color:var(--text);padding:0 16px;font-size:13px;font-weight:500;cursor:pointer;font-family:inherit;transition:all .2s cubic-bezier(0.4,0,0.2,1)}
button:hover{background:var(--sur2);transform:translateY(-1px);box-shadow:0 4px 12px rgba(0,0,0,0.15)}
button:active{transform:translateY(0)}
button.primary{background:linear-gradient(135deg,#3b82f6,#6366f1);border:none;color:#fff;font-weight:600;box-shadow:0 4px 15px rgba(59,130,246,0.35)}
button.primary:hover{opacity:1;transform:translateY(-1px);box-shadow:0 6px 20px rgba(59,130,246,0.5)}
button.primary:disabled{opacity:.45;cursor:not-allowed;transform:none;box-shadow:none}
button.danger{background:#450a0a;border-color:#7f1d1d;color:var(--red)}
button.danger:hover{background:#5c0f0f;box-shadow:0 4px 15px rgba(239,68,68,0.2)}
button.sm{height:28px;font-size:12px;padding:0 10px;border-radius:6px}

/* ── Manage tab sections ── */
.section{margin-bottom:28px}
.section-title{font-size:12px;text-transform:uppercase;letter-spacing:.8px;color:var(--text);margin-bottom:12px;display:flex;align-items:center;gap:10px;font-weight:600}
.section-title::after{content:'';flex:1;height:1px;background:var(--bor)}
.form-row{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px}
.form-row.triple{grid-template-columns:1fr 1fr 1fr}
.form-row.full{grid-template-columns:1fr}
.form-group{display:flex;flex-direction:column;gap:5px}
.form-group label{font-size:11px;text-transform:uppercase;letter-spacing:.6px;color:var(--dim)}

/* ── Rule cards ── */
.rule-card,.case-card,.res-card{background:var(--sur);backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);border:1px solid var(--bor);border-radius:10px;margin-bottom:12px;overflow:hidden;box-shadow:0 4px 15px rgba(0,0,0,0.1);transition:transform .2s ease,border-color .2s}
.rule-card:hover,.case-card:hover{transform:translateY(-2px);border-color:rgba(255,255,255,0.15)}
.rule-header{padding:10px 14px;display:flex;align-items:center;gap:10px;cursor:pointer}
.rule-header:hover{background:var(--sur2)}
.rule-tag{font-size:10px;padding:2px 8px;border-radius:10px;font-weight:600;background:#1e3a5f;color:var(--cyan);box-shadow:none}
.rule-id-label{font-family:monospace;font-size:12px;color:var(--dim)}
.rule-desc{flex:1;font-size:13px}
.rule-body{padding:0 14px 14px;display:none}
.rule-body.open{display:block;animation:fadeIn 0.2s}
.chev{font-size:10px;color:var(--dim);transition:transform .2s}
.chev.open{transform:rotate(90deg)}

/* ── Sub-tabs (manage cases) ── */
.sub-tabs{display:flex;gap:6px;margin-bottom:16px}
.sub-tab{padding:6px 14px;border-radius:20px;cursor:pointer;font-size:12px;font-weight:600;color:var(--dim);background:var(--sur);border:1px solid var(--bor);transition:all .2s ease;box-shadow:0 2px 5px rgba(0,0,0,0.1)}
.sub-tab.active{background:rgba(59,130,246,0.15);border-color:rgba(59,130,246,0.4);color:var(--blue)}
.sub-tab:hover:not(.active){background:var(--sur2)}

/* ── Case cards ── */
.case-card{background:var(--sur);border:1px solid var(--bor);border-radius:8px;margin-bottom:10px;overflow:hidden}
.case-hdr{padding:10px 14px;display:flex;align-items:center;gap:10px;cursor:pointer}
.case-hdr:hover{background:var(--sur2)}
.case-body-edit{padding:0 14px 14px;display:none}
.case-body-edit.open{display:block;animation:fadeIn 0.2s}

/* ── Run results ── */
.level-hdr{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:var(--cyan);margin-bottom:14px;margin-top:6px;display:flex;align-items:center;gap:10px;text-shadow:0 0 8px rgba(6,182,212,0.3)}
.level-hdr::after{content:'';flex:1;height:1px;background:linear-gradient(to right,var(--bor),transparent)}
.res-card{background:var(--sur);border:1px solid var(--bor);border-radius:8px;margin-bottom:12px;overflow:hidden}
.res-hdr{padding:10px 14px;display:flex;align-items:center;gap:10px;cursor:pointer}
.res-hdr:hover{background:var(--sur2)}
.res-body{padding:0 14px 14px;display:none}
.res-body.open{display:block}
.pill{font-size:11px;font-weight:700;padding:3px 10px;border-radius:10px}
.pill.pass{background:#14532d;color:var(--green)}.pill.fail{background:#450a0a;color:var(--red)}
.pill.running{background:#1e3a5f;color:var(--cyan);animation:pulse 1.2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
.res-table{width:100%;border-collapse:collapse;font-size:13px;margin-top:10px}
.res-table th{text-align:left;padding:6px 8px;color:var(--dim);font-weight:500;border-bottom:1px solid var(--bor);font-size:11px;text-transform:uppercase}
.res-table td{padding:7px 8px;border-bottom:1px solid #1e2130;vertical-align:top}
.res-table tr:last-child td{border:none}
.pass-ic{color:var(--green);font-weight:700}.fail-ic{color:var(--red);font-weight:700}
.mono{font-family:monospace;font-size:12px}
.dim{color:var(--dim);font-size:12px}
.excerpt{font-family:monospace;font-size:11px;color:var(--yellow)}
.resp-box{background:#0d1117;border:1px solid var(--bor);border-radius:5px;padding:9px 12px;margin:10px 0 4px;font-family:monospace;font-size:12px;color:#a0aec0;white-space:pre-wrap;line-height:1.6}
.resp-lbl{font-size:10px;text-transform:uppercase;letter-spacing:.6px;color:var(--dim);margin-top:8px}
.warn-box{background:#1c1400;border:1px solid #92400e;border-radius:5px;padding:7px 11px;margin-top:8px;font-size:12px;color:var(--yellow)}
.skip-box{background:#1c1917;border:1px solid #44403c;border-radius:7px;padding:11px 14px;color:var(--dim);font-size:13px;margin-bottom:12px}
.sum-bar{background:var(--sur);border:1px solid var(--bor);border-radius:8px;padding:14px 18px;margin-top:18px;display:flex;gap:20px;align-items:center;flex-wrap:wrap}
.sum-bar h3{font-size:11px;text-transform:uppercase;letter-spacing:.8px;color:var(--dim);flex-basis:100%}
.stat .num{font-size:26px;font-weight:700;text-align:center}
.stat .num.pass{color:var(--green)}.stat .num.fail{color:var(--red)}
.stat .lbl{font-size:11px;color:var(--dim);text-align:center}
.turn-row{display:flex;gap:10px;margin-bottom:4px}
.spk{font-size:11px;font-weight:700;width:48px;flex-shrink:0;padding-top:3px}
.spk.u{color:var(--blue)}.spk.a{color:var(--cyan)}
.bub{background:#0d1117;border-radius:5px;padding:7px 11px;font-size:13px;line-height:1.5;flex:1}
.assert-row{display:flex;align-items:center;gap:5px;font-size:11px;color:var(--dim);margin:3px 0 3px 58px}
.goal-blk{margin-top:10px;padding:9px 13px;border-radius:7px;font-size:13px}
.goal-blk.ok{background:#052e16;border:1px solid #166534;color:var(--green)}
.goal-blk.nok{background:#1c0505;border:1px solid #7f1d1d;color:var(--red)}
.empty-state{text-align:center;padding:80px 40px;color:var(--dim)}
.empty-state .icon{font-size:44px;margin-bottom:14px}
.badge-nav{font-size:11px;padding:2px 7px;border-radius:10px}
.badge-nav.pass{background:#14532d;color:var(--green)}
.badge-nav.fail{background:#450a0a;color:var(--red)}
.badge-nav.skip{background:#1c1917;color:var(--dim)}
.badge-nav.running{background:#1e3a5f;color:var(--cyan)}
.nav-lev{padding:7px 10px;border-radius:6px;cursor:pointer;display:flex;justify-content:space-between;align-items:center;margin-bottom:2px;transition:background .1s;font-size:13px}
.nav-lev:hover{background:var(--sur2)}

/* ── Modal ── */
.modal-bg{position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:100;display:flex;align-items:center;justify-content:center;display:none}
.modal-bg.open{display:flex}
.modal{background:var(--sur);border:1px solid var(--bor);border-radius:12px;padding:24px;min-width:360px;max-width:480px}
.modal h2{font-size:16px;font-weight:700;margin-bottom:16px}
.modal-actions{display:flex;justify-content:flex-end;gap:10px;margin-top:18px}

::-webkit-scrollbar{width:5px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:var(--bor);border-radius:3px}
.toast-container{position:fixed;bottom:20px;right:20px;display:flex;flex-direction:column;gap:8px;z-index:9999;pointer-events:none}
.toast{background:var(--sur2);border:1px solid var(--bor);color:var(--text);padding:10px 16px;border-radius:8px;box-shadow:0 4px 20px rgba(0,0,0,0.6);font-size:13px;display:flex;align-items:center;gap:8px;animation:toastIn 0.25s cubic-bezier(0.175,0.885,0.32,1.275) forwards;max-width:320px}
.toast.err{border-color:var(--red);background:#1c0505}
@keyframes toastIn{from{transform:translateX(110%);opacity:0}to{transform:translateX(0);opacity:1}}
.toast.fadeOut{animation:toastOut 0.2s ease-in forwards}
@keyframes toastOut{to{opacity:0;transform:translateY(6px)}}
/* sub-tab badge */
.stab-badge{font-size:10px;padding:1px 5px;border-radius:8px;background:var(--sur2);color:var(--dim);margin-left:4px}
.sub-tab.active .stab-badge{background:#1e2a3a;color:var(--blue)}
</style>
</head>
<body>

<div class="toast-container" id="toastContainer"></div>

<!-- Header -->
<header>
  <h1>🎯 Voice Agent Eval Pipeline</h1>
  <div class="hdr-right">
    <div style="display:flex;align-items:center;gap:7px">
      <div class="dot" id="geminiDot"></div>
      <span id="geminiLbl" style="font-size:12px">Gemini</span>
    </div>
    <div style="display:flex;align-items:center;gap:7px;margin-left:4px">
      <div class="dot" id="langsmithDot"></div>
      <span id="langsmithLbl" style="font-size:12px">LangSmith</span>
    </div>
    <span style="font-size:11px;color:var(--dim);padding:0 6px;background:var(--sur);border:1px solid var(--bor);border-radius:5px">gemini-3.1-flash-lite-preview</span>
  </div>
</header>

<!-- Body -->
<div class="body">

  <!-- Sidebar -->
  <div class="sidebar">
    <div class="sidebar-top">
      <button class="primary" style="width:100%" onclick="openNewProjectModal()">+ New Project</button>
    </div>
    <div class="proj-list" id="projList"></div>
  </div>

  <!-- Main panel -->
  <div class="panel">
    <div class="tabs" id="tabBar">
      <div class="tab active" onclick="showTab('run')">▶ Run</div>
      <div class="tab" onclick="showTab('manage')">⚙ Manage</div>
      <div class="tab" onclick="showTab('history')">📊 History</div>
      <div class="tab" onclick="showTab('settings')">🔑 Settings</div>
    </div>

    <!-- RUN TAB -->
    <div class="tab-content active" id="tab-run">
      <div class="run-controls">
        <div class="field">
          <label>Prompt Version</label>
          <select id="promptSel" disabled><option>Select a project</option></select>
        </div>
        <div class="field">
          <label>Level</label>
          <select id="levelSel" disabled>
            <option value="all">All Levels</option>
            <option value="1">Level 1 — Syntactic</option>
            <option value="2">Level 2 — Semantic</option>
            <option value="3">Level 3 — Multi-turn</option>
          </select>
        </div>
        <button class="primary" id="runBtn" onclick="startRun()" disabled>▶ Run Evals</button>
        <button id="cancelBtn" onclick="cancelRun()" style="display:none;height:36px;padding:0 14px;border-radius:8px;border:1px solid var(--red);background:transparent;color:var(--red);cursor:pointer;font-size:13px;font-weight:600">✕ Cancel</button>
        <button onclick="copyResults()" style="height:36px;font-size:13px;padding:0 14px;border-radius:7px;border:1px solid var(--bor);background:var(--sur);color:var(--text);cursor:pointer">📋 Copy Results</button>
      </div>

      <!-- Nav + results side by side -->
      <div style="display:flex;gap:0;height:calc(100vh - 210px)">
        <div style="width:170px;flex-shrink:0;border-right:1px solid var(--bor);padding:10px 8px;overflow-y:auto">
          <div style="font-size:11px;text-transform:uppercase;letter-spacing:.7px;color:var(--dim);margin-bottom:10px">Levels</div>
          <div id="levNav"></div>
        </div>
        <div style="flex:1;overflow-y:auto;padding:0 20px" id="runResults">
          <div class="empty-state" id="runEmptyState">
            <div class="icon">🎯</div>
            <p style="font-size:15px;font-weight:600;color:var(--text);margin-bottom:12px">Ready to evaluate</p>
            <div style="display:flex;flex-direction:column;gap:8px;text-align:left;max-width:280px;margin:0 auto">
              <div style="display:flex;align-items:center;gap:10px;font-size:13px">
                <span style="width:20px;height:20px;border-radius:50%;background:var(--sur2);display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;color:var(--blue);flex-shrink:0">1</span>
                <span>Select a project in the sidebar</span>
              </div>
              <div style="display:flex;align-items:center;gap:10px;font-size:13px">
                <span style="width:20px;height:20px;border-radius:50%;background:var(--sur2);display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;color:var(--blue);flex-shrink:0">2</span>
                <span>Choose a prompt version &amp; level</span>
              </div>
              <div style="display:flex;align-items:center;gap:10px;font-size:13px">
                <span style="width:20px;height:20px;border-radius:50%;background:var(--sur2);display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;color:var(--blue);flex-shrink:0">3</span>
                <span>Click <strong>▶ Run Evals</strong> <span style="font-size:11px;color:var(--dim)">(Ctrl+Enter)</span></span>
              </div>
            </div>
          </div>
        </div>
        <!-- Log panel -->
        <div id="logPanel" style="display:none;width:290px;flex-shrink:0;border-left:1px solid var(--bor);background:var(--sur);flex-direction:column;overflow:hidden">
          <div style="padding:8px 12px;border-bottom:1px solid var(--bor);font-size:11px;text-transform:uppercase;letter-spacing:.7px;color:var(--dim);display:flex;align-items:center;justify-content:space-between;flex-shrink:0">
            <span>Eval Log</span>
            <div style="display:flex;align-items:center;gap:8px">
              <span id="logCount" style="font-size:10px;color:var(--dim)">0 events</span>
              <button onclick="closeLogPanel()" title="Close log" style="height:20px;width:20px;font-size:12px;padding:0;border:none;background:none;color:var(--dim);cursor:pointer;display:flex;align-items:center;justify-content:center;border-radius:4px" onmouseover="this.style.background='var(--sur2)'" onmouseout="this.style.background='none'">✕</button>
            </div>
          </div>
          <div id="logLines" style="flex:1;overflow-y:auto;padding:8px;font-family:monospace;font-size:11px;line-height:1.7"></div>
        </div>
      </div>
    </div>

    <!-- MANAGE TAB -->
    <div class="tab-content" id="tab-manage">
      <div id="manageEmpty" class="empty-state"><div class="icon">📁</div><p>Select a project from the sidebar</p></div>
      <div id="manageContent" style="display:none">

        <!-- Sub-tabs -->
        <div class="sub-tabs">
          <div class="sub-tab active" onclick="showManageTab('config')">Config</div>
          <div class="sub-tab" onclick="showManageTab('prompts')">🗂 Prompts</div>
          <div class="sub-tab" id="stab-l1" onclick="showManageTab('l1cases')">L1 Cases<span class="stab-badge" id="badge-l1">0</span></div>
          <div class="sub-tab" id="stab-l2" onclick="showManageTab('l2cases')">L2 Cases<span class="stab-badge" id="badge-l2">0</span></div>
          <div class="sub-tab" id="stab-l3" onclick="showManageTab('l3cases')">L3 Cases<span class="stab-badge" id="badge-l3">0</span></div>
        </div>

        <!-- CONFIG editor -->
        <div id="mtab-config">
          <div class="section">
            <div class="section-title">Meta</div>
            <div class="form-row">
              <div class="form-group"><label>Project Name</label><input type="text" id="cfg-name"></div>
              <div class="form-group"><label>Description</label><input type="text" id="cfg-desc"></div>
            </div>
            <input type="hidden" id="cfg-sysprompt" value="">
          </div>

          <div class="section">
            <div class="section-title" style="cursor:pointer" onclick="toggleSec('sec-l1')">Level 1 Rules <button class="sm" onclick="event.stopPropagation();addRule()" style="margin-left:8px">+ Add Rule</button><div style="flex:1"></div><span class="chev open" id="sec-l1-chev">▶</span></div>
            <div id="sec-l1" style="display:block">
              <div id="rulesContainer"></div>
            </div>
          </div>

          <div class="section">
            <div class="section-title" style="cursor:pointer" onclick="toggleSec('sec-l2')">Level 2 Rubrics <button class="sm" onclick="event.stopPropagation();addRubric()" style="margin-left:8px">+ Add Rubric</button><div style="flex:1"></div><span class="chev open" id="sec-l2-chev">▶</span></div>
            <div id="sec-l2" style="display:block">
              <div id="rubricsContainer"></div>
            </div>
          </div>

          <div class="section">
            <div class="section-title" style="cursor:pointer" onclick="toggleSec('sec-l3')">Level 3 Settings<div style="flex:1"></div><span class="chev open" id="sec-l3-chev">▶</span></div>
            <div id="sec-l3" style="display:block">
              <div class="form-row triple">
                <div class="form-group"><label>Default Turns</label><input type="number" id="cfg-l3turns" min="1" max="20"></div>
                <div class="form-group"><label>Goal Check Model</label><input type="text" id="cfg-l3model"></div>
                <div class="form-group"></div>
              </div>
              <div class="form-group" style="margin-top:8px">
                <label>Goal Verification Prompt</label>
                <textarea id="cfg-l3prompt" rows="3"></textarea>
              </div>
              <div style="margin-top:14px">
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
                  <label style="margin:0">Per-Turn Assertions</label>
                  <button class="sm" onclick="addAssertion()" style="padding:2px 10px">+ Add</button>
                </div>
                <div id="assertionsList"></div>
              </div>
            </div>
          </div>

          <div style="display:flex;gap:10px;margin-top:4px;align-items:center">
            <button class="primary" onclick="saveConfig()">💾 Save Config</button>
            <button onclick="copyConfigYaml()" style="height:36px;font-size:13px;padding:0 14px;border-radius:7px;border:1px solid var(--bor);background:var(--sur);color:var(--text);cursor:pointer">📋 Copy YAML</button>
            <button onclick="importConfigYaml()" style="height:36px;font-size:13px;padding:0 14px;border-radius:7px;border:1px solid var(--bor);background:var(--sur);color:var(--text);cursor:pointer">📥 Import YAML</button>
          </div>
        </div>

        <!-- PROMPTS -->
        <div id="mtab-prompts" style="display:none">
          <div style="display:flex;gap:8px;margin-bottom:16px;align-items:center">
            <button class="primary sm" onclick="newPromptModal()">+ New Prompt</button>
            <span style="font-size:12px;color:var(--dim)">Files saved in <code>prompts/</code> — select one in the Run tab to use it</span>
          </div>
          <div id="promptFileList"></div>

          <!-- Inline editor -->
          <div id="promptEditor" style="display:none;margin-top:16px">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
              <span id="promptEditorTitle" style="font-weight:600;font-family:monospace"></span>
              <div style="flex:1"></div>
              <button class="sm primary" onclick="savePromptFile()">💾 Save</button>
              <button class="sm danger" onclick="deletePromptFile()" id="promptDeleteBtn">✕ Delete</button>
              <button class="sm" onclick="closePromptEditor()">Cancel</button>
            </div>
            <textarea id="promptEditorContent" rows="22" style="width:100%;font-family:monospace;font-size:12px;background:#0d1117;color:#a0aec0;border:1px solid var(--bor);border-radius:7px;padding:12px;resize:vertical"></textarea>
          </div>
        </div>

        <!-- L1 CASES -->
        <div id="mtab-l1cases" style="display:none">
          <div style="display:flex;gap:8px;margin-bottom:14px;align-items:center">
            <button class="primary sm" onclick="addCase(1)">+ Add Case</button>
            <button class="sm" onclick="importCases(1)">📥 Import JSON</button>
            <button class="sm" onclick="exportCases(1)">📋 Copy JSON</button>
          </div>
          <div id="l1CasesList"></div>
        </div>

        <!-- L2 CASES -->
        <div id="mtab-l2cases" style="display:none">
          <div style="display:flex;gap:8px;margin-bottom:14px;align-items:center">
            <button class="primary sm" onclick="addCase(2)">+ Add Case</button>
            <button class="sm" onclick="importCases(2)">📥 Import JSON</button>
            <button class="sm" onclick="exportCases(2)">📋 Copy JSON</button>
          </div>
          <div id="l2CasesList"></div>
        </div>

        <!-- L3 CASES -->
        <div id="mtab-l3cases" style="display:none">
          <div style="display:flex;gap:8px;margin-bottom:14px;align-items:center">
            <button class="primary sm" onclick="addCase(3)">+ Add Case</button>
            <button class="sm" onclick="importCases(3)">📥 Import JSON</button>
            <button class="sm" onclick="exportCases(3)">📋 Copy JSON</button>
          </div>
          <div id="l3CasesList"></div>
        </div>

      </div><!-- /manageContent -->
    </div><!-- /tab-manage -->

    <!-- HISTORY TAB -->
    <div class="tab-content" id="tab-history">
      <div id="historyEmpty" class="empty-state"><div class="icon">📊</div><p>Select a project to view its run history</p></div>
      <div id="historyContent" style="display:none">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">
          <h2 style="font-size:15px;font-weight:700">Run History Ledger</h2>
          <div style="display:flex;gap:10px;align-items:center">
            <select id="histFilter" onchange="renderLedger()" style="height:28px;font-size:12px;padding:0 8px;border-radius:5px;border:1px solid var(--bor);background:var(--sur);color:var(--text)">
              <option value="all">All Runs</option>
              <option value="fail">Failures Only</option>
              <option value="pass">Successes Only</option>
            </select>
            <button class="sm" onclick="loadLedger()">↻ Refresh</button>
          </div>
        </div>
        <div id="ledgerTable"></div>
      </div>
    </div>

    <!-- SETTINGS TAB -->
    <div class="tab-content" id="tab-settings">
      <div style="max-width:560px">

        <div class="section">
          <div class="section-title">Gemini API</div>
          <div class="form-group" style="margin-bottom:12px">
            <label>API Key</label>
            <div style="display:flex;gap:8px">
              <input type="password" id="apiKeyInput" placeholder="AIzaSy…" style="flex:1">
              <button onclick="toggleKeyVis('apiKeyInput','apiKeyEye')" id="apiKeyEye" style="height:36px;width:36px;font-size:15px;padding:0;border:1px solid var(--bor);background:var(--sur);border-radius:7px;cursor:pointer">👁</button>
            </div>
          </div>
          <div style="font-size:12px;color:var(--dim);margin-bottom:16px">Get a free key at <span style="color:var(--blue)">aistudio.google.com/apikey</span></div>
        </div>

        <div class="section">
          <div class="section-title">LangSmith <span style="font-size:11px;font-weight:400;color:var(--dim);letter-spacing:0;text-transform:none">— token &amp; trace tracking</span></div>
          <div class="form-group" style="margin-bottom:12px">
            <label>API Key</label>
            <div style="display:flex;gap:8px">
              <input type="password" id="lsKeyInput" placeholder="lsv2_pt_…" style="flex:1">
              <button onclick="toggleKeyVis('lsKeyInput','lsKeyEye')" id="lsKeyEye" style="height:36px;width:36px;font-size:15px;padding:0;border:1px solid var(--bor);background:var(--sur);border-radius:7px;cursor:pointer">👁</button>
            </div>
          </div>
          <div class="form-group" style="margin-bottom:12px">
            <label>Endpoint <span style="color:var(--dim);font-weight:400;text-transform:none;letter-spacing:0">(US or EU)</span></label>
            <select id="lsEndpointInput" style="width:320px;height:36px;border:1px solid var(--bor);background:var(--sur);color:var(--txt);border-radius:7px;padding:0 10px">
              <option value="https://api.smith.langchain.com">US — api.smith.langchain.com</option>
              <option value="https://eu.api.smith.langchain.com">EU — eu.api.smith.langchain.com</option>
            </select>
          </div>
          <div class="form-group" style="margin-bottom:16px">
            <label>Project Name <span style="color:var(--dim);font-weight:400;text-transform:none;letter-spacing:0">(traces grouped by this)</span></label>
            <input type="text" id="lsProjectInput" placeholder="voice-agent-evals" style="width:280px">
          </div>
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px">
            <button class="primary" onclick="testLangsmith()" id="lsTestBtn" style="height:34px;padding:0 14px;font-size:13px">🔗 Test Connection</button>
            <span id="lsTestResult" style="font-size:12px"></span>
          </div>
          <div style="font-size:12px;color:var(--dim);margin-bottom:16px">Get a key at <span style="color:var(--blue)">smith.langchain.com</span> → Settings → API Keys (needs write access). Token usage is posted after each LLM call (fire-and-forget, non-blocking)</div>
        </div>

        <button class="primary" onclick="saveAllSettings()" style="width:160px">💾 Save Settings</button>

      </div>
    </div>

  </div><!-- /panel -->
</div><!-- /body -->

<!-- New Project Modal -->
<div class="modal-bg" id="newProjModal">
  <div class="modal">
    <h2>New Project</h2>
    <div class="form-group">
      <label>Project Name</label>
      <input type="text" id="newProjName" placeholder="e.g. sales_bot" onkeydown="if(event.key==='Enter')confirmNewProject()">
    </div>
    <div class="modal-actions">
      <button onclick="closeNewProjectModal()">Cancel</button>
      <button class="primary" onclick="confirmNewProject()">Create</button>
    </div>
  </div>
</div>

<script>
const $=id=>document.getElementById(id);

function showToast(msg, type='success'){
  const c=$('toastContainer'); if(!c) return;
  const icons={success:'✅',error:'❌',info:'ℹ️'};
  const t=document.createElement('div');
  t.className='toast'+(type==='error'?' err':'');
  t.innerHTML=`<span style="flex-shrink:0">${icons[type]||icons.info}</span><span>${esc(msg)}</span>`;
  c.appendChild(t);
  const ttl=type==='error'?5000:2800;
  setTimeout(()=>{ t.classList.add('fadeOut'); setTimeout(()=>t.remove(),220); }, ttl);
}

// ── State ─────────────────────────────────────────────────────────────────────
let currentProject = null;
let cfgData = null;       // raw config object being edited
let casesData = {1:[],2:[],3:[]};

// ── Init ──────────────────────────────────────────────────────────────────────
(async()=>{
  await loadProjects();
  await loadSavedSettings();
  // Reconnect to a running eval, or replay last completed run
  try{
    const st=await fetch('/run-status').then(r=>r.json());
    if(st.status==='running'){
      attachToRunningEval();
    } else {
      await replayLastRun();
    }
  }catch(e){ await replayLastRun(); }
})();

document.addEventListener('keydown', e=>{
  if((e.ctrlKey||e.metaKey) && e.key==='Enter'){ e.preventDefault(); if(!$('runBtn').disabled) startRun(); }
  if(e.key==='Escape'){
    if($('newProjModal').classList.contains('open')) closeNewProjectModal();
    if($('newPromptModal')?.classList.contains('open')) closeNewPromptModal?.();
  }
});

function closeLogPanel(){
  $('logPanel').style.display='none';
}

// ── SETTINGS ──────────────────────────────────────────────────────────────────
async function loadSavedSettings(){
  try{
    const s=await fetch('/settings').then(r=>r.json());
    if(s.gemini_api_key)     $('apiKeyInput').value=s.gemini_api_key;
    if(s.langsmith_api_key)  $('lsKeyInput').value=s.langsmith_api_key;
    if(s.langsmith_project)  $('lsProjectInput').value=s.langsmith_project;
    if(s.langsmith_endpoint)  $('lsEndpointInput').value=s.langsmith_endpoint;
    updateGeminiStatus();
    updateLangsmithStatus();
  }catch{}
}

async function saveAllSettings(){
  const body={
    gemini_api_key:    $('apiKeyInput').value.trim(),
    langsmith_api_key: $('lsKeyInput').value.trim(),
    langsmith_project: $('lsProjectInput').value.trim()||'voice-agent-evals',
    langsmith_endpoint:  $('lsEndpointInput').value,
  };
  const res=await fetch('/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  if(res.ok){ updateGeminiStatus(); updateLangsmithStatus(); showToast('Settings saved'); }
  else showToast('Save failed','error');
}

// kept for Run tab Save button (legacy path)
async function saveApiKey(){
  await saveAllSettings();
}

function toggleKeyVis(inputId, btnId){
  const inp=$(inputId);
  inp.type = inp.type==='password'?'text':'password';
  $(btnId).textContent = inp.type==='password'?'👁':'🔒';
}

function updateGeminiStatus(){
  const key=$('apiKeyInput').value.trim();
  const dot=$('geminiDot'), lbl=$('geminiLbl');
  if(key){ dot.className='dot up'; lbl.textContent='Gemini'; }
  else { dot.className='dot down'; lbl.textContent='Gemini'; }
}

function updateLangsmithStatus(){
  const key=$('lsKeyInput').value.trim();
  const dot=$('langsmithDot'), lbl=$('langsmithLbl');
  if(key){ dot.className='dot up'; lbl.textContent='LangSmith'; }
  else { dot.className='dot'; lbl.textContent='LangSmith'; }
}

async function testLangsmith(){
  const key=$('lsKeyInput').value.trim();
  const project=$('lsProjectInput').value.trim()||'voice-agent-evals';
  const endpoint=$('lsEndpointInput').value;
  const btn=$('lsTestBtn'), res=$('lsTestResult');
  if(!key){ res.style.color='var(--red)'; res.textContent='No API key entered'; return; }
  btn.disabled=true; btn.textContent='Testing…'; res.textContent='';
  try{
    const r=await fetch('/test-langsmith',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({api_key:key,project,endpoint})});
    const d=await r.json();
    if(d.ok){
      res.style.color='var(--grn)';
      const matched=d.project_matched?`project "${project}" found ✓`:`project "${project}" will be auto-created`;
      res.textContent=`✓ Connected — ${matched} — IDs saved automatically`;
      $('langsmithDot').className='dot up';
    } else {
      res.style.color='var(--red)';
      res.textContent='✗ '+d.error;
    }
  }catch(e){
    res.style.color='var(--red)'; res.textContent='Network error: '+e.message;
  }finally{
    btn.disabled=false; btn.textContent='🔗 Test Connection';
  }
}

// ── Projects ──────────────────────────────────────────────────────────────────
async function loadProjects(){
  const {projects} = await fetch('/projects').then(r=>r.json());
  const list = $('projList');
  list.innerHTML = '';
  projects.forEach(p=>{
    const div = document.createElement('div');
    div.className = 'proj-item' + (p===currentProject?' active':'');
    div.innerHTML = `<span class="pname">${esc(p)}</span>
      <span class="del-btn" title="Delete project" onclick="deleteProject(event,'${esc(p)}')">✕</span>`;
    div.onclick = (e)=>{ if(!e.target.classList.contains('del-btn')) selectProject(p); };
    list.appendChild(div);
  });
}

async function selectProject(name){
  currentProject = name;
  loadProjects();
  $('promptSel').disabled=false;
  $('levelSel').disabled=false;
  $('runBtn').disabled=false;
  cfgData = await fetch(`/projects/${name}/config`).then(r=>r.json());
  for(const lvl of [1,2,3]){
    const {cases} = await fetch(`/projects/${name}/cases/${lvl}`).then(r=>r.json());
    casesData[lvl] = cases;
  }
  updateCaseBadges();
  renderConfig();
  renderCases(1); renderCases(2); renderCases(3);
  await loadPromptList();
  $('manageEmpty').style.display='none';
  $('manageContent').style.display='block';
  if($('tab-history').classList.contains('active')) loadLedger();
}

function updateCaseBadges(){
  [1,2,3].forEach(l=>{ const b=$(`badge-l${l}`); if(b) b.textContent=casesData[l]?.length||0; });
}

async function deleteProject(e, name){
  e.stopPropagation();
  if(!confirm(`Delete project "${name}"? This cannot be undone.`)) return;
  await fetch(`/projects/${name}`,{method:'DELETE'});
  if(currentProject===name){ currentProject=null; $('manageEmpty').style.display=''; $('manageContent').style.display='none'; }
  loadProjects();
}

// ── New project modal ─────────────────────────────────────────────────────────
function openNewProjectModal(){ $('newProjModal').classList.add('open'); $('newProjName').value=''; setTimeout(()=>$('newProjName').focus(),50); }
function closeNewProjectModal(){ $('newProjModal').classList.remove('open'); }
async function confirmNewProject(){
  const name = $('newProjName').value.trim();
  if(!name) return;
  const res = await fetch('/projects',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})});
  if(!res.ok){ const e=await res.json(); showToast(e.detail,'error'); return; }
  closeNewProjectModal();
  await loadProjects();
  selectProject(name.replace(/ /g,'_'));
  showTab('manage');
  showToast(`Project "${name}" created`);
}

// ── Tabs ──────────────────────────────────────────────────────────────────────
function showTab(id){
  document.querySelectorAll('.tab').forEach((t,i)=>t.classList.toggle('active',['run','manage','history','settings'][i]===id));
  document.querySelectorAll('.tab-content').forEach(t=>t.classList.remove('active'));
  $('tab-'+id).classList.add('active');
  if(id==='history' && currentProject) loadLedger();
}

function showManageTab(id){
  document.querySelectorAll('.sub-tab').forEach(t=>t.classList.remove('active'));
  event.currentTarget.classList.add('active');
  ['config','prompts','l1cases','l2cases','l3cases'].forEach(n=>{ const el=$('mtab-'+n); if(el) el.style.display=n===id?'':'none'; });
  if(id==='prompts') renderPromptFileList();
}

// ── CONFIG EDITOR ─────────────────────────────────────────────────────────────
function renderConfig(){
  if(!cfgData) return;
  const m = cfgData.meta||{};
  $('cfg-name').value = m.name||'';
  $('cfg-desc').value = m.description||'';
  $('cfg-sysprompt').value = m.system_prompt||'';
  const l3 = cfgData.level3||{};
  $('cfg-l3turns').value = l3.default_turns||4;
  $('cfg-l3model').value = l3.goal_check_model||'';
  $('cfg-l3prompt').value = l3.goal_verification_prompt||'';
  renderRules(); renderRubrics(); renderAssertions(l3.turn_assertions||[]);
}

// ── Turn Assertion Builder ─────────────────────────────────────────────────────

const ASSERTION_TYPES = [
  {value:'ends_with_question', label:'Ends with question'},
  {value:'max_words',          label:'Max words'},
  {value:'max_sentences',      label:'Max sentences'},
  {value:'no_forbidden_phrases', label:'No forbidden phrases'},
  {value:'contains_phrase',    label:'Contains phrase'},
  {value:'regex_forbidden',    label:'Regex forbidden'},
  {value:'regex_required',     label:'Regex required'},
];

function assertionParamsHtml(type, params){
  params = params||{};
  if(type==='ends_with_question') return `
    <div class="form-group" style="flex:1"><label>Max questions</label>
      <input type="number" class="ap-max_questions" value="${params.max_questions||1}" min="1"></div>`;
  if(type==='max_words') return `
    <div class="form-group" style="flex:1"><label>Max words</label>
      <input type="number" class="ap-max" value="${params.max||50}" min="1"></div>`;
  if(type==='max_sentences') return `
    <div class="form-group" style="flex:1"><label>Max sentences</label>
      <input type="number" class="ap-max" value="${params.max||3}" min="1"></div>`;
  if(type==='no_forbidden_phrases'||type==='contains_phrase') return `
    <div class="form-group" style="flex:1"><label>Phrases (one per line)</label>
      <textarea class="ap-phrases" rows="2" style="font-size:12px">${(params.phrases||[]).join('\n')}</textarea></div>`;
  if(type==='regex_forbidden'||type==='regex_required') return `
    <div class="form-group" style="flex:1"><label>Pattern (regex)</label>
      <input type="text" class="ap-pattern" value="${esc(params.pattern||'')}" style="font-family:monospace"></div>`;
  return '';
}

function buildAssertionRow(a, idx){
  const typeOpts = ASSERTION_TYPES.map(t=>`<option value="${t.value}"${a.type===t.value?' selected':''}>${t.label}</option>`).join('');
  const row = document.createElement('div');
  row.className='rule-row'; row.dataset.idx=idx;
  row.style='align-items:flex-start;gap:8px';
  row.innerHTML=`
    <div style="display:flex;flex-direction:column;gap:6px;flex:1">
      <div style="display:flex;gap:8px;align-items:center">
        <select class="a-type" style="flex:1;height:32px;border-radius:6px;border:1px solid var(--bor);background:var(--sur);color:var(--text);padding:0 8px;font-size:13px">
          ${typeOpts}
        </select>
        <input type="text" class="a-desc" placeholder="Description (shown in results)" value="${esc(a.description||a.type||'')}" style="flex:2">
      </div>
      <div class="a-params" style="display:flex;gap:8px">${assertionParamsHtml(a.type, a.params)}</div>
    </div>
    <button class="sm danger" onclick="removeAssertion(${idx})" style="margin-top:4px">✕</button>`;
  row.querySelector('.a-type').addEventListener('change', function(){
    row.querySelector('.a-params').innerHTML = assertionParamsHtml(this.value, {});
  });
  return row;
}

function renderAssertions(list){
  const c=$('assertionsList'); c.innerHTML='';
  (list||[]).forEach((a,i)=>c.appendChild(buildAssertionRow(a,i)));
}

function addAssertion(){
  const l3 = cfgData.level3||{};
  const list = [...(l3.turn_assertions||[]), {type:'ends_with_question', description:'Agent must end with a question', params:{max_questions:1}}];
  cfgData.level3 = {...l3, turn_assertions: list};
  renderAssertions(list);
}

function removeAssertion(idx){
  const l3 = cfgData.level3||{};
  const list = (l3.turn_assertions||[]).filter((_,i)=>i!==idx);
  cfgData.level3 = {...l3, turn_assertions: list};
  renderAssertions(list);
}

function collectAssertions(){
  const rows = $('assertionsList').querySelectorAll('.rule-row');
  return [...rows].map(row=>{
    const type = row.querySelector('.a-type').value;
    const desc = row.querySelector('.a-desc').value.trim();
    let params = {};
    const maxQ = row.querySelector('.ap-max_questions');
    const maxN  = row.querySelector('.ap-max');
    const phrases = row.querySelector('.ap-phrases');
    const pattern = row.querySelector('.ap-pattern');
    if(maxQ)    params.max_questions = parseInt(maxQ.value)||1;
    if(maxN)    params.max           = parseInt(maxN.value)||1;
    if(phrases) params.phrases       = phrases.value.split('\n').map(s=>s.trim()).filter(Boolean);
    if(pattern) params.pattern       = pattern.value.trim();
    return {type, description: desc||type, params};
  });
}

function toggleSec(id){
  const el=$(id), chev=$(id+'-chev');
  if(!el) return;
  const show=el.style.display==='none';
  el.style.display=show?'block':'none';
  if(chev) chev.classList.toggle('open', show);
}

// Rules
function renderRules(){
  const rules = (cfgData.level1||{}).rules||[];
  const c = $('rulesContainer'); c.innerHTML='';
  rules.forEach((r,i)=>c.appendChild(buildRuleCard(r,i)));
}

const RULE_TYPES = ['regex_forbidden','regex_required','max_sentences','starts_with_phrase','max_words'];

function buildRuleCard(rule, idx){
  const div = document.createElement('div');
  div.className='rule-card'; div.id=`rule-${idx}`;
  const typeOpts = RULE_TYPES.map(t=>`<option value="${t}"${t===rule.type?' selected':''}>${t}</option>`).join('');
  div.innerHTML=`
    <div class="rule-header" onclick="toggleRule(${idx})">
      <span class="rule-tag">${esc(rule.type||'?')}</span>
      <span class="rule-id-label">${esc(rule.id||'')}</span>
      <span class="rule-desc">${esc(rule.description||'')}</span>
      <span class="chev" id="rchev-${idx}">▶</span>
      <button class="sm danger" style="height:24px;font-size:11px" onclick="removeRule(event,${idx})">✕</button>
    </div>
    <div class="rule-body" id="rbody-${idx}">
      <div class="form-row" style="margin-top:10px">
        <div class="form-group"><label>ID</label><input type="text" id="rid-${idx}" value="${esc(rule.id||'')}"></div>
        <div class="form-group"><label>Type</label>
          <select id="rtype-${idx}" onchange="onRuleTypeChange(${idx})">${typeOpts}</select></div>
      </div>
      <div class="form-group" style="margin-bottom:10px">
        <label>Description</label><input type="text" id="rdesc-${idx}" value="${esc(rule.description||'')}">
      </div>
      <div id="rparams-${idx}">${buildRuleParams(rule,idx)}</div>
    </div>`;
  return div;
}

function buildRuleParams(rule, idx){
  const t = rule.type; const p = rule.params||{};
  if(t==='regex_forbidden'||t==='regex_required'){
    return `<div class="form-row">
      <div class="form-group"><label>Pattern (regex)</label><input type="text" id="rp-pattern-${idx}" value="${esc(p.pattern||'')}"></div>
      <div class="form-group"><label>Error Message</label><input type="text" id="rp-error_message-${idx}" value="${esc(p.error_message||'')}"></div>
    </div>`;
  }
  if(t==='max_sentences'){
    return `<div class="form-row">
      <div class="form-group"><label>Max Sentences</label><input type="number" id="rp-max-${idx}" value="${p.max||2}" min="1"></div>
      <div class="form-group"><label>Split Pattern</label><input type="text" id="rp-split_pattern-${idx}" value="${esc(p.split_pattern||'[.!?]+')}"></div>
    </div>`;
  }
  if(t==='max_words'){
    return `<div class="form-row">
      <div class="form-group"><label>Max Words</label><input type="number" id="rp-max-${idx}" value="${p.max||40}" min="1"></div>
      <div class="form-group"></div>
    </div>`;
  }
  if(t==='starts_with_phrase'){
    const phrases = (p.phrases||[]).join('\n');
    return `<div class="form-row">
      <div class="form-group"><label>First N Words</label><input type="number" id="rp-check_first_n_words-${idx}" value="${p.check_first_n_words||8}" min="1"></div>
      <div class="form-group"><label>Case Sensitive</label><select id="rp-case_sensitive-${idx}">
        <option value="false"${!p.case_sensitive?' selected':''}>No</option>
        <option value="true"${p.case_sensitive?' selected':''}>Yes</option>
      </select></div>
    </div>
    <div class="form-group" style="margin-top:6px">
      <label>Phrases (one per line)</label>
      <textarea id="rp-phrases-${idx}" rows="4">${esc(phrases)}</textarea>
    </div>`;
  }
  return '';
}

function onRuleTypeChange(idx){
  const t = $(`rtype-${idx}`).value;
  const rule = {type:t, params:{}};
  $(`rparams-${idx}`).innerHTML = buildRuleParams(rule, idx);
  // update tag
  document.querySelector(`#rule-${idx} .rule-tag`).textContent = t;
}

function toggleRule(idx){
  const b=$(`rbody-${idx}`),c=$(`rchev-${idx}`);
  const open=b.classList.toggle('open'); c.classList.toggle('open',open);
}

function addRule(){
  if(!cfgData) return;
  cfgData.level1 = cfgData.level1||{rules:[]};
  cfgData.level1.rules = cfgData.level1.rules||[];
  cfgData.level1.rules.push({id:'new_rule',type:'regex_forbidden',description:'',params:{pattern:'',error_message:''}});
  renderRules();
}

function removeRule(e,idx){
  e.stopPropagation();
  if(!cfgData?.level1?.rules) return;
  cfgData.level1.rules.splice(idx,1);
  renderRules();
}

function collectRules(){
  const rules=[];
  const count=(cfgData.level1||{}).rules?.length||0;
  for(let i=0;i<count;i++){
    const type=document.getElementById(`rtype-${i}`)?.value||'regex_forbidden';
    const rule={id:$(`rid-${i}`)?.value||'',type,description:$(`rdesc-${i}`)?.value||'',params:{}};
    if(type==='regex_forbidden'||type==='regex_required'){
      rule.params.pattern=$(`rp-pattern-${i}`)?.value||'';
      const em=$(`rp-error_message-${i}`)?.value; if(em) rule.params.error_message=em;
    } else if(type==='max_sentences'){
      rule.params.max=parseInt($(`rp-max-${i}`)?.value)||2;
      rule.params.split_pattern=$(`rp-split_pattern-${i}`)?.value||'[.!?]+';
    } else if(type==='max_words'){
      rule.params.max=parseInt($(`rp-max-${i}`)?.value)||40;
    } else if(type==='starts_with_phrase'){
      rule.params.check_first_n_words=parseInt($(`rp-check_first_n_words-${i}`)?.value)||8;
      rule.params.case_sensitive=$(`rp-case_sensitive-${i}`)?.value==='true';
      rule.params.phrases=($(`rp-phrases-${i}`)?.value||'').split('\n').map(s=>s.trim()).filter(Boolean);
    }
    rules.push(rule);
  }
  return rules;
}

// Rubrics
function renderRubrics(){
  const rubrics = (cfgData.level2||{}).rubrics||[];
  const c=$('rubricsContainer'); c.innerHTML='';
  rubrics.forEach((r,i)=>c.appendChild(buildRubricCard(r,i)));
}

function buildRubricCard(rub,idx){
  const div=document.createElement('div');
  div.className='rule-card';
  div.innerHTML=`
    <div class="rule-header" onclick="toggleRubric(${idx})">
      <span class="rule-tag" style="background:#2d1b4e;color:var(--violet)">rubric</span>
      <span class="rule-id-label">${esc(rub.id||'')}</span>
      <span class="rule-desc">${esc(rub.name||'')}</span>
      <span class="chev" id="rubchev-${idx}">▶</span>
      <button class="sm danger" style="height:24px;font-size:11px" onclick="removeRubric(event,${idx})">✕</button>
    </div>
    <div class="rule-body" id="rubbody-${idx}">
      <div class="form-row" style="margin-top:10px">
        <div class="form-group"><label>ID</label><input type="text" id="rubid-${idx}" value="${esc(rub.id||'')}"></div>
        <div class="form-group"><label>Name</label><input type="text" id="rubname-${idx}" value="${esc(rub.name||'')}"></div>
      </div>
      <div class="form-group" style="margin-bottom:6px">
        <label>Instruction (what the judge evaluates)</label>
        <textarea id="rubinstr-${idx}" rows="4">${esc(rub.instruction||'')}</textarea>
      </div>
    </div>`;
  return div;
}

function toggleRubric(idx){
  const b=$(`rubbody-${idx}`),c=$(`rubchev-${idx}`);
  const open=b.classList.toggle('open'); c.classList.toggle('open',open);
}

function addRubric(){
  if(!cfgData) return;
  cfgData.level2=cfgData.level2||{judge_model:'gemini-3.1-flash-lite-preview',temperature:0.0,verdict_format:'{"pass":true/false,"reason":"one sentence"}',rubrics:[]};
  cfgData.level2.rubrics=cfgData.level2.rubrics||[];
  cfgData.level2.rubrics.push({id:'new_rubric',name:'New Rubric',instruction:''});
  renderRubrics();
}

function removeRubric(e,idx){
  e.stopPropagation();
  if(!cfgData?.level2?.rubrics) return;
  cfgData.level2.rubrics.splice(idx,1);
  renderRubrics();
}

function collectRubrics(){
  const rubrics=[];
  const count=(cfgData.level2||{}).rubrics?.length||0;
  for(let i=0;i<count;i++){
    rubrics.push({id:$(`rubid-${i}`)?.value||'',name:$(`rubname-${i}`)?.value||'',instruction:$(`rubinstr-${i}`)?.value||''});
  }
  return rubrics;
}

async function saveConfig(){
  if(!currentProject||!cfgData) return;
  const cfg = JSON.parse(JSON.stringify(cfgData));
  cfg.meta={name:$('cfg-name').value,description:$('cfg-desc').value,system_prompt:$('cfg-sysprompt').value};
  cfg.level1={rules:collectRules()};
  cfg.level2={...cfg.level2,rubrics:collectRubrics()};
  cfg.level3={...cfg.level3,default_turns:parseInt($('cfg-l3turns').value)||4,goal_check_model:$('cfg-l3model').value,goal_verification_prompt:$('cfg-l3prompt').value,turn_assertions:collectAssertions()};
  const res=await fetch(`/projects/${currentProject}/config`,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(cfg)});
  if(res.ok){ showToast('Config saved'); cfgData=cfg; }
  else{ const e=await res.json(); showToast('Save failed: '+e.detail,'error'); }
}

// ── CASES EDITOR ──────────────────────────────────────────────────────────────
function renderCases(lvl){
  const id=['','l1CasesList','l2CasesList','l3CasesList'][lvl];
  const container=$(id); container.innerHTML='';
  (casesData[lvl]||[]).forEach((c,i)=>container.appendChild(buildCaseCard(lvl,c,i)));
}

function buildCaseCard(lvl,cas,idx){
  const div=document.createElement('div');
  div.className='case-card'; div.id=`ccase-${lvl}-${idx}`;
  const inner = buildCaseForm(lvl,cas,idx);
  div.innerHTML=`
    <div class="case-hdr" onclick="toggleCase(${lvl},${idx})">
      <span class="mono dim">${esc(cas.case_id||'')}</span>
      <span style="flex:1;margin-left:10px;font-size:13px">${esc(cas.description||'')}</span>
      <button class="sm danger" style="height:24px;font-size:11px" onclick="removeCase(event,${lvl},${idx})">✕</button>
      <span class="chev" id="ccchev-${lvl}-${idx}" style="margin-left:6px">▶</span>
    </div>
    <div class="case-body-edit" id="ccbody-${lvl}-${idx}">${inner}</div>`;
  return div;
}

function buildCaseForm(lvl,cas,idx){
  const p=cas.params||{};
  if(lvl===1){
    const expectedStr=(cas.expected_failures||[]).join(', ');
    return `<div class="form-row" style="margin-top:10px">
      <div class="form-group"><label>Case ID</label><input type="text" id="cc-id-1-${idx}" value="${esc(cas.case_id||'')}"></div>
      <div class="form-group"><label>Description</label><input type="text" id="cc-desc-1-${idx}" value="${esc(cas.description||'')}"></div>
    </div>
    <div class="form-group" style="margin-bottom:10px"><label>Agent Response</label>
      <textarea id="cc-response-1-${idx}" rows="3">${esc(cas.response||'')}</textarea></div>
    <div class="form-group" style="margin-bottom:10px"><label>Expected Failures (comma-separated rule IDs, or blank for all-pass)</label>
      <input type="text" id="cc-failures-1-${idx}" value="${esc(expectedStr)}"></div>
    <button class="sm primary" onclick="saveCase(1,${idx})">Save</button>`;
  }
  if(lvl===2){
    const expectedStr=(cas.expected_failures||[]).join(', ');
    return `<div class="form-row" style="margin-top:10px">
      <div class="form-group"><label>Case ID</label><input type="text" id="cc-id-2-${idx}" value="${esc(cas.case_id||'')}"></div>
      <div class="form-group"><label>Description</label><input type="text" id="cc-desc-2-${idx}" value="${esc(cas.description||'')}"></div>
    </div>
    <div class="form-group" style="margin-bottom:10px"><label>User Message (what the human said)</label>
      <textarea id="cc-usermsg-2-${idx}" rows="2">${esc(cas.user_message||'')}</textarea></div>
    <div class="form-group" style="margin-bottom:10px"><label>Agent Response</label>
      <textarea id="cc-response-2-${idx}" rows="3">${esc(cas.response||'')}</textarea></div>
    <div class="form-group" style="margin-bottom:10px"><label>Expected Failures (comma-separated rubric IDs, or blank for all-pass)</label>
      <input type="text" id="cc-failures-2-${idx}" value="${esc(expectedStr)}"></div>
    <button class="sm primary" onclick="saveCase(2,${idx})">Save</button>`;
  }
  if(lvl===3){
    return `<div class="form-row" style="margin-top:10px">
      <div class="form-group"><label>Case ID</label><input type="text" id="cc-id-3-${idx}" value="${esc(cas.case_id||'')}"></div>
      <div class="form-group"><label>Description</label><input type="text" id="cc-desc-3-${idx}" value="${esc(cas.description||'')}"></div>
    </div>
    <div class="form-group" style="margin-bottom:10px"><label>User Persona</label>
      <textarea id="cc-persona-3-${idx}" rows="3">${esc(cas.persona||'')}</textarea></div>
    <div class="form-group" style="margin-bottom:10px"><label>Goal State (what success looks like)</label>
      <textarea id="cc-goal-3-${idx}" rows="2">${esc(cas.goal_state||'')}</textarea></div>
    <div class="form-group" style="margin-bottom:10px"><label>Agent System Prompt</label>
      <textarea id="cc-sysprompt-3-${idx}" rows="4">${esc(cas.system_prompt||'')}</textarea></div>
    <div class="form-row">
      <div class="form-group"><label>Number of Turns</label><input type="number" id="cc-turns-3-${idx}" value="${cas.turns||4}" min="1" max="20"></div>
      <div class="form-group"><label>Opening Line (optional)</label><input type="text" id="cc-opening-3-${idx}" value="${esc(cas.opening_line||'')}"></div>
    </div>
    <button class="sm primary" style="margin-top:6px" onclick="saveCase(3,${idx})">Save</button>`;
  }
  return '';
}

function toggleCase(lvl,idx){
  const b=$(`ccbody-${lvl}-${idx}`),c=$(`ccchev-${lvl}-${idx}`);
  const open=b.classList.toggle('open'); c.classList.toggle('open',open);
}

function addCase(lvl){
  const defaults1={case_id:'new_case',description:'',response:'',expected_failures:[]};
  const defaults2={case_id:'new_case',description:'',user_message:'',response:'',expected_failures:[]};
  const defaults3={case_id:'new_case',description:'',persona:'',goal_state:'',system_prompt:'',turns:4,opening_line:''};
  const map={1:defaults1,2:defaults2,3:defaults3};
  casesData[lvl].push(map[lvl]);
  renderCases(lvl);
  updateCaseBadges();
  const idx=casesData[lvl].length-1;
  setTimeout(()=>toggleCase(lvl,idx),50);
}

function removeCase(e,lvl,idx){
  e.stopPropagation();
  casesData[lvl].splice(idx,1);
  renderCases(lvl);
  persistCases(lvl);
  updateCaseBadges();
}

function saveCase(lvl,idx){
  if(lvl===1){
    const failStr=$(`cc-failures-1-${idx}`)?.value||'';
    casesData[1][idx]={case_id:$(`cc-id-1-${idx}`).value,description:$(`cc-desc-1-${idx}`).value,
      response:$(`cc-response-1-${idx}`).value,
      expected_failures:failStr.split(',').map(s=>s.trim()).filter(Boolean)};
  } else if(lvl===2){
    const failStr=$(`cc-failures-2-${idx}`)?.value||'';
    casesData[2][idx]={case_id:$(`cc-id-2-${idx}`).value,description:$(`cc-desc-2-${idx}`).value,
      user_message:$(`cc-usermsg-2-${idx}`).value,response:$(`cc-response-2-${idx}`).value,
      expected_failures:failStr.split(',').map(s=>s.trim()).filter(Boolean)};
  } else if(lvl===3){
    casesData[3][idx]={case_id:$(`cc-id-3-${idx}`).value,description:$(`cc-desc-3-${idx}`).value,
      persona:$(`cc-persona-3-${idx}`).value,goal_state:$(`cc-goal-3-${idx}`).value,
      system_prompt:$(`cc-sysprompt-3-${idx}`).value,turns:parseInt($(`cc-turns-3-${idx}`).value)||4,
      opening_line:$(`cc-opening-3-${idx}`).value};
  }
  renderCases(lvl);
  persistCases(lvl);
  updateCaseBadges();
  showToast('Case saved');
}

async function persistCases(lvl){
  await fetch(`/projects/${currentProject}/cases/${lvl}`,{method:'PUT',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({cases:casesData[lvl]})});
}

// ── COPY CONFIG YAML ──────────────────────────────────────────────────────────
async function copyConfigYaml(){
  if(!currentProject) return;
  const {yaml}=await fetch(`/projects/${currentProject}/config/yaml`).then(r=>r.json());
  await navigator.clipboard.writeText(yaml);
  showToast('YAML copied to clipboard');
}

async function importConfigYaml(){
  const text = await navigator.clipboard.readText().catch(()=>null);
  if(!text){ showToast('Clipboard is empty or access denied','error'); return; }
  if(!confirm('Import YAML from clipboard? This will overwrite the current editor fields.')) return;
  const res=await fetch(`/projects/${currentProject}/config`,{method:'PUT',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({_raw_yaml:text})});
  if(!res.ok){ const e=await res.json(); showToast('Import failed: '+e.detail,'error'); return; }
  cfgData=await fetch(`/projects/${currentProject}/config`).then(r=>r.json());
  renderConfig();
  showToast('Config imported from clipboard');
}

// ── COPY RESULTS ──────────────────────────────────────────────────────────────
function copyResults(){
  const panel=$('runResults');
  if(!panel) return;
  // Walk the DOM and extract meaningful text
  const lines=[];
  panel.querySelectorAll('.res-card').forEach(card=>{
    const hdr=card.querySelector('.res-hdr');
    const pill=card.querySelector('.pill');
    if(hdr){
      const cid=hdr.querySelector('.mono')?.textContent||'';
      const desc=hdr.children[1]?.textContent||'';
      const status=pill?.textContent||'';
      lines.push(`\n[${status}] ${cid} — ${desc}`);
    }
    card.querySelectorAll('.res-table tbody tr').forEach(row=>{
      const cells=[...row.querySelectorAll('td')].map(td=>td.textContent.trim());
      lines.push('  '+cells.join(' | '));
    });
    // L3 turns
    card.querySelectorAll('.turn-row').forEach(row=>{
      const spk=row.querySelector('.spk')?.textContent||'';
      const msg=row.querySelector('.bub')?.textContent||'';
      lines.push(`  ${spk}: ${msg}`);
      const next=row.nextElementSibling;
      if(next && next.textContent.includes('⏱')) lines.push(`    ${next.textContent.trim()}`);
    });
    const goal=card.querySelector('.goal-blk');
    if(goal) lines.push('  '+goal.textContent.trim());
  });
  const text=lines.join('\n').trim();
  if(!text){ showToast('No results to copy yet','error'); return; }
  navigator.clipboard.writeText(text).then(()=>showToast('Results copied to clipboard'));
}

// ── RUN EVALS ─────────────────────────────────────────────────────────────────
let currentEs=null;
const navState={};

function updateLevNav(level,status,passed=0,failed=0){
  navState[level]={status,passed,failed};
  const nav=$('levNav'); nav.innerHTML='';
  [1,2,3].forEach(l=>{
    const s=navState[l]; if(!s) return;
    const div=document.createElement('div'); div.className='nav-lev'; div.onclick=()=>scrollToLev(l);
    let bc=s.status,bt='';
    if(s.status==='running'){bt='…';bc='running';}
    else if(s.status==='skip'){bt='skip';bc='skip';}
    else if(s.status==='done'){bt=`${s.passed}P/${s.failed}F`;bc=s.failed>0?'fail':'pass';}
    div.innerHTML=`<span>Level ${l}</span><span class="badge-nav ${bc}">${bt}</span>`;
    nav.appendChild(div);
  });
}

function scrollToLev(l){ const el=document.getElementById(`rlev-${l}`); if(el) el.scrollIntoView({behavior:'smooth'}); }

function getOrCreateLevSection(level,title){
  let el=document.getElementById(`rlev-${level}`);
  if(!el){
    el=document.createElement('div'); el.id=`rlev-${level}`;
    el.innerHTML=`<div class="level-hdr">L${level} — ${(title.split('—')[1]||title).trim()}</div><div id="rlev-${level}-body"></div>`;
    $('runResults').appendChild(el);
  }
  return document.getElementById(`rlev-${level}-body`);
}

async function replayLastRun(){
  try{
    const d=await fetch('/last-run').then(r=>r.json());
    if(!d.events||d.events.length===0) return;
    $('runResults').innerHTML='';
    $('levNav').innerHTML='';
    Object.keys(navState).forEach(k=>delete navState[k]);
    let containers={};
    for(const ev of d.events){
      if(ev.type==='meta'){
        const el=document.createElement('div'); el.style='margin-bottom:16px;color:var(--dim);font-size:13px';
        el.innerHTML=`Project: <strong style="color:var(--text)">${esc(ev.name)}</strong> &nbsp;·&nbsp; Provider: <span style="color:var(--violet)">Gemini</span> &nbsp;·&nbsp; Model: <code>${esc(ev.model)}</code> &nbsp;·&nbsp; Prompt: <code style="color:var(--green)">${esc(ev.prompt||'default.xml')}</code> &nbsp;<span style="font-size:11px;background:rgba(255,255,255,0.06);padding:2px 6px;border-radius:4px;margin-left:4px">restored</span>`;
        $('runResults').appendChild(el);
      }
      if(ev.type==='level_start'){updateLevNav(ev.level,'done');containers[ev.level]=getOrCreateLevSection(ev.level,ev.title);}
      if(ev.type==='level_skip'){updateLevNav(ev.level,'skip');if(containers[ev.level]){const s=document.createElement('div');s.className='skip-box';s.textContent='⚠ '+ev.msg;containers[ev.level].appendChild(s);}}
      if(ev.type==='l1_case') renderL1Card(ev,containers[1]);
      if(ev.type==='l2_case') renderL2Card(ev,containers[2]);
      if(ev.type==='l3_case') renderL3Card(ev,containers[3]);
      if(ev.type==='level_summary'){updateLevNav(ev.level,'done',ev.passed,ev.failed);if(containers[ev.level])appendSumBar(containers[ev.level],ev.level,ev.passed,ev.failed);}
    }
    updateCaseBadges();
  }catch(e){ /* no last run, that's fine */ }
}

async function startRun(){
  if(!currentProject){ alert('Select a project first'); return; }
  if(currentEs){currentEs.close();currentEs=null;}
  const level=$('levelSel').value;
  $('runResults').innerHTML='';
  $('levNav').innerHTML='';
  $('logLines').innerHTML='';
  $('logCount').textContent='0 events';
  $('logPanel').style.display='flex';
  $('logPanel').style.flexDirection='column';
  Object.keys(navState).forEach(k=>delete navState[k]);
  const btn=$('runBtn'); btn.disabled=true; btn.textContent='⏳ Running…';
  const cancelBtn=$('cancelBtn'); cancelBtn.style.display='inline-flex';
  let logCount=0;
  const promptFile=($('promptSel')?.value)||'default.xml';
  const apiKey=$('apiKeyInput').value.trim();
  const es=new EventSource(`/run?project=${encodeURIComponent(currentProject)}&level=${encodeURIComponent(level)}&prompt=${encodeURIComponent(promptFile)}&api_key=${encodeURIComponent(apiKey)}`);
  currentEs=es; let containers={};
  es.onmessage=(e)=>{
    const ev=JSON.parse(e.data);
    if(ev.type==='error'){$('runResults').innerHTML=`<div class="warn-box" style="margin:16px">Error: ${esc(ev.msg)}</div>`;done();return;}
    if(ev.type==='meta'){
      const d=document.createElement('div'); d.style='margin-bottom:16px;color:var(--dim);font-size:13px';
      d.innerHTML=`Project: <strong style="color:var(--text)">${esc(ev.name)}</strong> &nbsp;·&nbsp; Provider: <span style="color:var(--violet)">Gemini</span> &nbsp;·&nbsp; Model: <code>${esc(ev.model)}</code> &nbsp;·&nbsp; Prompt: <code style="color:var(--green)">${esc(ev.prompt||'default.xml')}</code>`;
      $('runResults').appendChild(d);
    }
    if(ev.type==='level_start'){updateLevNav(ev.level,'running');containers[ev.level]=getOrCreateLevSection(ev.level,ev.title);}
    if(ev.type==='level_skip'){updateLevNav(ev.level,'skip');if(containers[ev.level]){const s=document.createElement('div');s.className='skip-box';s.textContent='⚠ '+ev.msg;containers[ev.level].appendChild(s);}}
    if(ev.type==='l1_case') renderL1Card(ev,containers[1]);
    if(ev.type==='l2_thinking') renderThinkingCard(ev,'l2',containers[2]);
    if(ev.type==='l2_case') renderL2Card(ev,containers[2]);
    if(ev.type==='l3_thinking') renderThinkingCard(ev,'l3',containers[3]);
    if(ev.type==='l3_case') renderL3Card(ev,containers[3]);
    if(ev.type==='level_summary'){updateLevNav(ev.level,'done',ev.passed,ev.failed);if(containers[ev.level])appendSumBar(containers[ev.level],ev.level,ev.passed,ev.failed);}
    if(ev.type==='log'){
      logCount++;
      $('logCount').textContent=logCount+' event'+(logCount===1?'':'s');
      const line=document.createElement('div');
      const ts=new Date().toLocaleTimeString('en-US',{hour12:false,hour:'2-digit',minute:'2-digit',second:'2-digit'});
      const isPass=ev.msg.includes('PASS'), isFail=ev.msg.includes('FAIL'), isCancel=ev.msg.includes('cancel');
      line.style.color=isPass?'var(--green)':isFail?'var(--red)':isCancel?'var(--yellow)':'var(--dim)';
      line.textContent=`${ts}  ${ev.msg}`;
      $('logLines').appendChild(line);
      $('logLines').scrollTop=$('logLines').scrollHeight;
    }
    if(ev.type==='done') done(true);
    if(ev.type==='cancelled'){
      const d=document.createElement('div');
      d.style='margin:12px 0;padding:8px 12px;border-radius:6px;background:#1c1400;border:1px solid #92400e;color:var(--yellow);font-size:13px';
      d.textContent='⚠ Eval cancelled by user';
      $('runResults').appendChild(d);
      done();
    }
  };
  es.onerror=()=>done(false);
  function done(ok=true){if(currentEs){currentEs.close();currentEs=null;}btn.disabled=false;btn.textContent='▶ Run Evals';cancelBtn.style.display='none';if(ok)showToast('Eval run complete','info');}
}

function attachToRunningEval(){
  // Reconnect to an eval that kept running after a page refresh.
  // The server replays all past events then streams live ones.
  if(currentEs){currentEs.close();currentEs=null;}
  $('runResults').innerHTML='';
  $('levNav').innerHTML='';
  $('logLines').innerHTML='';
  $('logCount').textContent='0 events';
  $('logPanel').style.display='flex';
  $('logPanel').style.flexDirection='column';
  Object.keys(navState).forEach(k=>delete navState[k]);
  const btn=$('runBtn'); btn.disabled=true; btn.textContent='⏳ Running…';
  $('cancelBtn').style.display='inline-flex';
  showToast('Reconnected to running eval','info');
  let logCount=0;
  // Connect without project/level/prompt — server knows the running eval
  const es=new EventSource('/run');
  currentEs=es; let containers={};
  es.onmessage=(e)=>{
    const ev=JSON.parse(e.data);
    if(ev.type==='error'){$('runResults').innerHTML=`<div class="warn-box" style="margin:16px">Error: ${esc(ev.msg)}</div>`;rdone();return;}
    if(ev.type==='meta'){
      const d=document.createElement('div'); d.style='margin-bottom:16px;color:var(--dim);font-size:13px';
      d.innerHTML=`Project: <strong style="color:var(--text)">${esc(ev.name)}</strong> &nbsp;·&nbsp; Provider: <span style="color:var(--violet)">Gemini</span> &nbsp;·&nbsp; Model: <code>${esc(ev.model)}</code> &nbsp;·&nbsp; Prompt: <code style="color:var(--green)">${esc(ev.prompt||'default.xml')}</code> &nbsp;<span style="font-size:11px;background:rgba(16,185,129,0.12);color:var(--green);padding:2px 6px;border-radius:4px;margin-left:4px">reconnected</span>`;
      $('runResults').appendChild(d);
    }
    if(ev.type==='level_start'){updateLevNav(ev.level,'running');containers[ev.level]=getOrCreateLevSection(ev.level,ev.title);}
    if(ev.type==='level_skip'){updateLevNav(ev.level,'skip');if(containers[ev.level]){const s=document.createElement('div');s.className='skip-box';s.textContent='⚠ '+ev.msg;containers[ev.level].appendChild(s);}}
    if(ev.type==='l1_case') renderL1Card(ev,containers[1]);
    if(ev.type==='l2_thinking') renderThinkingCard(ev,'l2',containers[2]);
    if(ev.type==='l2_case') renderL2Card(ev,containers[2]);
    if(ev.type==='l3_thinking') renderThinkingCard(ev,'l3',containers[3]);
    if(ev.type==='l3_case') renderL3Card(ev,containers[3]);
    if(ev.type==='level_summary'){updateLevNav(ev.level,'done',ev.passed,ev.failed);if(containers[ev.level])appendSumBar(containers[ev.level],ev.level,ev.passed,ev.failed);}
    if(ev.type==='log'){
      logCount++;
      $('logCount').textContent=logCount+' event'+(logCount===1?'':'s');
      const line=document.createElement('div');
      const ts=new Date().toLocaleTimeString('en-US',{hour12:false,hour:'2-digit',minute:'2-digit',second:'2-digit'});
      const isPass=ev.msg.includes('PASS'),isFail=ev.msg.includes('FAIL'),isCancel=ev.msg.includes('cancel');
      line.style.color=isPass?'var(--green)':isFail?'var(--red)':isCancel?'var(--yellow)':'var(--dim)';
      line.textContent=`${ts}  ${ev.msg}`;
      $('logLines').appendChild(line);
      $('logLines').scrollTop=$('logLines').scrollHeight;
    }
    if(ev.type==='done') rdone(true);
    if(ev.type==='cancelled'){
      const d=document.createElement('div');
      d.style='margin:12px 0;padding:8px 12px;border-radius:6px;background:#1c1400;border:1px solid #92400e;color:var(--yellow);font-size:13px';
      d.textContent='⚠ Eval cancelled by user'; $('runResults').appendChild(d); rdone();
    }
  };
  es.onerror=()=>rdone(false);
  function rdone(ok=true){if(currentEs){currentEs.close();currentEs=null;}btn.disabled=false;btn.textContent='▶ Run Evals';$('cancelBtn').style.display='none';if(ok)showToast('Eval run complete','info');}
}

async function cancelRun(){
  if(currentEs){currentEs.close();currentEs=null;}
  await fetch('/cancel',{method:'POST'});
  const btn=$('runBtn'); btn.disabled=false; btn.textContent='▶ Run Evals';
  $('cancelBtn').style.display='none';
}

function toggleCard(id){const b=$(id+'-body'),c=$(id+'-chev');if(!b)return;const o=b.classList.toggle('open');if(c)c.classList.toggle('open',o);}

function renderL1Card(ev,container){
  const id=`rc-${ev.case_id.replace(/[^a-z0-9]/gi,'-')}`;
  const div=document.createElement('div'); div.className='res-card'; div.id=id;
  const rulesHtml=ev.rules.map(r=>`<tr>
    <td class="mono">${esc(r.rule_id)}</td>
    <td>${r.passed?'<span class="pass-ic">PASS</span>':'<span class="fail-ic">FAIL</span>'}</td>
    <td><span class="dim">${esc(r.detail)}</span>${r.excerpt?`<br><span class="excerpt">${esc(r.excerpt)}</span>`:''}</td>
  </tr>`).join('');
  const warns=[...ev.unexpected_passes.map(x=>`⚠ Expected fail but passed: <code>${x}</code>`),...ev.unexpected_failures.map(x=>`⚠ Unexpected fail: <code>${x}</code>`)].map(w=>`<div class="warn-box">${w}</div>`).join('');
  div.innerHTML=`<div class="res-hdr" onclick="toggleCard('${id}')">
    <span class="mono dim">${esc(ev.case_id)}</span>
    <span style="flex:1">${esc(ev.description)}</span>
    <span class="pill ${ev.passed?'pass':'fail'}">${ev.passed?'PASS':'FAIL'}</span>
    <span class="chev" id="${id}-chev" style="margin-left:8px">▶</span>
  </div>
  <div class="res-body" id="${id}-body">
    <div class="resp-lbl" style="margin-top:10px">Response</div>
    <div class="resp-box">${esc(ev.response)}</div>
    <table class="res-table"><thead><tr><th>Rule</th><th>Status</th><th>Detail</th></tr></thead><tbody>${rulesHtml}</tbody></table>
    ${warns}
  </div>`;
  container.appendChild(div);
}

function renderThinkingCard(ev,prefix,container){
  const id=`rc-${prefix}-${ev.case_id.replace(/[^a-z0-9]/gi,'-')}`;
  const div=document.createElement('div'); div.className='res-card'; div.id=id;
  div.innerHTML=`<div class="res-hdr"><span class="mono dim">${esc(ev.case_id)}</span><span style="flex:1">Evaluating…</span><span class="pill running">RUNNING</span></div>`;
  container.appendChild(div);
}

function renderL2Card(ev,container){
  const id=`rc-l2-${ev.case_id.replace(/[^a-z0-9]/gi,'-')}`;
  const existing=document.getElementById(id);
  const rubricsHtml=ev.rubrics.map(r=>`<tr>
    <td class="mono">${esc(r.rubric_id)}</td><td>${esc(r.name)}</td>
    <td>${r.passed?'<span class="pass-ic">PASS</span>':'<span class="fail-ic">FAIL</span>'}</td>
    <td class="dim">${esc(r.reason)}</td>
  </tr>`).join('');
  const warns=[...ev.unexpected_passes.map(x=>`⚠ Expected fail but passed: <code>${x}</code>`),...ev.unexpected_failures.map(x=>`⚠ Unexpected fail: <code>${x}</code>`)].map(w=>`<div class="warn-box">${w}</div>`).join('');
  const html=`<div class="res-hdr" onclick="toggleCard('${id}')">
    <span class="mono dim">${esc(ev.case_id)}</span><span style="flex:1">${esc(ev.description)}</span>
    <span class="pill ${ev.passed?'pass':'fail'}">${ev.passed?'PASS':'FAIL'}</span>
    <span class="chev" id="${id}-chev" style="margin-left:8px">▶</span>
  </div>
  <div class="res-body" id="${id}-body">
    ${ev.user_message?`<div class="resp-lbl" style="margin-top:10px">User Message</div><div class="resp-box">${esc(ev.user_message)}</div>`:''}
    <div class="resp-lbl" style="margin-top:10px">Agent Response</div>
    <div class="resp-box">${esc(ev.response)}</div>
    <table class="res-table"><thead><tr><th>Rubric</th><th>Name</th><th>Status</th><th>Reason</th></tr></thead><tbody>${rubricsHtml}</tbody></table>
    ${warns}
  </div>`;
  if(existing) existing.innerHTML=html;
  else{const div=document.createElement('div');div.className='res-card';div.id=id;div.innerHTML=html;container.appendChild(div);}
}

function renderL3Card(ev,container){
  const id=`rc-l3-${ev.case_id.replace(/[^a-z0-9]/gi,'-')}`;
  const existing=document.getElementById(id);
  let turnsHtml='';
  for(const t of ev.turns){
    const asserts=t.assertions.map(a=>`<div class="assert-row"><span class="${a.passed?'pass-ic':'fail-ic'}">${a.passed?'✓':'✗'}</span><span>${esc(a.assertion)}: ${esc(a.detail)}</span></div>`).join('');
    turnsHtml+=`<div style="margin-bottom:14px">
      <div style="font-size:10px;text-transform:uppercase;letter-spacing:.6px;color:var(--dim);margin-bottom:6px">Turn ${t.turn}</div>
      <div class="turn-row"><span class="spk u">USER</span><div class="bub">${esc(t.user)}</div></div>
      <div style="text-align:center;margin:2px 0"><span style="font-size:10px;color:var(--dim);background:var(--sur);border:1px solid var(--bor);border-radius:4px;padding:1px 6px">⏱ ${t.latency_ms!=null?(t.latency_ms/1000).toFixed(1)+'s':'?'}</span></div>
      <div class="turn-row"><span class="spk a">AGENT</span><div class="bub">${esc(t.agent)}</div></div>
      ${asserts}</div>`;
  }
  const html=`<div class="res-hdr" onclick="toggleCard('${id}')">
    <span class="mono dim">${esc(ev.case_id)}</span><span style="flex:1">${esc(ev.description)}</span>
    <span class="pill ${ev.passed?'pass':'fail'}">${ev.passed?'PASS':'FAIL'}</span>
    <span class="chev" id="${id}-chev" style="margin-left:8px">▶</span>
  </div>
  <div class="res-body" id="${id}-body">
    ${ev.error?`<div class="warn-box">Error: ${esc(ev.error)}</div>`:''}
    <div style="margin-top:10px">${turnsHtml}</div>
    <div class="goal-blk ${ev.goal_achieved?'ok':'nok'}"><strong>${ev.goal_achieved?'✓ Goal achieved':'✗ Goal not met'}</strong> — ${esc(ev.goal_reason)}</div>
  </div>`;
  if(existing) existing.innerHTML=html;
  else{const div=document.createElement('div');div.className='res-card';div.id=id;div.innerHTML=html;container.appendChild(div);}
}

function appendSumBar(container,level,passed,failed){
  const bar=document.createElement('div'); bar.className='sum-bar';
  bar.innerHTML=`<h3>Level ${level} Summary</h3>
    <div class="stat"><div class="num pass">${passed}</div><div class="lbl">Passed</div></div>
    <div class="stat"><div class="num fail">${failed}</div><div class="lbl">Failed</div></div>
    <div class="stat"><div class="num" style="color:var(--dim)">${passed+failed}</div><div class="lbl">Total</div></div>`;
  container.appendChild(bar);
}

// ── PROMPT VERSION MANAGEMENT ─────────────────────────────────────────────────
let promptFiles = [];
let editingPrompt = null;

async function loadPromptList(){
  if(!currentProject) return;
  const {prompts} = await fetch(`/projects/${currentProject}/prompts`).then(r=>r.json());
  promptFiles = prompts;
  // Update run tab selector
  const sel=$('promptSel'); sel.innerHTML='';
  prompts.forEach(p=>{
    const opt=document.createElement('option');
    opt.value=opt.textContent=p;
    if(p==='default.xml') opt.selected=true;
    sel.appendChild(opt);
  });
}

function renderPromptFileList(){
  const c=$('promptFileList'); c.innerHTML='';
  if(!promptFiles.length){ c.innerHTML='<div class="skip-box">No prompt files yet. Click + New Prompt.</div>'; return; }
  promptFiles.forEach(name=>{
    const div=document.createElement('div');
    div.className='rule-card';
    div.style='margin-bottom:8px';
    div.innerHTML=`<div class="rule-header" style="cursor:pointer" onclick="openPromptEditor('${esc(name)}')">
      <span class="rule-tag" style="background:#1a2a1a;color:var(--green)">prompt</span>
      <span style="font-family:monospace;font-size:13px;flex:1">${esc(name)}</span>
      <span style="font-size:12px;color:var(--dim)">click to edit</span>
    </div>`;
    c.appendChild(div);
  });
}

async function openPromptEditor(filename){
  const {content} = await fetch(`/projects/${currentProject}/prompts/${encodeURIComponent(filename)}`).then(r=>r.json());
  editingPrompt = filename;
  $('promptEditorTitle').textContent = filename;
  $('promptEditorContent').value = content;
  $('promptEditor').style.display='block';
  $('promptDeleteBtn').style.display = filename==='default.xml'?'none':'';
  $('promptSaveMsg').textContent='';
}

function closePromptEditor(){ $('promptEditor').style.display='none'; editingPrompt=null; }

async function savePromptFile(){
  if(!editingPrompt) return;
  const content=$('promptEditorContent').value;
  const res=await fetch(`/projects/${currentProject}/prompts/${encodeURIComponent(editingPrompt)}`,{
    method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({content})});
  if(res.ok) showToast(`Saved ${editingPrompt}`);
  else showToast('Save failed','error');
}

async function deletePromptFile(){
  if(!editingPrompt||editingPrompt==='default.xml') return;
  if(!confirm(`Delete "${editingPrompt}"?`)) return;
  await fetch(`/projects/${currentProject}/prompts/${encodeURIComponent(editingPrompt)}`,{method:'DELETE'});
  closePromptEditor();
  await loadPromptList();
  renderPromptFileList();
}

async function newPromptModal(){
  const filename=prompt('Prompt filename (e.g. v2_strict.xml):','v2_strict.xml');
  if(!filename) return;
  await fetch(`/projects/${currentProject}/prompts`,{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({filename,content:''})});
  await loadPromptList();
  renderPromptFileList();
  openPromptEditor(filename.endsWith('.xml')||filename.endsWith('.txt')||filename.endsWith('.md')?filename:filename+'.xml');
}

// ── HISTORY LEDGER ────────────────────────────────────────────────────────────
let currentLedgerEntries = [];
async function loadLedger(){
  if(!currentProject){ $('historyEmpty').style.display='';$('historyContent').style.display='none'; return; }
  $('historyEmpty').style.display='none'; $('historyContent').style.display='';
  const {entries} = await fetch(`/projects/${currentProject}/ledger`).then(r=>r.json());
  currentLedgerEntries = entries.map((e,i)=>({...e,_origIdx:i})).reverse();
  renderLedger();
}

function renderLedger(){
  const c=$('ledgerTable'); c.innerHTML='';
  if(!currentLedgerEntries.length){ c.innerHTML='<div class="skip-box">No runs recorded yet. Run evals to populate the ledger.</div>'; return; }
  const f=$('histFilter').value;
  const filtered = currentLedgerEntries.filter(e=>{
    const totalFail=(e.L1?.failed||0)+(e.L2?.failed||0)+(e.L3?.failed||0);
    if(f==='fail') return totalFail>0;
    if(f==='pass') return totalFail===0;
    return true;
  });
  if(!filtered.length){ c.innerHTML='<div class="skip-box">No runs match this filter.</div>'; return; }

  let html=`<table class="res-table"><thead><tr>
    <th>Timestamp</th><th>Prompt</th><th>Model</th>
    <th style="color:var(--cyan)">L1 P/F</th>
    <th style="color:var(--violet)">L2 P/F</th>
    <th style="color:var(--blue)">L3 P/F</th>
    <th>Result</th><th></th>
  </tr></thead><tbody>`;

  for(const e of filtered){
    const ts=new Date(e.timestamp).toLocaleString();
    const fmt=(lvl)=>{
      const d=e[lvl]; if(!d) return '<span style="color:var(--dim)">—</span>';
      return `<span class="${d.failed>0?'fail-ic':'pass-ic'}">${d.passed}/${d.failed}</span>`;
    };
    const totalFail=(e.L1?.failed||0)+(e.L2?.failed||0)+(e.L3?.failed||0);
    const ok=totalFail===0;
    html+=`<tr>
      <td class="dim">${ts}</td>
      <td><span class="mono" style="color:var(--green)">${esc(e.prompt_file)}</span></td>
      <td class="dim mono">${esc(e.model||e.ollama_model||'—')}</td>
      <td>${fmt('L1')}</td><td>${fmt('L2')}</td><td>${fmt('L3')}</td>
      <td><span class="pill ${ok?'pass':'fail'}" style="font-size:10px">${ok?'PASS':'FAIL'}</span></td>
      <td><button onclick="deleteLedgerEntry(${e._origIdx})" style="background:none;border:none;color:var(--red);cursor:pointer;font-size:14px;padding:2px 6px;border-radius:4px;opacity:.6" title="Delete">✕</button></td>
    </tr>`;
  }
  html+='</tbody></table>';
  c.innerHTML=html;
}

async function deleteLedgerEntry(idx){
  if(!confirm('Delete this history entry?')) return;
  await fetch(`/projects/${encodeURIComponent(currentProject)}/ledger/${idx}`,{method:'DELETE'});
  loadLedger();
}

// ── IMPORT / EXPORT CASES ─────────────────────────────────────────────────────
async function importCases(lvl){
  const text = await navigator.clipboard.readText().catch(()=>null);
  if(!text){ showToast('Clipboard is empty or access denied','error'); return; }
  let parsed;
  try{ parsed = JSON.parse(text); }
  catch(e){ showToast('Invalid JSON: '+e.message,'error'); return; }
  if(!Array.isArray(parsed)){ showToast('JSON must be an array [ {...}, ... ]','error'); return; }
  if(!confirm(`Import ${parsed.length} case(s) into L${lvl}? This will REPLACE all existing L${lvl} cases.`)) return;
  casesData[lvl] = parsed;
  renderCases(lvl);
  await persistCases(lvl);
  updateCaseBadges();
  showToast(`Imported ${parsed.length} L${lvl} case(s)`);
}

async function exportCases(lvl){
  const text = JSON.stringify(casesData[lvl], null, 2);
  await navigator.clipboard.writeText(text);
  showToast(`${casesData[lvl].length} L${lvl} case(s) copied to clipboard`);
}

function esc(s){return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}
</script>
</body>
</html>"""


if __name__ == "__main__":
    port = 7860
    print(f"Voice Agent Eval UI -> http://localhost:{port}")
    Thread(target=lambda: (time.sleep(1.2), webbrowser.open(f"http://localhost:{port}")), daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
