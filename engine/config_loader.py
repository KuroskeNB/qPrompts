"""
engine/config_loader.py
───────────────────────
Loads and validates a project_config.yaml file.
Also provides load_prompt() for the file-based prompt versioning system.

Usage:
    from engine.config_loader import load_config, load_prompt
    config = load_config(Path("projects/dental_booking_bot"))
    config = load_prompt(config, Path("projects/dental_booking_bot"), "default.xml")
"""

from __future__ import annotations

from pathlib import Path

import jsonschema
import yaml


# ── Validation schema ──────────────────────────────────────────────────────────

_RULE_TYPES = ["regex_forbidden", "regex_required", "max_sentences",
               "starts_with_phrase", "max_words"]

_SCHEMA = {
    "type": "object",
    "required": ["meta", "level1", "level2", "level3"],
    "additionalProperties": False,
    "properties": {

        "meta": {
            "type": "object",
            "required": ["name"],
            # system_prompt intentionally omitted — managed via prompts/ files
            "properties": {
                "name":          {"type": "string"},
                "description":   {"type": "string"},
                "ollama_model":  {"type": "string"},
                "ollama_url":    {"type": "string"},
                "system_prompt": {"type": "string"},
            },
        },

        "level1": {
            "type": "object",
            "required": ["rules"],
            "properties": {
                "rules": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["id", "type", "description", "params"],
                        "properties": {
                            "id":          {"type": "string"},
                            "type":        {"type": "string", "enum": _RULE_TYPES},
                            "description": {"type": "string"},
                            "params":      {"type": "object"},
                        },
                    },
                },
            },
        },

        "level2": {
            "type": "object",
            "required": ["judge_model", "temperature", "verdict_format", "rubrics"],
            "properties": {
                "judge_model":    {"type": "string"},
                "temperature":    {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "verdict_format": {"type": "string"},
                "rubrics": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["id", "name", "instruction"],
                        "properties": {
                            "id":          {"type": "string"},
                            "name":        {"type": "string"},
                            "instruction": {"type": "string"},
                        },
                    },
                },
            },
        },

        "level3": {
            "type": "object",
            "required": ["default_turns", "goal_check_model",
                         "turn_assertions", "goal_verification_prompt"],
            "properties": {
                "default_turns":            {"type": "integer", "minimum": 1},
                "goal_check_model":         {"type": "string"},
                "turn_assertions":          {"type": "array"},
                "goal_verification_prompt": {"type": "string"},
            },
        },
    },
}


# ── Public exceptions ──────────────────────────────────────────────────────────

class ConfigValidationError(Exception):
    """Raised when project_config.yaml fails schema validation."""

class PromptNotFoundError(FileNotFoundError):
    """Raised when the requested prompt file does not exist in prompts/."""


# ── Public API ─────────────────────────────────────────────────────────────────

def load_config(project_path: Path) -> dict:
    """
    Load and validate project_config.yaml inside `project_path`.
    Returns the config dict. Does NOT inject the system prompt —
    call load_prompt() separately to do that.
    """
    config_file = project_path / "project_config.yaml"

    if not config_file.exists():
        raise FileNotFoundError(
            f"No project_config.yaml found at: {config_file}\n"
            f"Expected path: {config_file.resolve()}"
        )

    with config_file.open(encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ConfigValidationError(f"YAML parse error in {config_file}:\n{exc}") from exc

    if not isinstance(config, dict):
        raise ConfigValidationError(
            f"{config_file} parsed to {type(config).__name__}, expected a mapping."
        )

    _validate(config, config_file)
    _check_duplicate_ids(config, config_file)

    return config


def load_prompt(config: dict, project_path: Path, prompt_filename: str = "default.xml") -> dict:
    """
    Read a prompt file from `project_path/prompts/` and inject its content
    into config["meta"]["system_prompt"], overriding any value already there.

    Falls back gracefully:
      1. Exact filename requested
      2. default.xml (if a different file was requested but missing)
      3. Raises PromptNotFoundError if neither exists

    Returns the modified config dict (same object, mutated in place).
    """
    prompts_dir = project_path / "prompts"
    target = prompts_dir / prompt_filename

    if not target.exists():
        # Try default as fallback if a specific version was requested
        fallback = prompts_dir / "default.xml"
        if prompt_filename != "default.xml" and fallback.exists():
            import warnings
            warnings.warn(
                f"Prompt '{prompt_filename}' not found — falling back to default.xml",
                stacklevel=2,
            )
            target = fallback
        else:
            raise PromptNotFoundError(
                f"Prompt file not found: {target}\n"
                f"Available prompts: {list_prompts(project_path)}"
            )

    content = target.read_text(encoding="utf-8").strip()
    config["meta"]["system_prompt"] = content
    config["meta"]["_prompt_file"] = prompt_filename          # track which file was loaded
    config["meta"]["_prompt_stem"] = Path(prompt_filename).stem
    return config


def list_prompts(project_path: Path) -> list[str]:
    """Return sorted list of prompt filenames in project_path/prompts/."""
    d = project_path / "prompts"
    if not d.exists():
        return []
    return sorted(f.name for f in d.iterdir() if f.is_file() and not f.name.startswith("."))


# ── Internal helpers ───────────────────────────────────────────────────────────

def _validate(config: dict, config_file: Path) -> None:
    validator = jsonschema.Draft7Validator(
        _SCHEMA,
        format_checker=jsonschema.FormatChecker(),
    )
    errors = sorted(validator.iter_errors(config), key=lambda e: list(e.path))

    if errors:
        messages = []
        for err in errors:
            path = " → ".join(str(p) for p in err.absolute_path) or "(root)"
            messages.append(f"  [{path}] {err.message}")
        raise ConfigValidationError(
            f"project_config.yaml validation failed ({len(errors)} error(s)):\n"
            + "\n".join(messages)
        )


def _check_duplicate_ids(config: dict, config_file: Path) -> None:
    for section, key in [("level1", "rules"), ("level2", "rubrics")]:
        items = config.get(section, {}).get(key, [])
        seen: set[str] = set()
        for item in items:
            item_id = item.get("id", "")
            if item_id in seen:
                raise ConfigValidationError(
                    f"Duplicate id '{item_id}' found in {section}.{key} "
                    f"in {config_file}"
                )
            seen.add(item_id)
