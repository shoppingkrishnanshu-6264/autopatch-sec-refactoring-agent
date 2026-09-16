"""
patch_parser.py
----------------
Parses the Refactoring Agent's raw model output into a validated,
structured PatchResult. Handles the common failure modes of quantized
7-8B models under JSON-output instructions:
  - Wrapping the JSON in ```json ... ``` fences anyway
  - Leading/trailing prose despite instructions not to
  - Minor trailing-comma / smart-quote issues
  - Missing optional fields

Also provides clean-diff extraction (unified diff between original and
patched function) for the PR/Review Agent to consume.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from typing import Optional


class PatchStatus(str, Enum):
    OK = "ok"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    PARSE_ERROR = "parse_error"  # assigned by this parser, not the model


class PatchParseError(Exception):
    """Raised when model output cannot be salvaged into a valid schema."""


@dataclass
class PatchResult:
    status: PatchStatus
    cwe: Optional[str] = None
    vulnerable_function: Optional[str] = None
    explanation: Optional[str] = None
    patched_code: Optional[str] = None
    diff_summary: Optional[str] = None
    raw_model_output: str = ""

    def is_usable(self) -> bool:
        return self.status == PatchStatus.OK and bool(self.patched_code)


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_blob(raw_text: str) -> str:
    """
    Pull the JSON object out of raw model output, tolerating markdown
    fences or stray prose the model wasn't supposed to add.
    """
    fenced = _JSON_FENCE_RE.search(raw_text)
    if fenced:
        return fenced.group(1)

    bare = _BARE_OBJECT_RE.search(raw_text)
    if bare:
        return bare.group(0)

    raise PatchParseError("No JSON object found in model output")


def _sanitize_json_text(blob: str) -> str:
    """Fix common small-model JSON quirks before parsing."""
    # Smart quotes -> straight quotes
    blob = blob.replace("\u201c", '"').replace("\u201d", '"')
    blob = blob.replace("\u2018", "'").replace("\u2019", "'")
    # Trailing commas before } or ]
    blob = re.sub(r",\s*([}\]])", r"\1", blob)
    return blob


def _strip_code_fences(code: str) -> str:
    """Remove stray markdown fences a model sometimes nests inside patched_code."""
    code = code.strip()
    if code.startswith("```python"):
        code = code[len("```python"):]
    elif code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    return code.strip()


def _salvage_parse(raw_text: str) -> Optional[PatchResult]:
    """
    Fallback for when strict JSON parsing fails because the model used an
    unescaped delimiter for patched_code instead of proper JSON string
    escaping — most commonly Python triple-quotes (\"\"\"...\"\"\") instead
    of \\n-escaped text. Extracts patched_code via delimiter matching, and
    the remaining scalar fields via simple single-line regexes (a safe
    assumption since only patched_code tends to be genuinely multi-line).
    """
    def _field(name: str) -> Optional[str]:
        m = re.search(rf'"{name}"\s*:\s*"((?:[^"\\]|\\.)*)"', raw_text)
        return m.group(1) if m else None

    status = _field("status")
    if status not in ("ok", "insufficient_context"):
        return None

    patched_code = None
    # Triple-quoted style: "patched_code": """<code>"""
    m = re.search(r'"patched_code"\s*:\s*"""(.*?)"""', raw_text, re.DOTALL)
    if m:
        patched_code = m.group(1)
    else:
        # Markdown-fenced style nested inside a normal JSON string:
        # "patched_code": "```python<code>```"
        m = re.search(r'"patched_code"\s*:\s*"(```.*?```)"', raw_text, re.DOTALL)
        if m:
            patched_code = m.group(1).encode().decode("unicode_escape")

    if patched_code is None:
        return None

    return PatchResult(
        status=PatchStatus(status),
        cwe=_field("cwe"),
        vulnerable_function=_field("vulnerable_function"),
        explanation=_field("explanation"),
        patched_code=_strip_code_fences(patched_code),
        diff_summary=_field("diff_summary"),
        raw_model_output=raw_text,
    )


def parse_patch_response(raw_text: str) -> PatchResult:
    """
    Main entry point: turn raw model text into a validated PatchResult.
    Never raises for malformed output — returns a PARSE_ERROR status so
    the orchestrator can route to a retry rather than crashing the loop.
    """
    try:
        blob = _extract_json_blob(raw_text)
        blob = _sanitize_json_text(blob)
        data = json.loads(blob)
    except (PatchParseError, json.JSONDecodeError) as e:
        salvaged = _salvage_parse(raw_text)
        if salvaged is not None:
            return salvaged
        return PatchResult(
            status=PatchStatus.PARSE_ERROR,
            explanation=f"Failed to parse model output as JSON: {e}",
            raw_model_output=raw_text,
        )

    status_str = data.get("status", "")
    try:
        status = PatchStatus(status_str)
    except ValueError:
        status = PatchStatus.PARSE_ERROR

    result = PatchResult(
        status=status,
        cwe=data.get("cwe"),
        vulnerable_function=data.get("vulnerable_function"),
        explanation=data.get("explanation"),
        patched_code=_strip_code_fences(data["patched_code"]) if data.get("patched_code") else None,
        diff_summary=data.get("diff_summary"),
        raw_model_output=raw_text,
    )

    if result.status == PatchStatus.OK and not result.patched_code:
        result.status = PatchStatus.PARSE_ERROR
        result.explanation = "status=ok but patched_code missing"

    return result


def make_unified_diff(
    original_code: str,
    patched_code: str,
    file_path: str = "file.py",
) -> str:
    """
    Produce a clean unified diff for the PR/Review Agent, given the
    original vulnerable function and the model's patched_code.
    """
    original_lines = original_code.splitlines(keepends=True)
    patched_lines = patched_code.splitlines(keepends=True)
    diff = difflib.unified_diff(
        original_lines,
        patched_lines,
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        lineterm="",
    )
    return "\n".join(diff)


if __name__ == "__main__":
    # Quick smoke test with a deliberately messy model output
    sample_raw = '''
Here is the fix:
```json
{
  "status": "ok",
  "cwe": "CWE-89",
  "vulnerable_function": "get_user_by_name",
  "explanation": "Query used f-string interpolation of user input; replaced with a parameterized cursor.execute call.",
  "patched_code": "def get_user_by_name(name):\\n    cursor.execute(\\"SELECT * FROM users WHERE name = ?\\", (name,))\\n    return cursor.fetchone()",
  "diff_summary": "f-string query -> parameterized query",
}
```
'''
    result = parse_patch_response(sample_raw)
    print(result.status, result.is_usable())
    print(result.patched_code)
