# tools/breakdown.py
import os
import re
import json
import uuid
from typing import Dict, Any, List, Optional

import google.generativeai as genai
from pathlib import Path

# Import your existing harness and helper(s)
from tools.answer_questions import run_generated_code
from main import extract_first_balanced_json  # if extract_first_balanced_json lives in main.py; else duplicate here
# If extract_first_balanced_json is not available via import, copy a small utility here.

# Basic sanitizer for generated cleaning code
_BAD_CODE_PATTERNS = [
    r"\bos\.system\b", r"\bsubprocess\b", r"\bsocket\b", r"\brequests\b", r"\burllib\b",
    r"\bftp\b", r"\bssh\b", r"\bparamiko\b", r"\bopen\([^,]*http", r"curl\s", r"wget\s",
    r"\beval\(", r"\bexec\(", r"popen\(", r"rm -rf", r"sh -c", r"chmod", r"chown"
]

def _looks_dangerous(code: str) -> bool:
    low = code.lower()
    for pat in _BAD_CODE_PATTERNS:
        if re.search(pat, low):
            return True
    return False

def _strip_code_fence(s: str) -> str:
    if not s:
        return s
    s = s.strip()
    # remove triple-backtick fences optionally with language
    s = re.sub(r"^```(?:python)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()

def analyze_other_info_payload(
    payload: Dict[str, Any],
    session_dir: str,
    uploaded_files: List[str],
    model_name: str = "gemini-1.5-flash",
    max_simplified_chars: int = 2000
) -> Dict[str, Any]:
    """
    Inspect the `payload` (expected keys: url, questions, response_format, sources, other_info).
    - If cleaning is mentioned, ask the LLM to produce cleaning code (that prints JSON with cleaned_path),
      run it, and attach any cleaned file(s) to uploaded_files and structure previews.
    - If schema/steps are mentioned, ask the LLM to simplify them crisply and write a .txt into session_dir.
    Returns a summary dict with actions taken and any new files added.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return {"ok": False, "error": "GOOGLE_API_KEY not set"}

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    # Build the inspection prompt: ask the model to look for cleaning/schema hints
    prompt = f"""
You are an assistant that inspects an LLM assistant's raw reply and decides whether it:
 - explicitly requests or suggests data cleaning (e.g., "remove nulls", "split column", "parse dates", "fix delimiter", "remove header rows", "drop rows with ..."),
 - or provides schema / transformation steps (for example: "rename columns X->Y", "cast column A to int", "group by B and sum C"), 
 - or neither.

Input (JSON object):
{json.dumps(payload, indent=2)}

Return ONLY a single JSON object (no explanation) with these keys:
 - "action": one of "none", "cleaning", "schema", "both", or "ask_user"
 - "reason": short text explaining why you chose this action
 - "cleaning_requested": true/false
 - "cleaning_instructions": short text summary if cleaning requested, or null
 - "cleaning_code": If cleaning is requested and you can, generate a short Python script (as a string) that when run locally
    will read the target file and print a JSON object like:
      {{ "status": "cleaned", "cleaned_path": "relative/or/absolute/path/to/cleaned_file.csv" }}
    If you cannot produce safe code, return null here.
 - "target_filenames": array of filenames (base names) referenced in the instructions that should be cleaned (or [] if none)
 - "schema_steps": array of short steps if schema/transform instructions were found, or null
 - "simplified_steps": a short plain-text summary suitable for saving to a .txt file (<= {max_simplified_chars} chars) or null

Constraints:
 - The cleaning_code must be self-contained, use only standard Python libraries or pandas, and must NOT make network calls.
 - The assistant must return valid JSON only.
"""
    resp = model.generate_content([{"text": prompt}])
    raw = resp.text

    # parse the json response (balanced JSON extraction)
    candidate = extract_first_balanced_json(raw)
    if not candidate:
        # last-resort: wrap the raw in a small JSON skeleton to avoid breaking the flow
        return {"ok": False, "error": "Could not parse LLM inspection response", "raw": raw[:1000]}

    try:
        parsed = json.loads(candidate)
    except Exception as e:
        return {"ok": False, "error": f"JSON parse failed: {e}", "raw": candidate[:2000]}

    summary: Dict[str, Any] = {
        "ok": True,
        "parsed_inspection": parsed,
        "files_added": [],
        "notes": []
    }

    action = parsed.get("action", "none")
    cleaning_requested = bool(parsed.get("cleaning_requested"))
    cleaning_code = parsed.get("cleaning_code")
    target_filenames = parsed.get("target_filenames") or []
    simplified_steps = parsed.get("simplified_steps")

    # If schema steps present, write to a text file in session_dir
    if parsed.get("schema_steps") or simplified_steps:
        txt_body = simplified_steps or json.dumps(parsed.get("schema_steps"), indent=2)
        safe_name = f"schema_steps_{uuid.uuid4().hex[:8]}.txt"
        txt_path = os.path.join(session_dir, safe_name)
        try:
            with open(txt_path, "w", encoding="utf-8") as fh:
                fh.write(txt_body[:max_simplified_chars])
            summary["files_added"].append(txt_path)
            summary["notes"].append(f"Wrote schema instructions to {txt_path}")
        except Exception as e:
            summary["notes"].append(f"Failed to write schema instructions: {e}")

    # If cleaning requested and code present, sanitize & run it
    if cleaning_requested and cleaning_code:
        code_clean = _strip_code_fence(cleaning_code)
        # quick sanity checks
        if _looks_dangerous(code_clean):
            summary["notes"].append("Cleaning code flagged as potentially dangerous — skipping execution.")
            summary["cleaning_executed"] = False
            summary["cleaning_error"] = "code flagged by sanitizer"
        else:
            # Save code to a file for auditing
            code_filename = os.path.join(session_dir, f"cleaning_{uuid.uuid4().hex[:8]}.py")
            try:
                with open(code_filename, "w", encoding="utf-8") as fh:
                    fh.write(code_clean)
                summary["notes"].append(f"Saved cleaning code to {code_filename}")
            except Exception as e:
                summary["notes"].append(f"Failed to save cleaning code: {e}")
                code_filename = None

            # Execute via your harness
            try:
                run_out = run_generated_code(code_clean)
                summary["notes"].append("Ran cleaning code; raw output captured.")
                summary["cleaning_executed"] = True
                summary["cleaning_raw_output"] = run_out[:2000]
                # Try to parse JSON output: prefer direct parse, else balanced JSON substring
                cleaned_json = None
                try:
                    cleaned_json = json.loads(run_out)
                except Exception:
                    try:
                        cand = extract_first_balanced_json(run_out)
                        if cand:
                            cleaned_json = json.loads(cand)
                    except Exception:
                        cleaned_json = None

                if cleaned_json and isinstance(cleaned_json, dict) and cleaned_json.get("cleaned_path"):
                    cleaned_path = cleaned_json["cleaned_path"]
                    # make absolute relative to session_dir if needed
                    if not os.path.isabs(cleaned_path):
                        cleaned_path = os.path.abspath(os.path.join(session_dir, cleaned_path))
                    if os.path.exists(cleaned_path):
                        # add to uploaded_files and report
                        if cleaned_path not in uploaded_files:
                            uploaded_files.append(cleaned_path)
                        # Optionally, produce a lightweight preview by returning a marker to caller
                        summary["files_added"].append(cleaned_path)
                        summary["notes"].append(f"Cleaning reported cleaned_path: {cleaned_path}")
                    else:
                        summary["notes"].append(f"Cleaning reported path does not exist on disk: {cleaned_path}")
                        summary["cleaning_error"] = "cleaned_path missing on disk"
                else:
                    summary["notes"].append("Could not parse cleaned_path from cleaning output.")
                    summary["cleaning_error"] = "no cleaned_path found"
            except Exception as e:
                summary["cleaning_executed"] = False
                summary["cleaning_error"] = str(e)

    # If cleaning was requested but no code was generated, note that
    if cleaning_requested and not cleaning_code:
        summary["notes"].append("Cleaning was requested by inspection but no cleaning_code provided by LLM.")

    return summary
