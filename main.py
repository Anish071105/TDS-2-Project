# main.py
from fastapi import FastAPI, UploadFile, Request
import os
import tempfile
import uuid
import shutil
import json
import re
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai  # Make sure installed
from openai import OpenAI
import asyncio
from pathlib import Path
import mimetypes
from typing import List, Dict, Any, Optional
import time

# Import your tools (ensure these exist)
from tools.playwright_scrap import scrape_website
from tools.get_dom import get_dom_structure
from tools.get_relevant_data import get_relevant_data
import tools.read_csv
import tools.get_image
import tools.get_json
from  tools.pdf import read_pdf_preview        # new: expects read_pdf_preview(file_path) -> dict
from tools.sqllite import read_sqlite_preview     # new: expects read_sqlite_preview(db_path) -> dict
from tools.parot import read_parquet_preview   # new: expects read_parquet_preview(file_path) -> dict
from tools.answer_questions import run_generated_code

app = FastAPI()

### Helpers ##################################################################

def describe_file(filepath: str) -> str:
    _, ext = os.path.splitext(filepath.lower())
    ext = ext.strip(".")

    if ext in {"png", "jpg", "jpeg", "gif", "bmp", "webp"}:
        return "image"
    if ext == "json":
        return "json"
    if ext == "jsonl":
        return "jsonl"
    if ext == "csv":
        return "csv"
    if ext in {"md", "markdown"}:
        return "markdown"
    if ext in {"txt"}:
        return "text"
    if ext == "pdf":
        return "pdf"
    if ext in {"parquet", "parq", "pq"}:
        return "parquet"
    if ext in {"db", "sqlite", "sqlite3", "db3", "duckdb"}:
        return "db"
    mime_type, _ = mimetypes.guess_type(filepath)
    if mime_type:
        if mime_type.startswith("image/"):
            return "image"
        if mime_type == "application/json":
            return "json"

    return "unknown"

# put this helper above extract_structure_data (after your other helpers)
def generate_extractor_for_unknown(filepath: str, max_preview_lines: int = 5, max_preview_chars: int = 2000) -> Dict[str, Any]:
    """
    Ask Gemini to generate a small Python extractor for `filepath`, run it, parse JSON output,
    and return a dict of the shape {"success": bool, "content": str} or {"success": False, "error": str}.
    Uses run_generated_code() to execute the generated script.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return {"success": False, "error": "GOOGLE_API_KEY not set; cannot generate extractor."}

    # Configure (harmless if already configured)
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    # Prompt asks for a compact, safe extractor script that prints JSON only.
    prompt = f"""
You are asked to write a short, self-contained Python 3 script that reads the local file at path:
{filepath}

Requirements:
- The script MUST print exactly one JSON object to stdout and nothing else.
- JSON object must be one of:
  - {{ "success": true, "content": "<string preview>" }}
  - {{ "success": false, "error": "<error message>" }}
- The preview should be concise (<= {max_preview_chars} characters). Prefer:
  - For tabular data: first {max_preview_lines} rows as CSV text (header + rows).
  - For text-like data: first {max_preview_chars} characters (trimmed).
  - For PDFs: first {max_preview_lines} table rows if tables found, otherwise first {max_preview_chars} chars of text.
- Use only standard Python libraries, or these libraries if available: pandas, pdfplumber, pyarrow.
- DO NOT make external network calls.
- Do not print any debug or explanatory text; only print the final JSON object.
- Use `print(json.dumps(obj))` for final output.
- Keep the script short and robust to exceptions.

Write the full Python script now.
"""
    try:
        response = model.generate_content([{"text": prompt}])
        code = response.text or ""
        # strip code fences if present
        if code.startswith("```"):
            code = re.sub(r"^```(?:python)?\s*|\s*```$", "", code).strip()
    except Exception as e:
        return {"success": False, "error": f"Failed to request extractor from LLM: {e}"}

    # Save code for debugging (optional)
    try:
        dbg_path = os.path.join("/tmp", f"generated_extractor_{uuid.uuid4().hex[:8]}.py")
        with open(dbg_path, "w", encoding="utf-8") as fh:
            fh.write(code)
        print(f"[debug] Saved generated extractor to: {dbg_path}")
    except Exception:
        pass

    # Execute the generated code via your existing harness
    try:
        exec_output = run_generated_code(code)
    except Exception as e:
        return {"success": False, "error": f"Error executing generated extractor: {e}"}

    # Try to parse JSON from the output
    try:
        # first try to parse entire output
        parsed = json.loads(exec_output)
        if isinstance(parsed, dict) and "success" in parsed:
            return parsed
    except Exception:
        pass

    # fallback: extract the first balanced JSON substring
    try:
        candidate = extract_first_balanced_json(exec_output)
        if candidate:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and "success" in parsed:
                return parsed
    except Exception as e:
        # continue to return parse failure below
        print(f"[debug] Failed to parse JSON from extractor output: {e}; raw output start:\n{exec_output[:1000]}")

    # If nothing parseable, return failure with raw output snippet for debugging
    snippet = exec_output.replace("\n", " ")[:1000]
    return {"success": False, "error": f"Extractor did not emit valid JSON. RawOutput: {snippet}"}

def extract_json_or_base64(text: str) -> dict:
    """
    Try to extract the first balanced JSON block from text.
    If that fails, try to extract a base64 string inside.
    Returns a dict (parsed JSON or {"image_base64": "..."} or {"raw_output": text}).
    """

    # --- Step 1: Balanced JSON scan ---
    start = None
    stack = []
    for i, ch in enumerate(text):
        if ch in "{[":
            if not stack:
                start = i
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                continue
            opening = stack.pop()
            if (opening == "{" and ch != "}") or (opening == "[" and ch != "]"):
                # mismatched brackets → reset
                stack.clear()
                start = None
            elif not stack and start is not None:
                candidate = text[start:i+1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return {"raw_output": candidate}

    # --- Step 2: Regex fallback for base64 ---
    # Look for a long base64-like string
    b64_match = re.search(r'([A-Za-z0-9+/]{100,}={0,2})', text)
    if b64_match:
        return {"image_base64": b64_match.group(1)}

    # --- Step 3: Nothing matched ---
    return {"raw_output": text}

def extract_first_balanced_json(text: str) -> str | None:
    """
    Extracts the first balanced JSON object/array from text.
    Handles very long base64 strings without truncation.
    Returns the JSON string if found, else None.
    """
    start = None
    stack = []

    for i, ch in enumerate(text):
        if ch in "{[":
            if not stack:
                start = i
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                continue
            opening = stack.pop()
            if (opening == "{" and ch != "}") or (opening == "[" and ch != "]"):
                # mismatched brackets, reset
                stack.clear()
                start = None
            elif not stack and start is not None:
                return text[start:i+1]

    return None

from concurrent.futures import ThreadPoolExecutor, as_completed

def extract_structure_data(uploaded_files: List[str]) -> Dict[str, Any]:
            """
            Build a lightweight structure dict mapping filepath -> content.
            For CSV/PDF/DB/Parquet we call small preview handlers that return:
              {"success": True/False, "content": "<string preview>"}
            For images we call tools.get_image.get_image(filepath) (string).
            For json/jsonl call tools.get_json.inspect_json_file(filepath).
            For other text files we read raw text (but avoid huge files).
            Uses ThreadPoolExecutor to parallelize file processing for efficiency.
            """
            structure: Dict[str, Any] = {}
            question_names = {"question.txt", "questions.txt", "question", "questions"}

            def process_file(filepath: str) -> tuple[str, Any]:
                """
                Process a single file and return (filepath, content) tuple.
                Uses streaming for large files and robust error handling.
                """
                base_name = os.path.basename(filepath).lower()
                if base_name in question_names:
                    return (filepath, None)

                ftype = describe_file(filepath)
                try:
                    if ftype == "csv":
                        content = tools.read_csv.read_csv_preview(filepath)
                    elif ftype == "pdf":
                        content = tools.read_pdf.read_pdf_preview(filepath)
                    elif ftype == "parquet":
                        content = tools.read_parquet.read_parquet_preview(filepath)
                    elif ftype == "db":
                        content = tools.read_sqlite.read_sqlite_preview(filepath)
                    elif ftype == "image":
                        content = tools.get_image.get_image(filepath)
                    elif ftype in {"json", "jsonl"}:
                        content = tools.get_json.inspect_json_file(filepath)
                    elif ftype in {"markdown", "text"}:
                        # Stream read for large text files (up to 2000 chars)
                        content = ""
                        try:
                            with open(filepath, "r", encoding="utf-8") as f:
                                for chunk in iter(lambda: f.read(1024), ""):
                                    content += chunk
                                    if len(content) >= 2000:
                                        content = content[:2000]
                                        break
                        except Exception as e:
                            content = f"(Could not read file: {os.path.basename(filepath)}; {e})"
                    else:
                        content = generate_extractor_for_unknown(filepath)
                except Exception as e:
                    content = {"success": False, "error": f"Error reading {ftype} file: {str(e)}"}
                return (filepath, content)

            # Use ThreadPoolExecutor for parallel file processing
            with ThreadPoolExecutor(max_workers=8) as executor:
                future_to_file = {executor.submit(process_file, fp): fp for fp in uploaded_files}
                for future in as_completed(future_to_file):
                    fp, content = future.result()
                    if content is not None:
                        structure[fp] = content

            return structure
        
def make_session_dir():
    # Always use /tmp for ephemeral storage (writable on Render)
    base_dir = "/tmp"
    sess_id = str(uuid.uuid4())
    sess_dir = os.path.join(base_dir, "sessions", sess_id)
    os.makedirs(sess_dir, exist_ok=True)
    return sess_dir

def extract_json_from_markdown(text: str) -> str:
    """Extract JSON from markdown code blocks or raw JSON in text"""
    json_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    match = re.search(json_pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    json_pattern2 = r'\{.*\}'
    match = re.search(json_pattern2, text, re.DOTALL)
    if match:
        return match.group(0).strip()

    return text.strip()

def extract_keys_from_instructions(text: str) -> Optional[Dict[str, str]]:
    matches = re.findall(r"-\s*`(\w+)`\s*:\s*(.+)", text)
    if matches:
        result = {}
        for key, typ in matches:
            typ = typ.strip()
            if "base64" in typ or "PNG" in typ or "image" in typ.lower():
                result[key] = "string"
            elif "number" in typ.lower() or "integer" in typ.lower() or "float" in typ.lower():
                result[key] = "number"
            else:
                result[key] = "string"
        return result
    return None

def extract_from_questions(question_file_path: str) -> dict:
    """
    Robust extractor:
      - Sends the question file to Gemini (but JSON-escapes the file text)
      - Parses Gemini output using balanced-brace extraction
      - If parsing fails, falls back to simple regex heuristics:
         * find first http(s):// URL
         * extract numbered or bullet questions
      - Returns a dict with keys: url, questions, response_format, sources, other_info
    """
    with open(question_file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # quick local heuristics (used as fallback)
    def heuristic_extract_url(text: str) -> Optional[str]:
        m = re.search(r"https?://[^\s\)\]\}\'\"<>]+", text)
        return m.group(0) if m else None

    def heuristic_extract_questions(text: str) -> List[str]:
        lines = text.splitlines()
        qs = []

        # numbered lines like "1. Question..."
        for ln in lines:
            m = re.match(r"^\s*\d+\.\s*(.+\S)", ln)
            if m:
                qs.append(m.group(1).strip())

        # bullets like "- Question" or "* Question"
        if not qs:
            for ln in lines:
                m = re.match(r"^\s*[-\*\u2022]\s*(.+\S)", ln)
                if m:
                    qs.append(m.group(1).strip())

        # fallback: look for lines that end with '?'
        if not qs:
            for ln in lines:
                if ln.strip().endswith("?"):
                    qs.append(ln.strip())

        return qs
    
    
    def infer_array_types_from_questions(questions: List[str]) -> List[str]:
      number_keywords = [
        "how many", "total", "sum", "count", "median",
        "average", "correlation", "max", "min", "percentage", "number", "rank", "score"
      ]
      inferred = []
      for q in questions:
        q_lower = q.lower()
        if any(kw in q_lower for kw in number_keywords):
            inferred.append("number")
        elif any(x in q_lower for x in ["plot", "image", "base64", "chart"]):
            inferred.append("string")
        else:
            inferred.append("string")
      return inferred

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY environment variable not set")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Embed the file content as an escaped JSON string to avoid breaking the prompt
    content_escaped = json.dumps(content)

    prompt = f"""Parse the following question file which is provided as a JSON-escaped string.

You must return a JSON object with exactly the keys:
"url", "questions", "response_format", "sources", "other_info".

1) "url": a string or null
2) "questions": an array of strings (each question)
3)"response_format": 
Rules for repsonse format:
- If expected response is a JSON object or Json with keys , return a mapping with retianing the keys and expected field: {{"total_sales": "number", "top_region": "string", }}(For image consider "string").
- If response is an array, return types in order. ["string" , "number"]
- Use "number" for numeric answers and "string" for text or base64 images.
- If expected format is none of above think of something else.
4)"sources": array of source names or null
5) "other_info": any other short instructions or null

Return ONLY raw JSON (no surrounding text).

The content to parse is provided below as a JSON string variable:

content = {content_escaped}
"""

    response = model.generate_content([{"text": prompt}])
    raw = response.text

    parsed = None
    try:
        candidate = extract_first_balanced_json(raw)
        if candidate:
            parsed = json.loads(candidate)
    except Exception:
        parsed = None

    # If Gemini failed to produce usable JSON, fallback to heuristics
    if not parsed:
        print("Warning: Gemini produced non-parseable JSON. Falling back to heuristics.")
        fallback_url = heuristic_extract_url(content)
        fallback_questions = heuristic_extract_questions(content)
        parsed = {
            "url": fallback_url,
            "questions": fallback_questions,
            "response_format": None,
            "sources": None,
            "other_info": f"Gemini raw reply: {raw[:1000].replace(chr(10),' ')}"
        }

    # Normalize fields
    parsed.setdefault("url", None)
    parsed.setdefault("questions", [])
    parsed.setdefault("response_format", None)
    parsed.setdefault("sources", None)
    parsed.setdefault("other_info", None)

    # If parsed['url'] is null but we can find a URL via regex, inject it
    if not parsed.get("url"):
        heuristic_url = heuristic_extract_url(content)
        if heuristic_url:
            parsed["url"] = heuristic_url

    return parsed

from typing import Any, Optional, Union

def placeholder_for_format(question_text: str, response_format: Optional[Union[str, list, dict]]) -> Any:
    """
    Heuristic placeholder when we couldn't generate a real answer.
    Handles string, list, and dict-based response formats.
    """
    if not response_format:
        return "Good"

    # Case 1: dict schema
    if isinstance(response_format, dict):
        for key, spec in response_format.items():
            # try fuzzy match question → key
            if key.replace("_", " ").lower() in question_text.lower():
                return placeholder_for_type(spec)
        # fallback if no match
        return {k: placeholder_for_type(v) for k, v in response_format.items()}

    # Case 2: list of types (parallel to questions)
    if isinstance(response_format, list):
        # fallback: just use first type
        if response_format:
            return placeholder_for_type(response_format[0])
        return "Good"

    # Case 3: plain string
    if isinstance(response_format, str):
        return placeholder_for_type(response_format)

    return "Good"


def placeholder_for_type(spec: Union[str, dict]) -> Any:
    """Map type spec to dummy placeholder value."""
    t = spec.get("type", "string") if isinstance(spec, dict) else spec

    if not isinstance(t, str):
        return "Good"

    t = t.lower()
    if "number" in t or "int" in t or "integer" in t or "float" in t:
        return 1
    if "string" in t or "text" in t:
        if isinstance(spec, dict) and spec.get("format") == "base64":
            return ""  # placeholder for base64 string
        return "Good"
    if "json" in t:
        return []
    return "Good"



# Modify get_relevant_data call to pass session_dir and uploaded_files, no returns expected
async def extract_impinfo_from_source(session_dir: str, url: str, questions: list, uploaded_files: list) -> None:
    page_html_path = os.path.join(session_dir, "page.html")
    dom_txt_path = os.path.join(session_dir, "dom.txt")

    # 1. Scrape website
    await scrape_website(url, output_file=page_html_path)

    # 2. Extract DOM structure
    dom_structure = get_dom_structure(page_html_path)
    Path(dom_txt_path).write_text(dom_structure, encoding="utf-8")

    prompt = f"""
You are given:
- The DOM structure of a webpage (formatted tree text)
- The URL of the page
- A list of questions about the page

Your task is to identify the **best CSS selector** that extracts a relevant **table or image** element needed to answer the questions.

**Important:**  
- Only select CSS selectors targeting `<table>` or `<img>` elements.  
- Do **not** select plain text or other element types.  
- Return only the CSS selector string without any explanation.

DOM structure:
\"\"\"{dom_structure}\"\"\" 

URL:
\"\"\"{url}\"\"\" 

Questions:
\"\"\"{json.dumps(questions, indent=2)}\"\"\" 
"""

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY environment variable not set")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    
    response = model.generate_content([{"text": prompt}])
    css_selector = response.text.strip()

    if css_selector.startswith("```"):
        css_selector = re.sub(r"^```.*\n|```$", "", css_selector).strip()

    # 4. Call get_relevant_data with selector and page.html, pass session_dir and uploaded_files for side-effects
    get_relevant_data(page_html_path, css_selector, session_dir=session_dir, uploaded_files=uploaded_files)

    print(f"Uploaded files after extraction: {uploaded_files}")

    # -------------------------
    # If a file named tgak.csv was added by get_relevant_data, attempt automatic cleaning
    # -------------------------
    try:
        # lazy import so the function doesn't fail if tools.get_cleaning is missing
        try:
            from tools.get_cleaning import cleaning
        except Exception:
            cleaning = None

        # find any uploaded file whose basename is 'tgak.csv' (case-insensitive)
        for candidate_fp in list(uploaded_files):
            if os.path.basename(candidate_fp).lower() != "tgak.csv":
                continue

            print(f"Detected tgak.csv at: {candidate_fp} — preparing preview and calling cleaning()")

            # Read first ~30 lines (fallback to whole file if shorter)
            preview_lines = []
            try:
                with open(candidate_fp, "r", encoding="utf-8", errors="ignore") as fh:
                    for _ in range(30):
                        ln = fh.readline()
                        if not ln:
                            break
                        preview_lines.append(ln)
            except Exception as e:
                print(f"Could not read preview from {candidate_fp}: {e}")

            preview_text = "".join(preview_lines)

            if cleaning is None:
                print("tools.get_cleaning.cleaning not available — skipping automatic cleaning for tgak.csv.")
                continue

            # Call cleaning(filepath, preview_text) — it should return a dict
            try:
                cleaning_result = cleaning(candidate_fp, preview_text)
            except Exception as e:
                print(f"cleaning() raised exception for {candidate_fp}: {e}")
                cleaning_result = {"needs_cleaning": False, "explanation": f"cleaning() error: {e}", "code": None}

            print("cleaning() returned:", cleaning_result if isinstance(cleaning_result, dict) else type(cleaning_result))

            # If cleaning requested, execute returned code (must print JSON with cleaned_path)
            if isinstance(cleaning_result, dict) and cleaning_result.get("needs_cleaning"):
                code_str = cleaning_result.get("code")
                explanation = cleaning_result.get("explanation", "")
                print(f"Cleaning requested for {candidate_fp}: {explanation}")

                if not code_str:
                    print("cleaning() indicated cleaning needed but returned no code. Skipping.")
                    continue

                # Save code for inspection (useful for debugging)
                try:
                    code_file = os.path.join(session_dir, f"cleaning_code_{os.path.basename(candidate_fp)}.py")
                    with open(code_file, "w", encoding="utf-8") as fh:
                        fh.write(code_str)
                    print(f"Saved generated cleaning code to: {code_file}")
                except Exception as e:
                    print(f"Failed to save cleaning code to disk: {e}")

                # Execute cleaning code with the harness
                print("Executing cleaning code via run_generated_code()...")
                cleaning_output = run_generated_code(code_str)
                print("=== RAW cleaning output START ===")
                print(cleaning_output)
                print("=== RAW cleaning output END ===")

                # Try to parse cleaning output (expecting JSON like {"status":"cleaned","cleaned_path":"..."} )
                cleaned_json = None
                try:
                    cleaned_json = json.loads(cleaning_output)
                except Exception:
                    # fallback: extract first balanced JSON substring
                    try:
                        candidate = extract_first_balanced_json(cleaning_output)
                        if candidate:
                            cleaned_json = json.loads(candidate)
                    except Exception as e:
                        print(f"Failed to parse JSON from cleaning output: {e}")

                if cleaned_json and isinstance(cleaned_json, dict) and cleaned_json.get("cleaned_path"):
                    cleaned_path = cleaned_json["cleaned_path"]
                    # Make absolute if relative
                    if not os.path.isabs(cleaned_path):
                        cleaned_path = os.path.abspath(os.path.join(session_dir, cleaned_path))
                    if os.path.exists(cleaned_path):
                        if cleaned_path not in uploaded_files:
                            uploaded_files.append(cleaned_path)
                            print(f"Appended cleaned file to uploaded_files: {cleaned_path}")
                        else:
                            print(f"Cleaned file already present in uploaded_files: {cleaned_path}")
                    else:
                        print(f"Cleaning reported cleaned_path but file does not exist on disk: {cleaned_path}")
                else:
                    print("Cleaning did not produce a valid cleaned_path JSON. cleaned_json:", cleaned_json)
            else:
                print(f"No cleaning required for {candidate_fp}: {cleaning_result.get('explanation') if isinstance(cleaning_result, dict) else 'no dict returned'}")

    except Exception as e:
        print(f"Unexpected error during automatic tgak.csv cleaning step: {e}")


async def process_questions_in_batches(
    extracted: Dict[str, Any],
    structure: Dict[str, Any],
    session_dir: str,
    uploaded_files: List[str],
    batch_size: int = 2,
    max_attempts: int = 1,
    retry_backoff: float = 1.0
) -> Any:
    """
    Main loop:
    - For each batch of questions, ask GPT (via AIPIPE proxy) to generate runnable Python code.
    - Execute that code and capture the raw outputs.
    - Store raw outputs, strip base64 before sending to Gemini for formatting.
    - Restore base64 blobs afterward.
    """

    # Configure Gemini (for final formatting)
    gemini_api_key = os.getenv("GOOGLE_API_KEY")
    if not gemini_api_key:
        raise RuntimeError("GOOGLE_API_KEY environment variable not set")
    genai.configure(api_key=gemini_api_key)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")

    # Configure GPT (via AIPIPE proxy)
    openai_api_key = os.getenv("AIPIPE_TOKEN") or os.getenv("OPENAI_API_KEY")
    openai_base_url = os.getenv("OPENAI_BASE_URL", "https://aipipe.org/openai/v1")
    if not openai_api_key:
        raise RuntimeError("AIPIPE_TOKEN or OPENAI_API_KEY environment variable not set")

    client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)

    questions: List[str] = extracted.get("questions", [])
    other_info: str = extracted.get("other_info", "")
    response_format: Optional[str] = extracted.get("response_format")

    # ------------------ File previews ------------------
    example_structure = []
    count = 0
    for fp, content in structure.items():
        if count >= 5:
            break
        entry = {"filepath": fp}
        ftype = describe_file(fp)
        entry["type"] = ftype
        try:
            if isinstance(content, dict):
                if content.get("success") and "content" in content and isinstance(content["content"], str):
                    lines = content["content"].splitlines()
                    entry["preview"] = "\n".join(lines[:5])
                else:
                    entry["preview_error"] = content.get("error", "No preview available")
            elif isinstance(content, str):
                if ftype in {"csv", "parquet", "db"}:
                    lines = content.splitlines()
                    entry["preview"] = "\n".join(lines[:5])
                elif ftype == "image":
                    entry["image_desc"] = content[:200]
                elif ftype in {"json", "jsonl", "markdown", "text"}:
                    entry["content_summary"] = content.replace("\n", " ")[:200]
                else:
                    entry["content_summary"] = content.replace("\n", " ")[:200]
            else:
                entry["content_summary"] = "N/A"
        except Exception as e:
            entry["content_summary"] = f"Error preparing preview: {e}"

        example_structure.append(entry)
        count += 1

    example_structure_json = json.dumps(example_structure, indent=2, ensure_ascii=False)

    # ------------------ Base64 handling ------------------
    # More aggressive base64 detection for various formats
    BASE64_LONG_RE = re.compile(r'(?<![A-Za-z0-9+/=])([A-Za-z0-9+/]{1000,}={0,2})(?![A-Za-z0-9+/=])')
    BASE64_MEDIUM_RE = re.compile(r'(?<![A-Za-z0-9+/=])([A-Za-z0-9+/]{500,999}={0,2})(?![A-Za-z0-9+/=])')
    BASE64_SHORT_RE = re.compile(r'(?<![A-Za-z0-9+/=])([A-Za-z0-9+/]{100,499}={0,2})(?![A-Za-z0-9+/=])')
    BASE64_MAX_LEN = 100000

    def find_base64_blobs_in_text(text: str) -> List[str]:
        blobs = []
        # Check for long base64 strings first (likely images)
        for m in BASE64_LONG_RE.finditer(text):
            s = m.group(1)
            if len(s) % 4 == 0:
                blobs.append(s)
        
        # Then medium strings
        existing_blobs = set(blobs)
        for m in BASE64_MEDIUM_RE.finditer(text):
            s = m.group(1)
            if len(s) % 4 == 0 and s not in existing_blobs:
                blobs.append(s)
                
        # Finally short strings (but be more selective)
        for m in BASE64_SHORT_RE.finditer(text):
            s = m.group(1)
            if len(s) % 4 == 0 and s not in existing_blobs:
                # Additional validation for short strings
                try:
                    import base64
                    base64.b64decode(s + '==', validate=True)
                    blobs.append(s)
                except Exception:
                    continue
                    
        return blobs

    def strip_base64_in_text(text: str) -> str:
        # Replace in order of size (largest first to avoid partial matches)
        text = BASE64_LONG_RE.sub("__KEEP_BASE64__", text)
        text = BASE64_MEDIUM_RE.sub("__KEEP_BASE64__", text) 
        text = BASE64_SHORT_RE.sub("__KEEP_BASE64__", text)
        return text

    def strip_base64(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: strip_base64(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [strip_base64(v) for v in obj]
        elif isinstance(obj, str):
            # Check if entire string is base64
            if BASE64_LONG_RE.fullmatch(obj) or BASE64_MEDIUM_RE.fullmatch(obj) or BASE64_SHORT_RE.fullmatch(obj):
                return "__KEEP_BASE64__"
            return strip_base64_in_text(obj)
        return obj

    def extract_base64(obj: Any, blobs: Optional[List[str]] = None) -> List[str]:
        if blobs is None:
            blobs = []
        if isinstance(obj, dict):
            for v in obj.values():
                extract_base64(v, blobs)
        elif isinstance(obj, list):
            for v in obj:
                extract_base64(v, blobs)
        elif isinstance(obj, str):
            if BASE64_LONG_RE.fullmatch(obj) or BASE64_MEDIUM_RE.fullmatch(obj) or BASE64_SHORT_RE.fullmatch(obj):
                blobs.append(obj)
            else:
                blobs.extend(find_base64_blobs_in_text(obj))
        return blobs

    def restore_base64(obj: Any, blobs_iter) -> Any:
        if isinstance(obj, dict):
            return {k: restore_base64(v, blobs_iter) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [restore_base64(v, blobs_iter) for v in obj]
        elif isinstance(obj, str) and "__KEEP_BASE64__" in obj:
            if obj == "__KEEP_BASE64__":
                try:
                    return next(blobs_iter)
                except StopIteration:
                    return "MISSING_BASE64"
            parts = obj.split("__KEEP_BASE64__")
            rebuilt = []
            for idx, p in enumerate(parts):
                rebuilt.append(p)
                if idx < len(parts) - 1:
                    try:
                        rebuilt.append(next(blobs_iter))
                    except StopIteration:
                        rebuilt.append("MISSING_BASE64")
            return "".join(rebuilt)
        return obj

    # ------------------ Main batching loop ------------------
    collected_answers = []

    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}: {batch}")

        prompt = f"""
You are a Python programmer tasked with answering user questions using uploaded files.

Questions:
{json.dumps(batch, indent=2)}

You have access to the following uploaded files, with their absolute paths, types, and stored content/previews:

{example_structure_json}

Guidelines for generating code:
- The `content` field contains only a **small preview** (usually first 5 rows of CSV, or a text extract).
- Use `content` ONLY to understand the **schema or structure** of the file.
- When writing Python code, always load the **entire dataset from `filepath`** 
  (e.g., `pd.read_csv(filepath)`), not from `content`.
- Only if the file at `filepath` cannot be read, then fall back to using `content`.
- For CSV/Parquet, always prefer `pd.read_csv(filepath)` or `pd.read_parquet(filepath)`.
- For text/Markdown/PDFs, `content` can be used directly since it already contains extracted text.
- For images, `content` will contain OCR text; use that for analysis.
- ⚠️ Never use deprecated APIs like `pandas.Series.append`; instead, use modern alternatives like `pd.concat`.
- For base64 image generation, return ONLY the base64 string (no data:image/png;base64, prefix)
- Ensure images are under 100kB by using appropriate DPI and figure size: figsize=(8, 6), dpi=80

Always output ONLY valid Python code.
- Do NOT explain.
- Do NOT answer directly.
- Do NOT include numbered lists.
- The generated python code must print **only the answers** (JSON, list, or plain text) — no debug, no explanations.

Example for image generation:

```python
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import json

# Load data
df = pd.read_csv(filepath)

# Create chart with size constraints
fig, ax = plt.subplots(figsize=(8, 6))
# ... your plotting code ...
ax.set_title('Chart Title')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')

# Save as base64 (optimized size)
buffer = BytesIO()
plt.savefig(buffer, format='png', bbox_inches='tight', dpi=80, facecolor='white')
buffer.seek(0)
img_base64 = base64.b64encode(buffer.getvalue()).decode()
plt.close()

# Print final result
result = {{
    "chart_field": img_base64  # Just the base64 string, no prefix
}}
print(json.dumps(result))
```

Write the full runnable Python code below.
"""

        attempt = 0
        success_for_batch = False
        last_output = None

        while attempt < max_attempts and not success_for_batch:
            attempt += 1
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=2000,
                )
                code = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"  OpenAI API error: {e}")
                time.sleep(retry_backoff)
                continue

            if code.startswith("```"):
                code = re.sub(r"^```.*\n|```$", "", code).strip()

            print("  Executing generated code...")
            output = run_generated_code(code)
            last_output = output

            if output.startswith("❌ Error while executing"):
                if attempt < max_attempts:
                    time.sleep(retry_backoff)
                    continue
                else:
                    break

            collected_answers.append({"batch": batch, "raw_output": output})
            success_for_batch = True

        if not success_for_batch:
            placeholders = [placeholder_for_format(q, response_format) for q in batch]
            collected_answers.append({"batch": batch, "raw_output": json.dumps(placeholders)})

    # ------------------ Assemble final ------------------
    stripped_collected = []
    all_blobs: List[str] = []

    print(f"Processing {len(collected_answers)} batch results...")

    for c in collected_answers:
        raw = c.get("raw_output", "")
        try:
            parsed_raw = None
            cand = extract_first_balanced_json(raw)
            if cand:
                parsed_raw = json.loads(cand)
            else:
                parsed_raw = json.loads(raw)
        except Exception:
            parsed_raw = None

        if parsed_raw is not None:
            blobs = extract_base64(parsed_raw)
            print(f"  Found {len(blobs)} base64 blobs in parsed JSON")
            
            # Filter and validate blobs
            valid_blobs = []
            for blob in blobs:
                if len(blob) <= BASE64_MAX_LEN:
                    valid_blobs.append(blob)
                    print(f"  ✓ Valid base64 blob: {len(blob)} chars")
                else:
                    valid_blobs.append("MISSING_BASE64")
                    print(f"  ✗ Base64 blob too large: {len(blob)} chars, using placeholder")
                    
            all_blobs.extend(valid_blobs)
            stripped = strip_base64(parsed_raw)
        else:
            # Handle raw text output
            blobs = find_base64_blobs_in_text(raw)
            print(f"  Found {len(blobs)} base64 blobs in raw text")
            
            valid_blobs = []
            for blob in blobs:
                if len(blob) <= BASE64_MAX_LEN:
                    valid_blobs.append(blob)
                    print(f"  ✓ Valid base64 blob: {len(blob)} chars")
                else:
                    valid_blobs.append("MISSING_BASE64")
                    print(f"  ✗ Base64 blob too large: {len(blob)} chars, using placeholder")
                    
            all_blobs.extend(valid_blobs)
            stripped = strip_base64_in_text(raw)

        stripped_collected.append({"batch": c.get("batch", []), "raw_output": stripped})

    print(f"Total base64 blobs collected: {len(all_blobs)}")

    # Create a simplified structure for Gemini
    final_prompt = f"""
You are a precise JSON assembler. Your task is to combine batch results into a single JSON response.

Questions to answer:
{json.dumps(questions, indent=2)}

Raw batch outputs (base64 data replaced with __KEEP_BASE64__ to save space):
{json.dumps(stripped_collected, indent=2)}

Additional context:
{other_info}

Expected output format:
{json.dumps(response_format, indent=2) if response_format else "A JSON object with answers to all questions"}

IMPORTANT RULES:
1. Output ONLY valid JSON, no markdown formatting
2. Combine all batch results into a single response
3. For __KEEP_BASE64__ placeholders, they will be replaced with actual base64 data after you respond
4. Use these defaults for missing values:
   - Missing numbers: 0
   - Missing strings: "UNKNOWN" 
   - Missing base64: "MISSING_BASE64"
5. Ensure all required fields are present in the final JSON

Respond with the JSON structure only:
"""

    print("Requesting final formatting from Gemini...")
    try:
        final_response = gemini_model.generate_content([{"text": final_prompt}])
        final_text = final_response.text.strip()
        
        # Clean up any markdown formatting
        final_text_extracted = extract_json_from_markdown(final_text)
        
        try:
            parsed_final = json.loads(final_text_extracted)
            print("✓ Successfully parsed Gemini response as JSON")
        except Exception as e:
            print(f"Failed to parse Gemini response as JSON: {e}")
            print(f"Gemini response: {final_text_extracted[:500]}...")
            # Try to construct a basic response
            parsed_final = {"error": "Failed to format response", "raw_data": stripped_collected}
    except Exception as e:
        print(f"Gemini API error: {e}")
        # Fallback: try to assemble the JSON ourselves
        parsed_final = {"error": f"Gemini error: {str(e)}", "raw_data": stripped_collected}

    # Restore base64 blobs
    def safe_blob_iter(blobs: List[str]):
        for b in blobs:
            yield b

    print(f"Restoring {len(all_blobs)} base64 blobs...")
    if isinstance(parsed_final, (dict, list)) and all_blobs:
        try:
            restored = restore_base64(parsed_final, iter(safe_blob_iter(all_blobs)))
            print("✓ Successfully restored base64 data")
            return restored
        except Exception as e:
            print(f"Error restoring base64: {e}")
            return parsed_final
    else:
        return parsed_final

### FastAPI endpoint #########################################################
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],                # change to your frontend origin(s) in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/")
async def handle_files(request: Request):
    session_dir = make_session_dir()
    question_file_path = None
    uploaded_files: List[str] = []

    valid_names = {"question.txt", "questions.txt", "question", "questions"}

    # Parse multipart/form-data form
    form = await request.form()
    for field_name, value in form.multi_items():
        # file field
        if hasattr(value, "filename") and hasattr(value, "read"):
            # sanitize filename if you want (this is basic)
            filename = os.path.basename(value.filename)
            file_path = os.path.join(session_dir, filename)
            with open(file_path, "wb") as f:
                f.write(await value.read())
            uploaded_files.append(file_path)

            clean_field_name = field_name.lower()
            clean_filename = os.path.basename(value.filename).lower() if value.filename else ""
            if clean_filename in valid_names or clean_field_name in valid_names:
                question_file_path = os.path.abspath(file_path)
        else:
            # ignore other non-file form fields for now
            pass

    print(f"Session directory: {session_dir}")
    print(f"Question file path: {question_file_path}")
    print(f"Uploaded files initially: {uploaded_files}")

    if not question_file_path:
        print("No question file found.")
        return {"error": "No question file found"}

    # 1) Extract metadata from questions file
    extracted = extract_from_questions(question_file_path)
    print("Extracted metadata from questions file:")
    print(json.dumps(extracted, indent=2))

    # 2) If URL present, run extraction flow (scrape & get relevant table/image)
    if extracted.get("url"):
        await extract_impinfo_from_source(
            session_dir,
            extracted["url"],
            extracted.get("questions", []),
            uploaded_files,
        )
    else:
        print("No URL provided in extracted questions.")

    # 3) Build lightweight structure (do NOT include large HTML files)
    structure = extract_structure_data(uploaded_files)
    print("Structure (uploaded file contents summary):")
    print(json.dumps(structure, indent=2, ensure_ascii=False))

    # 4) Process questions in batches, execute generated code, and collect formatted final answers
    final_answers = await process_questions_in_batches(
        extracted=extracted,
        structure=structure,
        session_dir=session_dir,
        uploaded_files=uploaded_files,
        batch_size=2,
        max_attempts=2,
        retry_backoff=1.0
    )

    print("Final formatted answers to return:")
    print(json.dumps(final_answers, indent=2, ensure_ascii=False))

    # Optional: cleanup session dir after processing
    shutil.rmtree(session_dir, ignore_errors=True)

    return final_answers


# simple root that accepts GET and HEAD (no POST)
@app.api_route("/", methods=["GET", "HEAD"])
async def root(request: Request):
    # For HEAD requests, FastAPI will return an empty body automatically.
    return {"message": "TDS Virtual TA API is running", "endpoints": ["/api/", "/health"]}


# health endpoint supports GET and HEAD
@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    # Return a small JSON for GET; FastAPI will auto-handle HEAD
    return {"status": "ok"}
