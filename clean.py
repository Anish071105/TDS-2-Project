# test_clean.py
import os
import json
from pathlib import Path

# adjust this import path if your tools package is elsewhere
from tools.get_cleaning import cleaning
from tools.answer_questions import run_generated_code

# helper: extract first balanced JSON substring (object or array)
def extract_first_balanced_json(s: str):
    if not s:
        return None
    start = None
    for i, ch in enumerate(s):
        if ch in "{[":
            start = i
            break
    if start is None:
        return None
    stack = []
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch in "{[":
            stack.append(ch)
            continue
        if ch in "}]":
            if not stack:
                return None
            opener = stack.pop()
            if (opener == "{" and ch != "}") or (opener == "[" and ch != "]"):
                return None
            if not stack:
                return s[start : i + 1]
    return None

def read_preview(filepath: str, max_lines: int = 30) -> str:
    lines = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
            for _ in range(max_lines):
                ln = fh.readline()
                if not ln:
                    break
                lines.append(ln)
    except Exception as e:
        print(f"ERROR reading preview from {filepath}: {e}")
    return "".join(lines)

def main():
    # absolute path to the csv you asked for    
    filepath = "/tmp/sessions/7ab84b96-14cd-445a-ad89-e58fc8716b32/tgak.csv"
    
    if not os.path.exists(filepath):
        print("File not found:", filepath)
        return

    # ensure GOOGLE_API_KEY is set
    if not os.getenv("GOOGLE_API_KEY"):
        print("WARNING: GOOGLE_API_KEY is not set in env. Set it before running so cleaning() can call Gemini.")
        # you can still call cleaning() but it will raise if code requires the key

    preview_text = read_preview(filepath, max_lines=30)
    print("==== Preview (first ~30 lines) ====")
    print(preview_text)
    print("==== End preview ====")

    print("Calling cleaning(...)")
    try:
        result = cleaning(filepath, preview_text)
    except Exception as e:
        print("cleaning() raised an exception:", repr(e))
        return

    print("\n=== cleaning() returned ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # if cleaning requested, run code
    if isinstance(result, dict) and result.get("needs_cleaning"):
        code_str = result.get("code")
        explanation = result.get("explanation", "")
        print("\nCleaning requested:", explanation)

        if not code_str:
            print("No code returned by cleaning(), nothing to execute.")
            return

        # save code for inspection
        session_dir = str(Path(filepath).parent)
        code_filename = os.path.join(session_dir, f"cleaning_code_{Path(filepath).name}.py")
        try:
            with open(code_filename, "w", encoding="utf-8") as fh:
                fh.write(code_str)
            print(f"Saved cleaning code to: {code_filename}")
        except Exception as e:
            print(f"Failed to save cleaning code to {code_filename}: {e}")

        # execute using harness
        print("\nExecuting cleaning code via run_generated_code() (harness)...")
        cleaning_output = run_generated_code(code_str)
        print("\n=== RAW cleaning output ===")
        print(cleaning_output)
        print("=== end raw output ===\n")

        # try parsing JSON directly
        parsed = None
        try:
            parsed = json.loads(cleaning_output)
            print("Parsed JSON from output:", parsed)
        except Exception:
            candidate = extract_first_balanced_json(cleaning_output)
            if candidate:
                try:
                    parsed = json.loads(candidate)
                    print("Parsed JSON from extracted candidate:", parsed)
                except Exception as e:
                    print("Failed to json.loads(candidate):", e)
                    print("Candidate was:", candidate)
            else:
                print("No JSON found in cleaning output.")

        # If cleaned_path present, verify file exists
        if parsed and isinstance(parsed, dict) and parsed.get("cleaned_path"):
            cp = parsed["cleaned_path"]
            if not os.path.isabs(cp):
                cp = os.path.abspath(os.path.join(session_dir, cp))
            print("cleaned_path reported:", cp)
            print("exists:", os.path.exists(cp))
    else:
        print("No cleaning required (or unexpected return).")

if __name__ == "__main__":
    main()
