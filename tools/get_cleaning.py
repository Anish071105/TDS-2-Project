import os
import json
import re
import google.generativeai as genai  # google.generativeai, make sure it's imported where you use this
from typing import Dict, Any, Optional

def _extract_first_balanced_json(s: str) -> Optional[str]:
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
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
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
    return None

def cleaning(filepath: str, preview_text: str, model_name: str = "gemini-2.0-flash") -> Dict[str, Any]:
    """
    Ask Gemini whether the CSV at `filepath` needs cleaning based on the provided preview_text
    (first ~30 lines) and request code to clean it if necessary.

    Returns a dict:
      {
        "needs_cleaning": bool,
        "explanation": str,
        "code": str | None
      }

    The returned `code` (if present) is a full, runnable Python script as a string.
    IMPORTANT: The script MUST NOT contain import statements for pandas/re/numpy/json because
    run_generated_code() already provides `pd`, `re`, `np`, `json` in globals.
    The script must end with a print(json.dumps({...})) statement so the harness can capture the result.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY environment variable not set")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    prompt = f"""

    You are given the first 30 rows of a CSV file along with its header.

Your task:

Determine if the file needs cleaning To determine that chk for  things like
Inspect the file for any of the following issues:

Missing or empty values in critical columns.
Incorrect data types (numbers stored as strings, dates as text, etc.).
Extra whitespace or characters like "F$1,290,000,000" in money  Dont destroy dollar sign in such cases we need it But f is not required.
Invalid formats (e.g., numeric columns containing currency symbols, alphabetic characters, units, or formatting errors such as T$, F$, F8$, SM$).
Duplicates.
Wrong or inconsistent column names.
Inconsistent casing.
Placeholder timestamps or placeholders like N/A, -, etc.


If cleaning is needed, generate valid Python code (pandas-based) that:
Reads the full CSV from the provided filepath.
Cleans column names (strip whitespace, standardise naming if needed).
Converts columns to correct data types.
Strips whitespace from string fields.
Removes unwanted characters from numeric/text fields.
Saves the cleaned data back to the same filepath, overwriting the original.
If no cleaning is needed, output an empty string.

Strict output formatting rules:
Provide the code without any markdown formatting or code fences (no triple backticks, no language tags).
Output only the raw Python code text.
The output must be plain text — no code fences, no triple backticks, no syntax highlighting tags, no "python" label, and no additional commentary.

The first three lines must be exactly:
import pandas as pd
import numpy as np
import re

The output must define exactly one function named clean_csv(filepath) which performs the cleaning.
After defining the function, immediately call clean_csv("{filepath}") so the cleaning is executed automatically when the script runs.
Few more instructions 
1. Always include:
   import os
   import json

2. At the end of the code:
   - Save the cleaned CSV using:
       df.to_csv(filepath, index=False)
   - Output the cleaned file path in JSON format for harness usage:
    ```json
    {{"cleaned_path": "{filepath}"}}
3. Make sure the `cleaned_path` key is exactly spelled as shown.
4. Do not add extra print statements before or after the JSON output.
5. Ensure that the JSON is the last printed output.

CSV sample (first 30 rows):
{preview_text}
"""


    response = model.generate_content([{"text": prompt}])
    raw = response.text

    parsed = None
    # Try to extract a JSON object from model response
    try:
        candidate = _extract_first_balanced_json(raw)
        if candidate:
            parsed = json.loads(candidate)
    except Exception:
        parsed = None

    if not parsed:
        # If model didn't return clean JSON, be conservative: package the raw text as 'explanation' and
        # put the entire model response into 'code' (caller can inspect/modify).
        return {
            "needs_cleaning": True,
            "explanation": "Could not parse model JSON response; returning raw response as code for inspection.",
            "code": raw
        }

    # Normalize keys and ensure expected shape
    parsed.setdefault("needs_cleaning", False)
    parsed.setdefault("explanation", "")
    parsed.setdefault("code", None)

    return parsed
