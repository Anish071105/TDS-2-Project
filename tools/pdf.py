import os
import pdfplumber
import pandas as pd
import re

def read_pdf_preview(file_path: str, max_headings: int = 5, nrows: int = 5) -> dict:
    """
    Reads a PDF file and returns:
    - Total pages
    - First `max_headings` detected headings
    - First `nrows` rows of each table found

    Returns:
        {"success": bool, "content": str or error message}
    """
    if not os.path.exists(file_path):
        return {"success": False, "error": f"File not found: {file_path}"}

    try:
        with pdfplumber.open(file_path) as pdf:
            num_pages = len(pdf.pages)
            headings = []
            tables_preview = []

            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""

                # Detect headings
                for line in text.split("\n"):
                    if re.match(r"^[A-Z\s]{5,}$", line.strip()) or re.match(r"^[A-Z][a-z]+(\s[A-Z][a-z]+){2,}$", line.strip()):
                        if len(headings) < max_headings:
                            headings.append(f"Page {page_num}: {line.strip()}")

                # Extract tables
                for idx, table in enumerate(page.extract_tables(), start=1):
                    df = pd.DataFrame(table[1:], columns=table[0])
                    df_preview = df.head(nrows)
                    tables_preview.append(f"--- Page {page_num} - Table {idx} ---\n{df_preview.to_csv(index=False)}")

        # Combine into a single content string
        content = [
            f"📄 File: {os.path.basename(file_path)}",
            f"📑 Total Pages: {num_pages}",
            "\n🔹 Headings Found (preview):",
            "\n".join(headings) if headings else "(No headings detected)",
            "\n📊 Tables Extracted (preview):",
            "\n".join(tables_preview) if tables_preview else "(No tables found)"
        ]

        return {"success": True, "content": "\n".join(content)}

    except Exception as e:
        return {"success": False, "error": str(e)}
