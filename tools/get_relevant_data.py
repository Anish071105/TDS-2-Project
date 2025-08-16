import base64
import os
import tempfile
from typing import Optional, List
from bs4 import BeautifulSoup
import pandas as pd
from markdownify import markdownify as md

def get_relevant_data(
    file_name: str,
    css_selector: Optional[str] = None,
    session_dir: Optional[str] = None,
    uploaded_files: Optional[List[str]] = None,
) -> None:
    if uploaded_files is None:
        uploaded_files = []

    with open(file_name, encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")

    if css_selector:
        elements = soup.select(css_selector)
        if not elements:
            return

        el = elements[0]

        if el.name == "table" and session_dir:
            df = pd.read_html(str(el))[0]
            csv_path = os.path.join(session_dir, "tgak.csv")
            df.to_csv(csv_path, index=False)
            uploaded_files.append(csv_path)
            return

        elif el.name == "img":
            img_src = el.get("src")
            if not img_src:
                return
            if img_src.startswith("data:") and session_dir:
                header, encoded = img_src.split(",", 1)
                img_data = base64.b64decode(encoded)
                img_path = os.path.join(session_dir, "tgak.png")
                with open(img_path, "wb") as img_file:
                    img_file.write(img_data)
                uploaded_files.append(img_path)
            else:
                # For external URLs or local paths, just append the src string
                uploaded_files.append(img_src)
            return

        elif session_dir:
            text = el.get_text(separator="\n", strip=True)
            md_content = md(text)
            md_path = os.path.join(session_dir, "tgak.md")
            with open(md_path, "w", encoding="utf-8") as md_file:
                md_file.write(md_content)
            uploaded_files.append(md_path)
            return

    else:
        if session_dir:
            text = soup.get_text(separator="\n", strip=True)
            md_content = md(text)
            md_path = os.path.join(session_dir, "tgak.md")
            with open(md_path, "w", encoding="utf-8") as md_file:
                md_file.write(md_content)
            uploaded_files.append(md_path)
            return


