import requests
from pathlib import Path

def scrape_with_scraperapi(
    url: str,
    output_file: str = "page.html",
    api_key: str = "2f8c10574cac1edf7579912d630186fb"
) -> str:
    payload = {
        'api_key': api_key,
        'url': url,
        'render': 'true'  # JavaScript rendering
    }

    response = requests.get("http://api.scraperapi.com", params=payload)

    if response.status_code == 200:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(output_file).write_text(response.text, encoding="utf-8")
        return f"✅ Saved rendered HTML to {output_file}"
    else:
        return f"❌ Failed to fetch page: {response.status_code} - {response.text}"
