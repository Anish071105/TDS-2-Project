from playwright.async_api import async_playwright
import asyncio
from pathlib import Path

async def scrape_website(url: str, output_file: str = "tools.page.html"):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        content = await page.content()
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(output_file).write_text(content, encoding="utf-8")
        await browser.close()
    return f"Saved HTML to {output_file}"

# def main():
#    url = "https://en.wikipedia.org/wiki/Economy_of_India"
#    output_file = "economy_of_india.html"
#    result = asyncio.run(scrape_website(url, output_file))
#    print(result)
#
# if __name__ == "__main__":
#    main()
