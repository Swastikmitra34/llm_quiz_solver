from playwright.async_api import async_playwright
from typing import Tuple, List


async def fetch_page_html_and_text(url: str) -> Tuple[str, str]:
    """
    Returns (html_content, visible_text)
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, wait_until="networkidle")
        # Small wait for JS that sets innerHTML
        await page.wait_for_timeout(1500)

        html = await page.content()
        text = await page.inner_text("body")

        await browser.close()
        return html, text
