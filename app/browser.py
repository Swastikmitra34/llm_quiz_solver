from playwright.async_api import async_playwright
from typing import Tuple


async def fetch_page_html_and_text(url: str) -> Tuple[str, str]:
    """
    Returns (html_content, visible_text)
    Cloud-safe Playwright configuration for Render.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--single-process"
            ]
        )

        page = await browser.new_page()
        await page.goto(url, wait_until="networkidle", timeout=60000)

        # Allow JS to finish rendering dynamic content
        await page.wait_for_timeout(2000)

        html = await page.content()
        text = await page.inner_text("body")

        await browser.close()
        return html, text
