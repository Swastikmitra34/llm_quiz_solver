from playwright.async_api import async_playwright
from typing import Tuple


async def fetch_page_html_and_text(url: str) -> Tuple[str, str]:
    """
    Returns (html_content, visible_question_text)
    Robust Playwright extraction for IITM quiz pages (base64 + JS rendered).
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

        # Load and allow JS execution
        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(3000)

        # Ensure dynamic content appears
        try:
            await page.wait_for_selector("#result", timeout=10000)
            visible_text = await page.inner_text("#result")
        except Exception:
            # Fallback: extract full body if #result is missing
            visible_text = await page.inner_text("body")

        html = await page.content()
        await browser.close()

        return html, visible_text

