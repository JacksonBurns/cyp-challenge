"""Scrape the live leaderboards from the CYP challenge HF space via Playwright (robust)."""
import json
import time

from playwright.sync_api import sync_playwright

URL = "https://huggingface.co/spaces/openadmet/cyp-challenge"
EXE = "/home/jackson/.cache/ms-playwright/chromium_headless_shell-1228/chrome-headless-shell-linux64/chrome-headless-shell"


def get_app_frame(page):
    for f in page.frames:
        if f != page.main_frame and "spaces" in (f.url or "") and f.url != URL:
            return f
    return None


results = {}
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, executable_path=EXE)
    page = browser.new_page(viewport={"width": 1800, "height": 2600})
    page.goto(URL, timeout=180_000, wait_until="domcontentloaded")

    frame = None
    deadline = time.time() + 300
    while time.time() < deadline:
        try:
            frame = get_app_frame(page)
            if frame:
                txt = frame.evaluate("() => document.body ? document.body.innerText.length : 0")
                if txt and txt > 500:
                    break
        except Exception:
            frame = None
        time.sleep(5)
    if not frame:
        print("NO FRAME; urls:", [f.url for f in page.frames])
        raise SystemExit(1)

    # wait for tab buttons to render
    try:
        frame.get_by_text("Live Leaderboard").first.wait_for(timeout=60_000)
    except Exception as e:
        print("tab not found:", e)
        body_txt = frame.evaluate("() => document.body.innerText.slice(0,800)")
        print("BODY:", body_txt)
        raise SystemExit(1)

    frame.get_by_text("Live Leaderboard").first.click()
    time.sleep(10)

    # collect tables repeatedly as they stream in
    collected = {}
    for attempt in range(6):
        try:
            tables = frame.eval_on_selector_all(
                "table",
                "els => els.map(t => Array.from(t.rows).map(r => Array.from(r.cells).map(c => c.innerText.trim())))",
            )
            for ti, t in enumerate(tables):
                key = tuple(t[0]) if t else ("empty",)
                collected[key] = t
        except Exception as e:
            print("attempt", attempt, "extract err:", type(e).__name__)
        time.sleep(8)

    with open("/tmp/leaderboards_raw.json", "w") as fh:
        json.dump(list(collected.values()), fh, indent=1)
    print("tables collected:", len(collected))
    for t in collected.values():
        print("=== TABLE", len(t), "rows ===")
        for row in t[:14]:
            print(" | ".join(str(c)[:24] for c in row))
    browser.close()
