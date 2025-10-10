# tests/smoke_claude.py
import os
import sys
import time

import requests

BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
TIMEOUT = 6


def get(path):
    r = requests.get(f"{BASE}{path}", timeout=TIMEOUT)
    r.raise_for_status()
    # Tjek at charset sættes korrekt af vores middleware
    ct = r.headers.get("content-type", "")
    assert (
        "application/json" in ct and "charset" in ct.lower()
    ), f"Content-Type uden charset: {ct}"
    return r.json()


def check_summary():
    js = get("/ai/summary?limit_days=7")
    txt = js.get("text", "").strip()
    assert len(txt) > 0, "AI-summary tom"
    return (
        "fallback"
        if txt.startswith("(AI slået fra)")
        else "timeout" if txt.startswith("(AI timeout/fejl)") else "ok"
    )


def check_explain():
    js = get("/ai/explain_trade?i=0&context_bars=60")
    txt = js.get("text", "").strip()
    assert len(txt) > 0, "Explain tom"
    return (
        "fallback"
        if txt.startswith("(AI slået fra)")
        else "timeout" if txt.startswith("(AI timeout/fejl)") else "ok"
    )


def main():
    ok = 0
    warn = 0
    print(f"[i] Base URL: {BASE}")
    # health
    h = get("/healthz")
    print("[✓] healthz:", h)
    # summary
    s = check_summary()
    print(f"[✓] /ai/summary  => {s}")
    # explain
    e = check_explain()
    print(f"[✓] /ai/explain  => {e}")

    # statuskode
    status = {"ok": 0, "fallback": 1, "timeout": 1}
    code = max(status[s], status[e])
    if code == 0:
        print("\nALL GOOD ✅  (live AI svarer uden fallback)")
    elif s == "fallback" or e == "fallback":
        print(
            "\nAI FALLBACK MODE ⚠️  (ANTHROPIC_API_KEY mangler eller AI_FORCE_FALLBACK=1)"
        )
    else:
        print("\nAI TIMEOUT/ERROR ⚠️  (øgn AI_HTTP_TIMEOUT eller tjek netværk/nøgle)")
    sys.exit(code)


if __name__ == "__main__":
    main()
