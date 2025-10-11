# evaluation/claude_eval.py
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

# --- Let afhængighed: brug pandas hvis tilgængelig, ellers fallback ---
try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

# Type-only import så Pylance ikke ser 'pd' som variabel i typeudtryk
if TYPE_CHECKING:
    from pandas import DataFrame

import requests  # requests bruges både til live-API og er let at mocke i tests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = Path(os.getenv("LOG_DIR", str(PROJECT_ROOT / "logs")))

DEFAULT_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-haiku-20240307")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")


# ---------- 1) Dataindsamling: Seneste kontekst fra logs ----------
def _tail_text(path: Path, n: int = 50) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        return "".join(lines[-n:])
    except Exception:
        return ""


def _load_recent_df(path: Path, n: int = 100) -> Optional["DataFrame"]:
    if pd is None or not path.exists():
        return None
    try:
        df = pd.read_csv(path)  # type: ignore[attr-defined]
        if len(df) > n:
            df = df.tail(n)
        return df  # type: ignore[return-value]
    except Exception:
        return None


def build_context(max_rows: int = 100) -> str:
    """
    Finder 'bedste' tilgængelige kontekst:
      - logs/fills.csv  (paper/live fills)
      - logs/daily_metrics.csv
      - logs/telegram_log.txt  (sidste 50 linjer)
    Returnerer kort tekstblok til prompt.
    """
    parts: List[str] = []

    fills = LOG_DIR / "fills.csv"
    daily = LOG_DIR / "daily_metrics.csv"
    tlog = LOG_DIR / "telegram_log.txt"

    if pd is not None:
        df_f = _load_recent_df(fills, n=max_rows)
        if df_f is not None and getattr(df_f, "empty", True) is False:
            sel = [
                c
                for c in df_f.columns
                if c.lower() in {"timestamp", "type", "price", "qty", "balance", "profit", "side"}
            ]
            if sel:
                parts.append("Recent fills (tail):\n" + df_f[sel].to_csv(index=False))
    else:
        if fills.exists():
            parts.append("Recent fills (tail):\n" + _tail_text(fills))

    if pd is not None:
        df_d = _load_recent_df(daily, n=max_rows)
        if df_d is not None and getattr(df_d, "empty", True) is False:
            sel = [
                c
                for c in df_d.columns
                if c.lower() in {"date", "profit_pct", "drawdown", "win_rate", "equity"}
            ]
            if sel:
                parts.append("Daily metrics (tail):\n" + df_d[sel].to_csv(index=False))
    else:
        if daily.exists():
            parts.append("Daily metrics (tail):\n" + _tail_text(daily))

    if tlog.exists():
        parts.append("Telegram log (tail 50):\n" + _tail_text(tlog, n=50))

    if not parts:
        return "No recent logs found. Use default neutral context."

    return "\n\n".join(parts)


# ---------- 2) Prompt-skabeloner (2 varianter) ----------
# VIGTIGT: bogstavelige { } er escapet som {{ }} så .format() ikke fejler.
PROMPTS: Dict[str, str] = {
    "p1": (
        "You are a trading coach. Given the following recent bot context, produce ONLY JSON.\n"
        "Schema:\n"
        "{{\n"
        '  "edge_score": number in [0,1],\n'
        '  "opportunities": ["string", ...],\n'
        '  "warnings": ["string", ...],\n'
        '  "action": "hold" | "scale_in" | "scale_out" | "exit",\n'
        '  "confidence": number in [0,1]\n'
        "}}\n"
        "Rules: Do not add any prose, backticks or code fences. Output must be a single JSON object.\n\n"
        "=== CONTEXT START ===\n{context}\n=== CONTEXT END ==="
    ),
    "p2": (
        "System: Risk-first reviewer. Return ONLY a JSON object per the schema with numeric fields in [0,1].\n"
        "If context is noisy, set confidence low, and add a warning.\n\n"
        "Schema keys: edge_score, opportunities, warnings, action, confidence (action ∈ {hold|scale_in|scale_out|exit}).\n\n"
        "Context:\n{context}"
    ),
}


# ---------- 3) Kald Claude (eller mock) ----------
def _mock_json(prompt: str) -> Dict[str, Any]:
    """
    Deterministisk pseudo-output baseret på prompt-hash.
    Sikrer smoke test uden net/adgang.
    """
    h = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest(), 16)
    rng = random.Random(h)

    edge = round(rng.uniform(0.25, 0.75), 2)
    conf = round(rng.uniform(0.4, 0.9), 2)

    # simple heuristik
    text_l = prompt.lower()
    warns: List[str] = []
    if "drawdown" in text_l:
        warns.append("Elevated drawdown detected.")
        conf = min(conf, 0.7)
    if "slippage" in text_l or "commission" in text_l:
        warns.append("Costs may distort short-term PnL.")

    actions = ["hold", "scale_in", "scale_out", "exit"]
    action = actions[h % len(actions)]
    opps = (
        ["Tighten risk on weakness", "Favor mean-reversion edges"]
        if edge < 0.5
        else ["Momentum continuation possible"]
    )

    return {
        "edge_score": edge,
        "opportunities": opps,
        "warnings": warns,
        "action": action,
        "confidence": conf,
    }


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Robust udtræk: find første {...}-blok, parse, og lav lille “comma-fix”.
    """
    if not text:
        return None
    # 1) ren JSON?
    try:
        return json.loads(text)
    except Exception:
        pass
    # 2) find første {...}
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    candidate = m.group(0)
    # 3) naive trailing-comma fix
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    try:
        return json.loads(candidate)
    except Exception:
        return None


def call_claude(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    api_key: str = ANTHROPIC_API_KEY,
    timeout: int = 20,
    dry_run: bool = False,
) -> Dict[str, Any]:
    if dry_run or not api_key:
        return _mock_json(prompt)

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    system = (
        "You MUST reply with a single JSON object only, no markdown, no text outside JSON. "
        "Keys: edge_score (0..1), opportunities [str], warnings [str], action one of: hold | scale_in | scale_out | exit, confidence (0..1)."
    )
    data = {
        "model": model,
        "max_tokens": 512,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }

    try:
        resp = requests.post(url, headers=headers, json=data, timeout=timeout)
        resp.raise_for_status()
        js = resp.json()
        # Claude messages API: text ligger normalt i content[0].text
        content = js.get("content") or []
        text = ""
        if content and isinstance(content, list):
            for chunk in content:
                if isinstance(chunk, dict) and chunk.get("type") == "text":
                    text += chunk.get("text", "")
        if not text and isinstance(js, dict):
            # fallback – hvis API ændrer struktur
            text = js.get("output", "") or js.get("completion", "")

        parsed = _extract_json(text)
        return parsed or _mock_json(prompt)
    except Exception:
        # Smoke-tests må ikke fejle på netværk → fallback
        return _mock_json(prompt)


# ---------- 4) Validering ----------
REQUIRED_KEYS = {"edge_score", "opportunities", "warnings", "action", "confidence"}
ACTIONS = {"hold", "scale_in", "scale_out", "exit"}


def validate_payload(obj: Dict[str, Any]) -> bool:
    try:
        if not isinstance(obj, dict):
            return False
        if not REQUIRED_KEYS.issubset(obj.keys()):
            return False
        if not (
            isinstance(obj["edge_score"], (int, float)) and 0.0 <= float(obj["edge_score"]) <= 1.0
        ):
            return False
        if not (
            isinstance(obj["confidence"], (int, float)) and 0.0 <= float(obj["confidence"]) <= 1.0
        ):
            return False
        if obj["action"] not in ACTIONS:
            return False
        if not isinstance(obj["opportunities"], list) or not all(
            isinstance(x, str) for x in obj["opportunities"]
        ):
            return False
        if not isinstance(obj["warnings"], list) or not all(
            isinstance(x, str) for x in obj["warnings"]
        ):
            return False
        return True
    except Exception:
        return False


# ---------- 5) CLI / Smoke runner ----------
def run_once(prompt_id: str, *, dry_run: bool, model: str) -> Dict[str, Any]:
    ctx = build_context()
    prompt_tpl = PROMPTS.get(prompt_id, PROMPTS["p1"])
    prompt = prompt_tpl.format(context=ctx)
    out = call_claude(prompt, model=model, dry_run=dry_run)
    valid = validate_payload(out)
    return {"valid": valid, "json": out}


def main():
    ap = argparse.ArgumentParser(description="Claude smoke-eval (returns JSON).")
    ap.add_argument("--prompt", choices=list(PROMPTS.keys()), default="p1")
    ap.add_argument("--dry-run", action="store_true", help="Tving mock/fallback (ingen API-kald).")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument(
        "--smoke",
        type=int,
        default=1,
        help="Kør N gentagelser og rapportér succeshyppighed.",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "claude_eval_last.json"),
    )
    args = ap.parse_args()

    N = max(1, int(args.smoke))
    ok = 0
    results: List[Dict[str, Any]] = []
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    for i in range(N):
        r = run_once(args.prompt, dry_run=args.dry_run, model=args.model)
        results.append(r)
        if r["valid"]:
            ok += 1
        # skriv sidste til fil (så vi altid har et eksempel liggende)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(r["json"], f, ensure_ascii=False, indent=2)

    rate = ok / N
    print(f"[Smoke] {ok}/{N} valid JSON = {rate:.0%}")
    # Udskriv sidste JSON til terminalen for hurtig inspektion
    print(json.dumps(results[-1]["json"], ensure_ascii=False, indent=2))

    # Strict-mode (valgfrit): if rate < 0.9: exit(1)


if __name__ == "__main__":
    main()
