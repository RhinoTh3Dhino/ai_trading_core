# evaluation/run_claude_eval.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

from .claude_eval import ANTHROPIC_API_KEY  # for auto-dry-run hvis nøgle mangler
from .claude_eval import call_claude, validate_payload
from .loaders import load_recent_trades, summarize_for_prompt


def _read_prompt_template(root: Path) -> str:
    p = root / "prompts" / "claude_eval_v1.md"
    if p.exists():
        return p.read_text(encoding="utf-8")
    # Fallback mini-skabelon hvis filen ikke findes
    return (
        "You are a trading coach. Return ONLY a JSON object with keys:\n"
        "edge_score (0..1), opportunities [str], warnings [str], "
        "action {hold|scale_in|scale_out|exit}, confidence (0..1).\n"
        "Use the data below."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Claude eval against recent trades and save JSON.")
    ap.add_argument("--root", default=".", help="Projektrod (hvor data/prompts ligger).")
    ap.add_argument("--model", default="claude-3-haiku-20240307", help="Claude modelnavn.")
    ap.add_argument("--out", default="outputs/evals/last_eval.json", help="Output JSON-fil.")
    ap.add_argument("--n", type=int, default=20, help="Antal nyeste trades der hentes til prompt.")
    ap.add_argument("--dry-run", action="store_true", help="Tving mock (ingen eksternt API-kald).")
    args = ap.parse_args()

    root = Path(args.root).resolve()

    # 1) Byg brugerprompt: skabelon + (kort) CSV + META
    prompt_tmpl = _read_prompt_template(root)
    df = load_recent_trades(root, n=args.n)
    meta = summarize_for_prompt(df)

    try:
        csv_short = df.tail(min(len(df), 25)).to_csv(index=False)
    except Exception:
        csv_short = ""

    user_prompt = (
        f"{prompt_tmpl}\n\n"
        f"# DATA\n{csv_short}\n"
        f"# META\n{json.dumps(meta, ensure_ascii=False)}"
    )

    # 2) Kald modellen (eller mock)
    force_dry = bool(args.dry_run or not ANTHROPIC_API_KEY)
    result: Dict[str, Any] = call_claude(
        user_prompt,
        model=args.model,
        dry_run=force_dry,
    )

    # 3) Valider og skriv til fil
    valid = validate_payload(result)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] saved → {outp}")
    print(
        f"[valid={valid}] action={result.get('action')} "
        f"edge_score={result.get('edge_score')} "
        f"confidence={result.get('confidence')}"
    )
    # Returnér 0 selv hvis invalid, hvis du vil gøre det CI-strengt så returnér 2 på invalid:
    return 0 if valid else 2


if __name__ == "__main__":
    sys.exit(main())
