import importlib
import math
import pytest

CANDIDATES = [
    # (modul, attr-navn) – test leder efter et af disse "kontrakt"-kald
    ("bot.engine", "compute_attribution_frame"),   # forventet: DataFrame m. kolonner
    ("bot.engine", "reconcile_attribution"),       # forventet: dict/obj m. summer
]

def _find_api():
    for mod, attr in CANDIDATES:
        try:
            m = importlib.import_module(mod)
            if hasattr(m, attr):
                return m, getattr(m, attr)
        except Exception:
            continue
    return None, None

@pytest.mark.contract
def test_pnl_attribution_sums_to_gross():
    mod, api = _find_api()
    if api is None:
        pytest.skip("Kontrakt-API til PnL-attribution ikke fundet – skipper test.")
    # Kald API'et i en minimal/standard konfiguration.
    # Forvent én af to former:
    # - DataFrame med kolonner: ['gap','directional','fees','slippage','funding','theta','residual','gross']
    # - Dict med samme nøgler.
    res = api()  # skal håndtere defaults i din implementering
    if hasattr(res, "to_dict"):
        data = res.to_dict(orient="list")
        # sum pr. kolonne
        sums = {k: float(sum(v)) for k, v in data.items()}
    elif isinstance(res, dict):
        sums = {k: float(v) for k, v in res.items()}
    else:
        pytest.skip("Ukendt returtype fra attribution-API – skipper test.")

    gross = sums.get("gross", None)
    assert gross is not None, "Kontrakt kræver en 'gross'-nøgle/kolonne"

    total_attr = (
        sums.get("gap", 0.0) + sums.get("directional", 0.0) +
        sums.get("fees", 0.0) + sums.get("slippage", 0.0) +
        sums.get("funding", 0.0) + sums.get("theta", 0.0) +
        sums.get("residual", 0.0)
    )
    assert math.isfinite(total_attr) and math.isfinite(gross)
    assert abs(total_attr - gross) < 1e-6, f"Sum(attr)={total_attr} != gross={gross}"
