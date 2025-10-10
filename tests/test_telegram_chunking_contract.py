import importlib

import pytest

CANDIDATES = [
    ("utils.telegram_utils", "chunk_message"),
    ("utils.telegram_utils", "chunk_markdown"),
]


def _find_api():
    for mod, attr in CANDIDATES:
        try:
            m = importlib.import_module(mod)
            if hasattr(m, attr):
                return getattr(m, attr)
        except Exception:
            continue
    return None


@pytest.mark.contract
def test_telegram_chunking_contract():
    chunker = _find_api()
    if chunker is None:
        pytest.skip("Chunking-API ikke fundet – skipper test.")
    text = "A" * 12000  # 12k tegn → skal chunks ≤4096
    chunks = list(chunker(text)) if callable(chunker) else []
    assert chunks, "Chunker skal returnere mindst 1 chunk"
    assert all(
        len(c) <= 4096 for c in chunks
    ), "En eller flere chunks overstiger 4096 tegn"
    assert sum(len(c) for c in chunks) == len(
        text
    ), "Samlet længde af chunks ≠ original længde"
