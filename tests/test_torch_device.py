# tests/test_torch_device.py
import sys
from pathlib import Path

import pytest

# Sørg for at projektroden er på sys.path (når testen køres direkte/lokalt)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_torch_import_and_device_smoke(capsys=None):
    """
    Røgsignal-test:
    - Skipper hvis torch ikke er installeret (så "lette" CI-jobs kan køre uden PyTorch).
    - Fejler ikke hvis CUDA ikke er tilgængelig (CPU er OK).
    """
    torch = pytest.importorskip("torch", reason="torch not installed; skipping torch device test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Udskriv lidt info (hjælpsomt i logs), men uden at kræve CUDA
    print("Fundet device:", device)
    if torch.cuda.is_available():
        try:
            print("CUDA device navn:", torch.cuda.get_device_name(0))
        except Exception as e:  # beskyttelse mod miljø/drivere
            print("Kunne ikke læse CUDA device navn:", e)
        try:
            mb = int(torch.cuda.memory_allocated() // (1024**2))
            print("RAM brugt (MB):", mb)
        except Exception as e:
            print("Kunne ikke læse CUDA RAM:", e)
    else:
        print("Ingen CUDA-GPU fundet – kører på CPU")

    # Enkle, robuste asserts
    assert isinstance(device, torch.device)
    if torch.cuda.is_available():
        assert device.type == "cuda"
    else:
        assert device.type == "cpu"
