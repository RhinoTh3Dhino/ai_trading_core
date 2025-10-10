# engines/arch.py
from __future__ import annotations

"""
Adapter til model-eksport og -indlæsning.

Brug med exporter:

  # 1) Brug din "rigtige" model via bot.engine (hvis du har en factory/klasse der):
  python -m scripts.script_export ^
    --arch engines.arch:build_model ^
    --ckpt models/best_pytorch_model.pt ^
    --out  models/best_pytorch_model.ts ^
    --init "num_features=35,hidden=64,dropout=None,out_dim=2" ^
    --mode script ^
    --strict-load

  # 2) Simpel fallback-MLP (MyNet), der matcher checkpunktets layout:
  python -m scripts.script_export ^
    --arch engines.arch:MyNet ^
    --ckpt models/best_pytorch_model.pt ^
    --out  models/best_pytorch_model.ts ^
    --init "num_features=35,hidden=64,dropout=None,out_dim=2" ^
    --mode script ^
    --strict-load

Checkpointet du viste har lag:
  net.0: Linear(35 -> 64)
  net.2: Linear(64 -> 64)
  net.4: Linear(64 -> 2)
→ derfor skal vi undgå at indsætte ekstra lag (fx Dropout/BatchNorm), da de forskubber indeks.
"""

import inspect
from typing import Callable, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------
# Lille hjælp: vælg aktiveringsfunktion (default ReLU)
# ---------------------------------------------------------
def _make_act(name: str) -> nn.Module:
    name = (name or "relu").lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "silu" or name == "swish":
        return nn.SiLU()
    # default
    return nn.ReLU()


# ---------------------------------------------------------
# En generisk MLP, der KUN registrerer lag når de er “aktive”
# (så lagindekser matcher checkpointet når dropout=None/use_bn=False)
# ---------------------------------------------------------
class MyNet(nn.Module):
    def __init__(
        self,
        num_features: int = 34,
        hidden: int = 64,
        out_dim: int = 2,
        dropout: Optional[float] = None,  # Brug None for at udelade Dropout-lag helt
        use_bn: bool = False,  # BatchNorm kun hvis True
        act: str = "relu",  # aktiveringsnavn
    ):
        super().__init__()
        self.num_features = int(num_features)
        self.hidden = int(hidden)
        self.out_dim = int(out_dim)

        act1 = _make_act(act)
        act2 = _make_act(act)

        layers: list[nn.Module] = []

        # Blok 1
        layers.append(nn.Linear(self.num_features, self.hidden))
        if use_bn:
            layers.append(nn.BatchNorm1d(self.hidden))
        layers.append(act1)
        if dropout is not None and dropout > 0:
            layers.append(nn.Dropout(float(dropout)))

        # Blok 2
        layers.append(nn.Linear(self.hidden, self.hidden))
        if use_bn:
            layers.append(nn.BatchNorm1d(self.hidden))
        layers.append(act2)
        if dropout is not None and dropout > 0:
            layers.append(nn.Dropout(float(dropout)))

        # Output
        layers.append(nn.Linear(self.hidden, self.out_dim))

        # VIGTIGT: rækkefølgen her afgør modulindeks (0,2,4 = Linear hvis ingen Dropout/BN)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forvent [N, num_features] eller [*, num_features]
        if not x.is_floating_point():
            x = x.float()
        if x.dim() > 2:
            # fold batch-dimensioner sammen undtagen feature-aksen
            x = x.view(-1, x.size(-1))
        return self.net(x)


# ---------------------------------------------------------
# Resolver til din "rigtige" model i bot.engine
# ---------------------------------------------------------
def _get_engine_module():
    try:
        import bot.engine as E  # type: ignore

        return E
    except Exception as e:
        raise ModuleNotFoundError(
            "Kunne ikke importere 'bot.engine'. Kør fra repo-roden eller tjek din PYTHONPATH."
        ) from e


def _wrap_class_ctor(cls: type) -> Callable[..., nn.Module]:
    def _ctor(**kwargs) -> nn.Module:
        return cls(**kwargs)

    return _ctor


def _find_factory_in_engine() -> Optional[Callable[..., nn.Module]]:
    """
    Prøv i rækkefølge:
      1) Fabriksfunktioner i bot.engine (build_model, build_torch_model, make_model, create_model, get_model)
      2) Klasse TradingNet på toplevel (eller første bedste nn.Module-klasse)
    Returnerer en callable(**kwargs)->nn.Module eller None.
    """
    E = _get_engine_module()

    # 1) Kendte fabriksnavne
    for name in [
        "build_model",
        "build_torch_model",
        "make_model",
        "create_model",
        "get_model",
    ]:
        f = getattr(E, name, None)
        if callable(f):
            return f  # type: ignore[return-value]

    # 2) TradingNet-klasse på modul-toplevel
    TradingNet = getattr(E, "TradingNet", None)
    if inspect.isclass(TradingNet) and issubclass(TradingNet, nn.Module):
        return _wrap_class_ctor(TradingNet)

    # 3) Sidste udvej: første nn.Module-klasse i modulet
    for attr_name, obj in vars(E).items():
        if inspect.isclass(obj) and issubclass(obj, nn.Module):
            return _wrap_class_ctor(obj)

    return None


# ---------------------------------------------------------
# Offentlig entry: build_model
# ---------------------------------------------------------
def build_model(**kwargs) -> nn.Module:
    """
    Returnér en model-instans fra bot.engine hvis muligt; ellers giv en brugbar fejl,
    så du kan falde tilbage til engines.arch:MyNet.
    """
    factory = _find_factory_in_engine()
    if factory is None:
        E = _get_engine_module()
        classes = [n for n, o in vars(E).items() if inspect.isclass(o)]
        callables = [n for n, o in vars(E).items() if callable(o)]
        raise ImportError(
            "Kunne ikke finde en fabriksfunktion eller nn.Module-klasse i 'bot.engine'.\n"
            f"Tilgængelige klasser: {classes}\n"
            f"Tilgængelige callables: {callables}\n"
            "Løsning: Tilføj fx 'def build_model(**kwargs): ...' i bot/engine.py, "
            "eller brug fallback 'engines.arch:MyNet'."
        )
    return factory(**kwargs)


__all__ = ["MyNet", "build_model"]
