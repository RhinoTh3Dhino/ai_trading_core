# scripts/script_export.py
from __future__ import annotations

import argparse
import ast
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

# --------------------------------------------------------------------
# Sørg for at repo-roden er på sys.path, uanset hvorfra scriptet køres
# --------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _import_class(dotted: str):
    """
    Importer en klasse via 'modul:Klasse' eller 'modul.Klasse'.
    Eksempler:
      engines.arch:MyNet
      engines.arch.MyNet
    """
    if ":" in dotted:
        mod_name, cls_name = dotted.split(":", 1)
    else:
        parts = dotted.rsplit(".", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Ugyldigt --arch format: {dotted!r} (brug 'modul:Klasse' eller 'modul.Klasse')"
            )
        mod_name, cls_name = parts
    try:
        mod = importlib.import_module(mod_name)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Kunne ikke importere modulet {mod_name!r}. "
            f"Sørg for at køre fra repo-roden, eller at modulet findes. "
            f"(sys.path[0]={sys.path[0]!r})"
        ) from e
    try:
        cls = getattr(mod, cls_name)
    except AttributeError as e:
        raise ImportError(f"Kunne ikke finde klasse {cls_name!r} i modul {mod_name!r}") from e
    return cls


def _parse_init(s: Optional[str], init_json: Optional[str]) -> Dict[str, Any]:
    """
    Parse init-kwargs for modelkonstruktion.
    - --init "hidden=128,dropout=0.2,num_features=34,use_bias=True"
    - --init-json '{"hidden":128,"dropout":0.2}'
      eller --init-json path/to/kwargs.json
    """
    out: Dict[str, Any] = {}
    if init_json:
        if init_json.strip().startswith("{"):
            out.update(json.loads(init_json))
        else:
            with open(init_json, "r", encoding="utf-8") as f:
                out.update(json.load(f))

    if s:
        for token in filter(None, [t.strip() for t in s.split(",")]):
            if "=" not in token:
                raise ValueError(f"Init arg skal være key=val (fik {token!r})")
            k, v = token.split("=", 1)
            k = k.strip()
            v = v.strip()
            # prøv at fortolke Python literal (int/float/bool/None/list/dict)
            try:
                out[k] = ast.literal_eval(v)
            except Exception:
                out[k] = v  # behold som string
    return out


def _find_state_dict(obj: Any) -> Optional[Dict[str, Any]]:
    """
    Find state_dict i et checkpoint-objekt.
    Understøtter typiske nøgler: 'state_dict', 'model', 'model_state', etc.
    """
    if isinstance(obj, dict):
        # Mest almindelige varianter
        for key in ("state_dict", "model_state", "weights", "params"):
            sd = obj.get(key)
            if isinstance(sd, dict):
                return sd

        # Nogle gemmer hele nn.Module under 'model'
        m = obj.get("model", None)
        if m is not None:
            if isinstance(m, dict):
                return m
            if hasattr(m, "state_dict"):
                return m.state_dict()

        # Hvis alt i dict ligner et state_dict (string->tensor/obj)
        if all(isinstance(k, str) for k in obj.keys()):
            # heuristik: i mange state_dicts er værdierne ikke nested dicts
            return obj

    # Direkte nn.Module?
    if hasattr(obj, "state_dict"):
        return obj.state_dict()  # type: ignore[attr-defined]

    return None


def _try_load_as_torchscript(path: Path, device: str) -> Optional[torch.jit.ScriptModule]:
    try:
        m = torch.jit.load(str(path), map_location=device)
        m.eval()
        return m
    except Exception:
        return None


def _make_example(
    example_shape: Optional[str], example_npy: Optional[str], dtype: str = "float32"
) -> Optional[torch.Tensor]:
    """
    Lav et eksempel-input til tracing/validering.
    - example_shape: "1,34" eller "1x10x34"
    - example_npy: sti til .npy-fil (foretrækkes hvis givet)
    """
    if example_npy:
        arr = np.load(example_npy)
        t = torch.tensor(arr)
        return t

    if not example_shape:
        return None

    sep = "x" if "x" in example_shape.lower() else ","
    shape = tuple(int(s) for s in example_shape.replace("X", "x").split(sep) if s.strip())
    if not shape:
        return None

    dt = dict(float32=torch.float32, float64=torch.float64, float16=torch.float16).get(
        dtype.lower(), torch.float32
    )
    return torch.randn(*shape, dtype=dt)


def export_to_torchscript(
    arch: str,
    ckpt_path: Union[str, Path],
    out_path: Optional[Union[str, Path]] = None,
    init_args: Optional[str] = None,
    init_json: Optional[str] = None,
    mode: str = "script",  # "script" eller "trace"
    example_shape: Optional[str] = None,
    example_npy: Optional[str] = None,
    strict_load: bool = False,
    device: str = "cpu",
) -> Path:
    """
    Hovedfunktion: loader checkpoint, bygger model, og gemmer TorchScript.
    """
    ckpt_p = Path(ckpt_path)
    if out_path is None:
        # Hvis input hedder *.pt → gem som *.ts, ellers bare tilføj .ts
        if ckpt_p.suffix.lower() == ".pt":
            out_p = ckpt_p.with_suffix(".ts")
        else:
            out_p = ckpt_p.with_suffix(ckpt_p.suffix + ".ts")
    else:
        out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # 0) Hvis det allerede ER TorchScript, så gen-gem bare.
    scripted = _try_load_as_torchscript(ckpt_p, device=device)
    if scripted is not None:
        scripted.save(str(out_p))
        print(f"[OK] Input var allerede TorchScript → {out_p}")
        return out_p

    # 1) Ellers er det sandsynligvis et almindeligt checkpoint/state_dict
    print(f"[INFO] Loader checkpoint: {ckpt_p}")
    ckpt_obj = torch.load(str(ckpt_p), map_location=device)

    state = _find_state_dict(ckpt_obj)
    if state is None:
        raise RuntimeError(
            "Kunne ikke finde state_dict i checkpointet. "
            "Hvis dit checkpoint ikke er et TorchScript-arkiv, skal du angive --arch samt init-kwargs der matcher træningen."
        )

    # 2) Importér din modelklasse og init med kwargs
    if not arch:
        raise ValueError("--arch er tom; krævet når checkpoint ikke er TorchScript.")
    kwargs = _parse_init(init_args, init_json)
    cls = _import_class(arch)
    try:
        model = cls(**kwargs)  # VIGTIGT: skal matche træningens hyperparametre
    except TypeError as e:
        raise TypeError(
            f"Kunne ikke konstruere modellen {arch!r} med kwargs={kwargs}. "
            "Sørg for at --init / --init-json matcher træningens parametre."
        ) from e

    missing, unexpected = model.load_state_dict(state, strict=strict_load)
    if missing:
        print(f"[ADVARSEL] Missing keys i state_dict: {sorted(missing)}")
    if unexpected:
        print(f"[ADVARSEL] Unexpected keys i state_dict: {sorted(unexpected)}")

    model.to(device).eval()

    # 3) Eksempel-input (krævet for trace, valgfrit for script; bruges også til sanity-check)
    example = _make_example(example_shape, example_npy)

    # valider forward (hvis vi har example)
    if example is not None:
        with torch.no_grad():
            try:
                _ = model(example.to(device))
                print("[INFO] Forward sanity-check OK.")
            except Exception as e:
                print(f"[ADVARSEL] Forward sanity-check fejlede: {e}")

    # 4) Script/Trace
    if mode == "trace":
        if example is None:
            raise ValueError("--mode trace kræver --example-shape eller --example-npy")
        with torch.no_grad():
            scripted = torch.jit.trace(model, example.to(device), strict=False)
    else:
        # default: script
        scripted = torch.jit.script(model)

    scripted.save(str(out_p))
    print(f"[OK] TorchScript gemt → {out_p}")
    return out_p


def main() -> int:
    ap = argparse.ArgumentParser(description="Eksportér PyTorch checkpoint til TorchScript (.ts)")
    ap.add_argument(
        "--arch",
        required=False,
        help="Dotted sti til modelklasse, fx engines.arch:MyNet eller engines.arch.MyNet",
    )
    ap.add_argument(
        "--ckpt",
        default="models/best_pytorch_model.pt",
        help="Sti til checkpoint (.pt eller allerede .ts)",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Gem som (.ts). Default: samme navn som ckpt, men med .ts",
    )
    ap.add_argument(
        "--init",
        dest="init_args",
        default=None,
        help="Kommasepareret key=val, fx 'hidden=128,dropout=0.2,num_features=34'",
    )
    ap.add_argument(
        "--init-json",
        default=None,
        help="JSON string eller sti til .json med init kwargs",
    )
    ap.add_argument(
        "--mode",
        choices=["script", "trace"],
        default="script",
        help="JIT metode: script (default) eller trace",
    )
    ap.add_argument(
        "--example-shape",
        default=None,
        help="Eksempel-shape til trace/validering, fx '1,34' eller '1x10x34'",
    )
    ap.add_argument(
        "--example-npy",
        default=None,
        help="Sti til .npy med eksempel-input til trace/validering",
    )
    ap.add_argument("--strict-load", action="store_true", help="Brug strict=True i load_state_dict")
    ap.add_argument("--device", default="cpu", help="cpu/cuda (hvis tilgængelig)")
    args = ap.parse_args()

    # Hjælper også når man kører direkte: gør roden synlig for underprocesser
    os.environ.setdefault("PYTHONPATH", str(ROOT))

    # Hvis ckpt allerede er TorchScript, kan --arch være unødvendig.
    # Men til state_dict kræver vi --arch.
    ts_try = _try_load_as_torchscript(Path(args.ckpt), device=args.device)
    if ts_try is None and not args.arch:
        ap.error("--arch er påkrævet, når checkpoint ikke er et TorchScript-arkiv.")

    export_to_torchscript(
        arch=args.arch or "",
        ckpt_path=args.ckpt,
        out_path=args.out,
        init_args=args.init_args,
        init_json=args.init_json,
        mode=args.mode,
        example_shape=args.example_shape,
        example_npy=args.example_npy,
        strict_load=args.strict_load,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
