# engines/inference.py
from __future__ import annotations
import json, pickle
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple

def load_feature_order(path: Path) -> Optional[list]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None

def load_scaler(path: Path) -> Optional[object]:
    p = Path(path)
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None

def load_torch_model(path: Path, device: str="cpu") -> torch.nn.Module:
    p = Path(path)
    try:
        m = torch.jit.load(str(p), map_location=device)  # hvis jit
        m.eval()
        return m
    except Exception:
        obj = torch.load(str(p), map_location=device)
        if isinstance(obj, torch.nn.Module):
            obj.eval()
            return obj
        # hvis det er state_dict kræver det din model-klasse; her en simpel MLP stub
        raise RuntimeError("Torch model kunne ikke loades (kræver din model-klasse).")

@torch.no_grad()
def run_inference(df_features, feature_order: Optional[list], scaler, model) -> Tuple[np.ndarray, np.ndarray]:
    X = df_features[feature_order].to_numpy(dtype=np.float32) if feature_order else df_features.to_numpy(dtype=np.float32)
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception:
            pass
    tens = torch.from_numpy(X)
    logits = model(tens)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    arr = logits.detach().cpu().numpy()
    # binær: brug sigmoid; multi-class: softmax->klasse 1 sandsynlighed
    if arr.ndim==1 or arr.shape[1]==1:
        prob = 1/(1+np.exp(-arr.squeeze()))
        pred = (prob >= 0.5).astype(int)
        return pred, prob
    else:
        e = np.exp(arr - arr.max(axis=1, keepdims=True))
        sm = e / e.sum(axis=1, keepdims=True)
        prob = sm[:,1] if sm.shape[1]>1 else sm[:,0]
        pred = sm.argmax(axis=1)
        return pred, prob
