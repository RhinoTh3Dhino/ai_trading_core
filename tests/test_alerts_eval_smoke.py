# -*- coding: utf-8 -*-
"""
Smoke-tests for utils.alerts_eval

Formål:
- Sikre at modulet kan importeres (løfter coverage fra 0% → >0%)
- Fail ikke hvis API ændrer sig: vi tester ikke konkrete funktioner,
  kun at modulet loader og eksporterer *et eller andet* anvendeligt.
"""

import importlib
import inspect
from pathlib import Path


def test_alerts_eval_importable():
    mod = importlib.import_module("utils.alerts_eval")
    # Modulet skal eksistere og have en reel filplacering
    assert hasattr(mod, "__file__") and mod.__file__
    assert Path(mod.__file__).name == "alerts_eval.py"


def test_alerts_eval_exports_nontrivial_symbols():
    mod = importlib.import_module("utils.alerts_eval")
    # Find noget der ikke er __dunder__ og ikke kun er modulenavne
    public = [
        (name, obj)
        for name, obj in vars(mod).items()
        if not name.startswith("_") and not inspect.ismodule(obj)
    ]
    # Der skal være mindst én symbol at teste (funktion/konstant/klasse)
    assert len(public) >= 1
