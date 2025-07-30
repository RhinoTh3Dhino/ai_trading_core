"""
utils/project_path.py

Fælles funktion til at sikre korrekt sys.path setup og give nem adgang til projektroden som Path-objekt.
Gør det muligt at importere moduler på tværs af mapper og altid bygge paths uden hardcode.
"""

import os
import sys
from pathlib import Path

# Automatisk fastlæg projektroden som Path-objekt
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

def ensure_project_root(verbose: bool = False):
    """
    Sikrer at projektets rodmappe er tilføjet til sys.path (til imports).
    Skal kaldes i alle scripts, hvis import-fejl opstår, men run.py håndterer det normalt automatisk.
    """
    project_root_str = str(PROJECT_ROOT)
    if project_root_str not in sys.path:
        sys.path.insert(0, str(project_root_str))
        if verbose:
            print(f"[INFO] Added PROJECT_ROOT to sys.path: {project_root_str}")
    elif verbose:
        print(f"[INFO] PROJECT_ROOT allerede i sys.path: {project_root_str}")
