"""
utils/project_path.py

Fælles funktion til at sikre korrekt sys.path setup for hele AI trading bot projektet.
Gør det muligt at importere moduler på tværs af mapper uden problemer.
"""

import os
import sys

def ensure_project_root(verbose: bool = False):
    """
    Sikrer at projektets rodmappe er tilføjet til sys.path.
    Skal kaldes i alle scripts der bruges med run.py eller standalone.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        if verbose:
            print(f"[INFO] Added PROJECT_ROOT to sys.path: {project_root}")
    elif verbose:
        print(f"[INFO] PROJECT_ROOT allerede i sys.path: {project_root}")
