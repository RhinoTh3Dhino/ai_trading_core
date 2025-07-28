import os
import re
from pathlib import Path  # <-- VIGTIG: Tilføjet import

REPLACEMENT = 'PROJECT_ROOT = Path(__file__).parent.parent  # AUTO-FIXED PATHLIB'
IMPORT_LINE = 'from pathlib import Path\n'

def needs_import(lines):
    return not any('from pathlib import Path' in l for l in lines)

def fix_project_root_in_file(filepath):
    with open(filepath, encoding='utf-8') as f:
        lines = f.readlines()

    changed = False
    new_lines = []
    import_added = False

    for idx, line in enumerate(lines):
        # Find linjer med PROJECT_ROOT = ...
        if re.search(r'PROJECT_ROOT\s*=', line):
            # Fang forskellige "forkerte" måder:
            if ("os.path" in line or "os.path.abspath" in line
                or '"' in line or "'" in line):
                # Erstat med ny version
                print(f"[RETTER] {filepath} - linje {idx+1}: {line.strip()}")
                new_lines.append(REPLACEMENT + '\n')
                changed = True
                continue
        new_lines.append(line)

    # Tjek for import-statement
    if changed and needs_import(lines):
        # Find første import-linje
        for idx, line in enumerate(new_lines):
            if str(line).startswith("import") or str(line).startswith("from "):
                # Indsæt før første import
                new_lines.insert(idx, IMPORT_LINE)
                import_added = True
                print(f"[IMPORT] Tilføjer 'from pathlib import Path' til: {filepath}")
                break
        else:
            # Hvis ingen import, sæt øverst
            new_lines.insert(0, IMPORT_LINE)
            import_added = True
            print(f"[IMPORT] Tilføjer 'from pathlib import Path' i toppen: {filepath}")

    if changed or import_added:
        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        return True
    return False

def main():
    n_fixed = 0
    for root, dirs, files in os.walk("."):
        # Undgå virtuelle miljøer og backup-mapper
        if ".venv" in root or "backups" in root or "__pycache__" in root:
            continue
        for fname in files:
            if str(fname).endswith(".py"):
                fpath = os.path.join(root, fname)
                if fix_project_root_in_file(fpath):
                    n_fixed += 1
    print(f"\n[FÆRDIG] Antal filer rettet: {n_fixed}")

if __name__ == "__main__":
    main()
