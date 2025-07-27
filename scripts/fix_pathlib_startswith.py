import os
import re

# Forbedret regex: Matcher kun brug på variable col (fx for col in df.columns osv.)
COLUMN_VAR_NAMES = ["col", "c", "column"]  # Udvid evt. listen hvis du har flere aliaser

def fix_pathlib_startswith_in_file(filepath):
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    changed = False
    new_lines = []
    for i, line in enumerate(lines):
        original_line = line
        # Tjek kun .startswith på variable der typisk bruges til kolonner
        for var in COLUMN_VAR_NAMES:
            # Vi vil kun ændre fx: col.name.startswith('target')
            pattern = rf"\b{var}\.startswith\("
            pattern_name = rf"\b{var}\.name\.startswith\("
            # Kun hvis der IKKE allerede står .name
            if re.search(pattern, line) and not re.search(pattern_name, line):
                # Erstat col.startswith( med col.name.startswith(
                line = re.sub(pattern, f"{var}.name.startswith(", line)
                if original_line != line:
                    print(f"[RETTER] {filepath} ({i+1}): {original_line.strip()} --> {line.strip()}")
                    changed = True
        new_lines.append(line)

    if changed:
        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
    return changed

def main():
    n_fixed = 0
    for root, dirs, files in os.walk("."):
        if ".venv" in root or "backups" in root or "__pycache__" in root:
            continue
        for fname in files:
            if fname.endswith(".py"):
                fpath = os.path.join(root, fname)
                if fix_pathlib_startswith_in_file(fpath):
                    n_fixed += 1
    print(f"\n[FÆRDIG] Antal filer rettet: {n_fixed}")

if __name__ == "__main__":
    main()
