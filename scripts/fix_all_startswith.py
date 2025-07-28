import os
import re

# Variabelnavne der sandsynligvis er Path/filer
PATH_LIKE = {"f", "file", "fname", "filepath", "path", "p", "fp"}

def fix_startswith_in_file(filepath):
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    changed = False
    new_lines = []
    for i, line in enumerate(lines):
        orig = line
        # Matcher <var>.startswith( -- men ikke allerede str(var) eller var.name
        match = re.findall(r"\b(\w+)\.startswith\(", line)
        for var in set(match):
            # Udeluk str(x).startswith eller x.name.startswith
            if f"str({var}).startswith" in line or f"{var}.name.startswith" in line:
                continue
            # Path-lignende variabler
            if var in PATH_LIKE:
                # Skift til var.str(name).startswith(
                line = re.sub(rf"\b{var}\.startswith\(", f"{var}.str(name).startswith(", line)
                print(f"[RETTER] {filepath} ({i+1}): {orig.strip()} --> {line.strip()}")
                changed = True
            else:
                # Skift til str(var).startswith(
                line = re.sub(rf"\b{var}\.startswith\(", f"str({var}).startswith(", line)
                print(f"[RETTER] {filepath} ({i+1}): {orig.strip()} --> {line.strip()}")
                changed = True
        new_lines.append(line)

    if changed:
        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
    return changed

def main():
    n_fixed = 0
    for root, dirs, files in os.walk("."):
        # Spring virtuelle miljøer og backup over
        if ".venv" in root or "backups" in root or "__pycache__" in root:
            continue
        for fname in files:
            if str(fname).endswith(".py"):
                fpath = os.path.join(root, fname)
                if fix_startswith_in_file(fpath):
                    n_fixed += 1
    print(f"\n[FÆRDIG] Antal filer rettet: {n_fixed}")

if __name__ == "__main__":
    main()
