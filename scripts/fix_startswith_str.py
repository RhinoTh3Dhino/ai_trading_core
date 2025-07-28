import os
import re

def fix_startswith_str_in_file(filepath):
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    changed = False
    new_lines = []
    for i, line in enumerate(lines):
        orig = line
        # Find .str(name).startswith(   ->   str(col).startswith(
        pattern_name = r'(\w+)\.name\.startswith\('
        line = re.sub(pattern_name, r'str(\1).startswith(', line)
        # Find .startswith(   ->   str(col).startswith(
        # (men undgå at ramme dem der allerede er str(...).startswith())
        pattern_col = r'(?<!str\()(\b\w+)\.startswith\('
        line = re.sub(pattern_col, r'str(\1).startswith(', line)
        if orig != line:
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
        if ".venv" in root or "backups" in root or "__pycache__" in root:
            continue
        for fname in files:
            if str(fname).endswith(".py"):
                fpath = os.path.join(root, fname)
                if fix_startswith_str_in_file(fpath):
                    n_fixed += 1
    print(f"\n[FÆRDIG] Antal filer rettet: {n_fixed}")

if __name__ == "__main__":
    main()
