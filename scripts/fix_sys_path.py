import os
import re

# Mønstre at finde og rette
patterns = [
    # sys.path.insert(0, str(Path(...)))
    (r"(sys\.path\.insert\(\s*\d+\s*,\s*)([^)\n]+)(\))", r"\1str(\2)\3"),
    # sys.path.append(str(Path(...)))
    (r"(sys\.path\.append\(\s*)([^)\n]+)(\))", r"\1str(\2)\3"),
    # sys.path.insert(0, str(PROJECT_ROOT)) - uden str()
    (r"(sys\.path\.insert\(\s*\d+\s*,\s*)(PROJECT_ROOT)(\))", r"\1str(\2)\3"),
    # sys.path.append(str(PROJECT_ROOT)) - uden str()
    (r"(sys\.path\.append\(\s*)(PROJECT_ROOT)(\))", r"\1str(\2)\3"),
]

def fix_sys_path_in_file(filepath):
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    changed = False
    new_lines = []
    for i, line in enumerate(lines):
        orig = line
        for pattern, replacement in patterns:
            if re.search(pattern, line):
                line_fixed = re.sub(pattern, replacement, line)
                if line_fixed != line:
                    print(f"[RETTER] {filepath} (linje {i+1}): {orig.strip()} → {line_fixed.strip()}")
                    changed = True
                    line = line_fixed
        new_lines.append(line)

    if changed:
        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
    return changed

def main():
    n_fixed = 0
    for root, dirs, files in os.walk("."):
        if any(skip in root for skip in [".venv", "__pycache__", "backups", "env", ".git"]):
            continue
        for fname in files:
            if fname.endswith(".py"):
                path = os.path.join(root, fname)
                if fix_sys_path_in_file(path):
                    n_fixed += 1
    print(f"\n[FÆRDIG] Antal filer rettet: {n_fixed}")

if __name__ == "__main__":
    print("[SCAN & FIX] Scanner for sys.path-problemer og autokorrigerer...")
    main()
