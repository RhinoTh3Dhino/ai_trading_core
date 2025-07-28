import os
import re

# Typiske variable-navne for kolonner – udvid listen hvis du bruger flere aliaser
COLUMN_VAR_NAMES = ["col", "c", "column"]

def fix_pathlib_startswith_in_file(filepath):
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    changed = False
    new_lines = []
    for i, line in enumerate(lines):
        original_line = line

        # Ret alle startswith/endswith på kolonne-variable til str(col).startswith/endswith
        for var in COLUMN_VAR_NAMES:
            # Matcher fx str(col).startswith("target") eller str(col).startswith("target")
            pattern = rf"\b{var}(?:\.name)?\.(starts|ends)with\("
            matches = list(re.finditer(pattern, line))
            # For hver forekomst
            for match in reversed(matches):  # reversed så vi ikke forskubber indekser
                method = match.group(1)
                # Find det fulde match og erstatter det
                # Fx str(col).startswith( --> str(col).startswith(
                prefix = f"{var}.name.{method}with("
                prefix2 = f"{var}.{method}with("
                if prefix in line:
                    line = line.replace(prefix, f"str({var}).{method}with(")
                    changed = True
                elif prefix2 in line:
                    line = line.replace(prefix2, f"str({var}).{method}with(")
                    changed = True

        if original_line != line:
            print(f"[RETTER] {filepath} ({i+1}): {original_line.strip()} --> {line.strip()}")
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
