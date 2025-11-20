import os
import re


def fix_abspath_join_in_file(filepath):
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    changed = False
    new_lines = []
    # RegEx: Find forkert abspath-join-usage (ekstra parantes!)
    pattern = re.compile(
        r"os\.path\.abspath\(\s*os\.path\.join\(os\.path\.dirname\(__file__\)\)\s*,\s*'(\.\.)'\s*\)"
    )
    # Korrekt version
    replacement = r"os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))"

    for i, line in enumerate(lines):
        if pattern.search(line):
            line_fixed = pattern.sub(replacement, line)
            if line != line_fixed:
                print(f"[RETTER] {filepath} (linje {i+1}): {line.strip()} → {line_fixed.strip()}")
                changed = True
            new_lines.append(line_fixed)
        else:
            new_lines.append(line)

    if changed:
        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
    return changed


def main():
    n_fixed = 0
    for root, dirs, files in os.walk("."):
        if ".venv" in root or "__pycache__" in root or "backups" in root:
            continue
        for fname in files:
            if fname.endswith(".py"):
                fpath = os.path.join(root, fname)
                if fix_abspath_join_in_file(fpath):
                    n_fixed += 1
    print(f"\n[FÆRDIG] Antal filer rettet: {n_fixed}")


if __name__ == "__main__":
    print(
        "[SCAN & FIX] Scanner for forkert brug af os.path.abspath(os.path.join(...)), '..') og retter automatisk..."
    )
    main()
