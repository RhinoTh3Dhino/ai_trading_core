import os
import re

# Typiske variable du bruger på kolonner, paths, filer osv.
TYPICAL_VAR_NAMES = [
    "col",
    "c",
    "file",
    "fname",
    "filename",
    "path",
    "first_line",
    "d",
    "f",
]

# Problematiske/ubrugelige auto-rettelser eller fejlmønstre
ERROR_PATTERNS = [
    r"str\(str\([^)]+\)\)\.(startswith|endswith)\(",  # Dobbelt str(str(...)).startswith
    r"str\([^)]+\.name\)\.(startswith|endswith)\(",  # str(col.name).startswith(...)
    r"str\([^)]+\.columns\)\.(startswith|endswith)\(",  # str(df.columns).startswith(...)
    r"\.name\.name\.",  # dobbelt .name
    r"\.name\.columns\.",  # .name.columns
]


def auto_fix_startswith_in_file(filepath):
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    changed = False
    new_lines = []
    fix_pattern = re.compile(
        r"(\b(?:" + "|".join(TYPICAL_VAR_NAMES) + r")\b)\.(startswith|endswith)\("
    )

    for i, line in enumerate(lines):
        # --- Fix phase ---
        matches = list(fix_pattern.finditer(line))
        if matches:
            new_line = line
            for m in reversed(matches):
                varname = m.group(1)
                start, end = m.span(1)
                new_line = new_line[:start] + f"str({varname})" + new_line[end:]
            if new_line != line:
                print(
                    f"[RETTER] {filepath} ({i+1}): {line.strip()} → {new_line.strip()}"
                )
                changed = True
            new_lines.append(new_line)
        else:
            new_lines.append(line)

    # --- Detection phase: Find fejlagtige eller mistænkelige mønstre ---
    problems = []
    for i, line in enumerate(new_lines):
        for pat in ERROR_PATTERNS:
            if re.search(pat, line):
                problems.append((i + 1, line.strip(), pat))

    if changed:
        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

    return changed, problems


def main():
    n_fixed = 0
    n_problems = 0
    print("\n[SCAN & FIX] Scanner for startswith/endswith fejl og autokorrigerer...")
    for root, dirs, files in os.walk("."):
        if any(x in root for x in [".venv", "venv", "__pycache__", "backups"]):
            continue
        for fname in files:
            if str(fname).endswith(".py"):
                fpath = os.path.join(root, fname)
                fixed, problems = auto_fix_startswith_in_file(fpath)
                if fixed:
                    n_fixed += 1
                if problems:
                    n_problems += len(problems)
                    print(
                        f"\n[ADVARSEL] Potentielt UGYLDIG eller MISTÆNKELIG kode i {fpath}:"
                    )
                    for lineno, snippet, pattern in problems:
                        print(f"  Linje {lineno}: {snippet}")
                        print(f"    ↳ Matcher mønster: {pattern}")

    print(f"\n[FÆRDIG] Antal filer autokorrigeret: {n_fixed}")
    print(f"[INFO] Antal linjer med mistænkelig/ubrugelig kode: {n_problems}")
    print("Gennemgå ADVARSLER manuelt for at sikre korrekthed!")


if __name__ == "__main__":
    main()
