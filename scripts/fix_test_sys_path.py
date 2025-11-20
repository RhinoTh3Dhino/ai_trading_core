import os
from pathlib import Path

SYS_PATH_HACK = """import sys
import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(str(PROJECT_ROOT)))

"""


def patch_file(file_path):
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    # Hvis allerede sat ind, skip!
    if "sys.path.insert(0, str(str(PROJECT_ROOT)))" in content:
        return False

    # Find første ikke-docstring import eller tom linje
    lines = content.splitlines()
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("import") or line.strip().startswith("from"):
            insert_idx = i
            break

    # Indsæt hack før første import
    new_content = lines[:insert_idx] + [SYS_PATH_HACK.rstrip()] + lines[insert_idx:]
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(new_content))
    print(f"[RETTER] {file_path}")
    return True


def main():
    n = 0
    for root, dirs, files in os.walk("tests"):
        for fname in files:
            if str(fname).endswith(".py"):
                fpath = os.path.join(root, fname)
                if patch_file(fpath):
                    n += 1
    print(f"\n[FÆRDIG] Rettet sys.path i {n} test-filer.")


if __name__ == "__main__":
    main()
