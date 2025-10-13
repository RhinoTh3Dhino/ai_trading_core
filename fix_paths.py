import os
import re

PROJECT_ROOT_IMPORT = "from utils.project_path import PROJECT_ROOT"

MARKER = ""


def find_py_files(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip venv og cache
        if any(skip in dirpath for skip in [".venv", "venv", "__pycache__"]):
            continue
        for filename in filenames:
            if str(filename).endswith(".py") and not filename == "fix_paths.py":
                yield os.path.join(dirpath, filename)


def needs_project_root(content):
    # Tjek om scriptet bruger fil-IO der bør fixes
    search_terms = [
        "read_csv(",
        "to_csv(",
        "to_pickle(",
        "read_pickle(",
        '"data/',
        "'data/",
        '"outputs/',
        "'outputs/",
        '"models/',
        "'models/",
        "os.path.join('data'",
        'os.path.join("data"',
        "os.path.join('outputs'",
        'os.path.join("outputs"',
        'os.path.join("models"',
        "os.path.join('models'",
    ]
    return any(term in content for term in search_terms)


def insert_import(content):
    lines = content.splitlines()
    # Find første linje efter øvrige imports (typisk under "import" eller "from ...")
    insert_idx = 0
    for idx, line in enumerate(lines):
        if not (
            str(line).startswith("import") or str(line).startswith("from") or line.strip() == ""
        ):
            insert_idx = idx
            break
    lines.insert(insert_idx, PROJECT_ROOT_IMPORT)
    return "\n".join(lines)


def replace_paths(content):
    # Brug regex til at erstatte hardcode paths med PROJECT_ROOT – marker linjerne
    orig_content = content

    # Eksempler: "data/...", 'data/...', outputs/, models/
    content = re.sub(
        r'(["\'])(data|outputs|models)/([^"\']+)(["\'])',
        r'PROJECT_ROOT / "\2" / "\3"' + MARKER,
        content,
    )
    # os.path.join('data', ...) eller "outputs", ...
    content = re.sub(
        r'os\.path\.join\((["\'])(data|outputs|models)\1\s*,\s*["\']([^"\']+)["\']\)',
        r'PROJECT_ROOT / "\2" / "\3"' + MARKER,
        content,
    )
    # Pandas read_csv / to_csv direkte på fx 'data/fil.csv'
    content = re.sub(
        r'pd\.read_csv\((["\'])(data|outputs|models)/([^"\']+)(["\'])',
        r'pd.read_csv(PROJECT_ROOT / "\2" / "\3"' + MARKER,
        content,
    )
    content = re.sub(
        r'pd\.to_csv\((["\'])(data|outputs|models)/([^"\']+)(["\'])',
        r'pd.to_csv(PROJECT_ROOT / "\2" / "\3"' + MARKER,
        content,
    )
    # Hvis linjen allerede er markeret, undgå dobbelt-marker
    content = content.replace(MARKER + MARKER, MARKER)
    return content, (content != orig_content)


def fix_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    already_has_import = "from utils.project_path import PROJECT_ROOT" in content

    changed = False
    # Kun hvis filen bruger data/outputs/models
    if needs_project_root(content):
        print(f"[FIX] Retter: {filepath}")

        if not already_has_import:
            content = insert_import(content)
            print("  [IMPORT] Tilføjet PROJECT_ROOT import")

        content, changed_paths = replace_paths(content)
        if changed_paths:
            print("  [PATHS] Hardcode paths udskiftet og markeret")
            changed = True

        if changed:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            print("  [DONE] Fil opdateret\n")
        else:
            print("  [SKIP] Ingen paths at ændre\n")
    else:
        print(f"[SKIP] Ingen fil-IO: {filepath}")


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"=== Scanning Python-filer i {root_dir} ===")
    for pyfile in find_py_files(root_dir):
        fix_file(pyfile)
    print("\n=== FÆRDIG! ===")
