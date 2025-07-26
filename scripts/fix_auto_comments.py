import os
import re

PROJECT_ROOT = "."  # Kør fra projektroden

def fix_autopath_comments(rootdir=PROJECT_ROOT):
    changed_files = 0
    skipped_files = []
    total_lines_changed = 0

    for subdir, _, files in os.walk(rootdir):
        for fname in files:
            if fname.endswith(".py"):
                fpath = os.path.join(subdir, fname)
                try:
                    with open(fpath, encoding="utf-8") as f:
                        lines = f.readlines()
                except UnicodeDecodeError:
                    print(f"[SKIP] Kan ikke læse (forkert encoding): {fpath}")
                    skipped_files.append(fpath)
                    continue

                new_lines = []
                lines_changed = 0
                for line in lines:
                    if "" in line:
# AUTO PATH CONVERTED
                        fixed_line = re.sub(r"\s*", "", line.rstrip())
                        # Tilføj kommentaren på linjen over, hvis det er en function def eller assignment
                        if "(" in fixed_line or "=" in fixed_line:
# AUTO PATH CONVERTED
                            new_lines.append("\n")
                        new_lines.append(fixed_line + "\n")
                        lines_changed += 1
                    else:
                        new_lines.append(line)

                if new_lines != lines:
                    print(f"[RETTER] {fpath} ({lines_changed} linjer ændret)")
                    with open(fpath, "w", encoding="utf-8") as f:
                        f.writelines(new_lines)
                    changed_files += 1
                    total_lines_changed += lines_changed

    print("\n=== Summary ===")
    print(f"Filer rettet: {changed_files}")
    print(f"Antal linjer ændret: {total_lines_changed}")
    if skipped_files:
        print("\n[ADVARSEL] Filer sprunget over pga. encoding-fejl:")
        for f in skipped_files:
            print(f"  - {f}")
# AUTO PATH CONVERTED
    print("\nAlle inline kommentarer er nu flyttet/fjernet.")

if __name__ == "__main__":
    fix_autopath_comments()
