from datetime import datetime
import os

def append_to_changelog(message, changelog_file="CHANGELOG.md"):
    """
    Tilføjer en entry til CHANGELOG.md (eller testfil) med timestamp og besked.
    Opretter changelog-filen hvis den ikke findes.
    """
    # Opret filen hvis den ikke eksisterer (for test og CI)
    if not os.path.exists(changelog_file):
        with open(changelog_file, "w", encoding="utf-8") as f:
            f.write("# CHANGELOG\n\n")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"\n### {timestamp}\n{message}\n---\n"
    try:
        with open(changelog_file, "a", encoding="utf-8") as f:
            f.write(entry)
        print(f"✅ CHANGELOG.md opdateret ({changelog_file}).")
    except Exception as e:
        print(f"❌ Kunne ikke opdatere {changelog_file}: {e}")
