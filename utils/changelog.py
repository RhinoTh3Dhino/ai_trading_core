from datetime import datetime

def append_to_changelog(message, changelog_file="CHANGELOG.md"):
    """
    Tilføjer en entry til CHANGELOG.md med timestamp og besked.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"\n### {timestamp}\n{message}\n---\n"
    try:
        with open(changelog_file, "a", encoding="utf-8") as f:
            f.write(entry)
        print("✅ CHANGELOG.md opdateret.")
    except Exception as e:
        print(f"❌ Kunne ikke opdatere CHANGELOG.md: {e}")
