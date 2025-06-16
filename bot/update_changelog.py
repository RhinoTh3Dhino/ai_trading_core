# bot/update_changelog.py
from datetime import datetime

def update_changelog(message="Auto-update CHANGELOG.md"):
    dato = datetime.now().strftime("%Y-%m-%d")
    line = f"\n\n## [{dato}]\n- {message}\n"
    with open("CHANGELOG.md", "a", encoding="utf-8") as f:
        f.write(line)

if __name__ == "__main__":
    # Her kan du ændre beskeden eller hente fra env/argumenter hvis du vil
    update_changelog("Step 5: Automatisk changelog test")
