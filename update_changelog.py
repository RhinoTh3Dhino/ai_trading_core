# bot/update_changelog.py
from datetime import datetime
import subprocess

def update_changelog(message="Auto-update CHANGELOG.md"):
    dato = datetime.now().strftime("%Y-%m-%d")
    line = f"\n\n## [{dato}]\n- {message}\n"

    with open("CHANGELOG.md", "a", encoding="utf-8") as f:
        f.write(line)

def commit_changelog():
    try:
        subprocess.run(["git", "add", "CHANGELOG.md"], check=True)
        subprocess.run(["git", "commit", "-m", "Auto-update CHANGELOG.md"], check=True)
        subprocess.run(["git", "push"], check=True)
        print("[INFO] Changelog committed og pushet til GitHub.")
    except subprocess.CalledProcessError as e:
        print("[ERROR] Git-kommando fejlede:", e)

if __name__ == "__main__":
    update_changelog("Step 5: Automatisk changelog test")
    commit_changelog()
