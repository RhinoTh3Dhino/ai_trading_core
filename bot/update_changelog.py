from datetime import datetime
import subprocess

def update_changelog(message="Opdatering uden note"):
    dato = datetime.now().strftime("%Y-%m-%d")
    linje = f"\n## [{dato}]\n✅ {message}\n"

    with open("CHANGELOG.md", "a", encoding="utf-8") as f:
        f.write(linje)

def commit_changelog():
    try:
        subprocess.run(["git", "add", "CHANGELOG.md"], check=True)
        subprocess.run(["git", "commit", "-m", "Auto-update CHANGELOG.md"], check=True)
        subprocess.run(["git", "push"], check=True)
        print("✅ Changelog committed og pushed")
    except subprocess.CalledProcessError as e:
        print("❌ Git-kommando fejlede:", e)

if __name__ == "__main__":
    update_changelog("Step 4 test")
    commit_changelog()
