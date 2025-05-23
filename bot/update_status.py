import subprocess
from datetime import datetime

def update_bot_status(version="v0.1", strategy="N/A", model="N/A", status="Aktiv"):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = f"""# 🤖 AI Trading Bot Status

| Felt | Værdi |
|------|-------|
| **Sidst opdateret** | {now} |
| **Version** | {version} |
| **Aktiv strategi** | {strategy} |
| **Model** | {model} |
| **Status** | {status} |
"""
    with open("BotStatus.md", "w", encoding="utf-8") as f:
        f.write(content)

if __name__ == "__main__":
    update_bot_status()

def commit_bot_status():
    try:
        subprocess.run(["git", "add", "BotStatus.md"], check=True)
        subprocess.run(["git", "commit", "-m", "🔄 Auto-update BotStatus.md"], check=True)
        subprocess.run(["git", "push"], check=True)
        print("✅ BotStatus.md committed og pushet til GitHub")
    except subprocess.CalledProcessError as e:
        print("⚠️ Git-kommando fejlede:", e)

if __name__ == "__main__":
    update_bot_status()
    commit_bot_status()