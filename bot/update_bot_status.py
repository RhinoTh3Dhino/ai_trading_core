import subprocess
from datetime import datetime

def update_bot_status(
    version="v0.1", 
    strategy="N/A", 
    model="N/A", 
    status="Aktiv",
    backup_path="Ingen",
    error_msg="Ingen"
):
    # Nu med korrekt encoding og tilpasset format til test/produktion
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = (
        "# BotStatus\n\n"
        f"**Seneste kørsel:**  {date_str}  \n"
        f"**Status:** {status}  \n"
        f"**Sidste backup:** {backup_path}  \n"
        f"**Fejl:** {error_msg}  \n"
        "\n---\n"
    )

    with open("BotStatus.md", "w", encoding="utf-8") as f:
        f.write(content)

def commit_bot_status():
    try:
        subprocess.run(["git", "add", "BotStatus.md"], check=True)
        subprocess.run(["git", "commit", "-m", "Auto-update BotStatus.md"], check=True)
        subprocess.run(["git", "push"], check=True)
        print("✅ BotStatus.md er committet og pushet til GitHub")
    except subprocess.CalledProcessError as e:
        print(f"❌ Git-kommando fejlede: {e}")

if __name__ == "__main__":
    # Testkørsel (kan køres direkte)
    update_bot_status(status="✅ Test", backup_path="dummy_path", error_msg="Ingen")
    # commit_bot_status()  # Uncomment hvis du vil auto-pushe
