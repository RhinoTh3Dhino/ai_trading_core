# utils/botstatus.py

from datetime import datetime


def update_bot_status(
    status="✅ Succes", backup_path=None, error_msg=None, extra_fields=None
):
    """
    Opdaterer BotStatus.md med status for sidste kørsel.
    Skriver de vigtigste felter og kan udvides med flere felter via extra_fields (dict).
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# BotStatus",
        "",
        f"**Seneste kørsel:** {now}  ",
        f"**Status:** {status}  ",
        f"**Sidste backup:** {backup_path if backup_path else 'Ingen'}  ",
        f"**Fejl:** {error_msg if error_msg else 'Ingen'}  ",
    ]
    # Mulighed for at tilføje ekstra felter (fx balance, handler, etc.)
    if extra_fields is not None:
        for key, value in extra_fields.items():
            lines.append(f"**{key}:** {value}  ")

    lines.append("\n---\n")

    with open("BotStatus.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("📝 BotStatus.md opdateret.")


# Eksempel på direkte test/run (kan slettes eller kommenteres ud)
if __name__ == "__main__":
    update_bot_status(
        status="✅ Testkørsel OK",
        backup_path="backups/backup_2025-05-24_00-00-00",
        error_msg=None,
        extra_fields={"Balance": "10000 USDT", "Antal handler": 42},
    )
