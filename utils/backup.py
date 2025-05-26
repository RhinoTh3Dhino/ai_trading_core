import os
import shutil
from datetime import datetime

def make_backup(
    backup_folders=None,
    backup_dir="backups",
    keep_days=7,
    keep_per_day=10,
    create_dummy_if_empty=True
):
    """
    Laver backup af valgte mapper/filer med timestamp i dato-undermappe og sletter gamle backups.
    Opretter dummy test-fil hvis intet andet findes, så workflow altid producerer noget.
    """
    if backup_folders is None:
        backup_folders = [
            "models", "logs", "tuner_cache", "data",
            "BotStatus.md", "CHANGELOG.md",
        ]
    print(f"📦 Forsøger at tage backup af: {backup_folders}")

    # Lav undermappe til dagens dato
    date_str = datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.now().strftime("%H-%M-%S")
    datedir = os.path.join(backup_dir, date_str)
    backup_path = os.path.join(datedir, f"backup_{timestamp}")
    try:
        os.makedirs(backup_path, exist_ok=True)
        print(f"✅ Backup-mappe oprettet: {backup_path}")
    except Exception as e:
        print(f"❌ Fejl ved oprettelse af backup-mappe: {backup_path}: {e}")
        return None

    found_any = False
    for item in backup_folders:
        print(f"🔍 Tjekker om {item} findes: {os.path.exists(item)}")
        if os.path.exists(item):
            try:
                # Brug shutil.copytree hvis mappe, ellers copy2 hvis fil
                dst = os.path.join(backup_path, item)
                if os.path.isdir(item):
                    shutil.copytree(item, dst)
                else:
                    shutil.copy2(item, backup_path)
                print(f"✅ Backed up: {item}")
                found_any = True
            except Exception as e:
                print(f"❌ Kunne ikke backe op: {item}: {e}")
        else:
            print(f"⚠️ Advarsel: {item} findes ikke og blev ikke backet op.")

    # Hvis ingen filer/mapper blev backet op, lav en dummy-fil (så artifact upload virker)
    if not found_any and create_dummy_if_empty:
        dummy_path = os.path.join(backup_path, "dummy_backup.txt")
        with open(dummy_path, "w") as f:
            f.write("Ingen af de forventede filer/mapper fandtes - dummy backup.\n")
        print(f"🟡 Oprettede dummy-fil: {dummy_path}")

    # Slet gamle backups pr. dag og gamle dato-mapper
    try:
        cleanup_old_backups(backup_dir, keep_days, keep_per_day)
    except Exception as e:
        print(f"❌ Fejl under sletning af gamle backups: {e}")

    return backup_path

def cleanup_old_backups(backup_dir, keep_days=7, keep_per_day=10):
    """
    Sletter gamle backups:
    - Gem kun keep_days dage tilbage.
    - Gem max keep_per_day backups pr. dag.
    """
    if not os.path.exists(backup_dir):
        return

    # Ryd op i backups pr. dag
    for datedir in os.listdir(backup_dir):
        day_path = os.path.join(backup_dir, datedir)
        if os.path.isdir(day_path):
            backups = [
                d for d in os.listdir(day_path)
                if d.startswith("backup_")
            ]
            backups.sort(reverse=True)  # nyeste først
            for b in backups[keep_per_day:]:
                full_path = os.path.join(day_path, b)
                try:
                    shutil.rmtree(full_path)
                    print(f"🗑️ Slettet gammel backup: {full_path}")
                except Exception as e:
                    print(f"❌ Kunne ikke slette: {full_path}: {e}")

    # Ryd op i dato-mapper (hvis der er mere end keep_days)
    days = [
        d for d in os.listdir(backup_dir)
        if os.path.isdir(os.path.join(backup_dir, d))
    ]
    days.sort(reverse=True)
    for day_to_remove in days[keep_days:]:
        full_path = os.path.join(backup_dir, day_to_remove)
        try:
            shutil.rmtree(full_path)
            print(f"🗑️ Slettet gammel backup-dag: {full_path}")
        except Exception as e:
            print(f"❌ Kunne ikke slette dag: {full_path}: {e}")

# Test direkte hvis filen køres solo
if __name__ == "__main__":
    make_backup()
