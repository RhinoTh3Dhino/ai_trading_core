import os
import shutil
from datetime import datetime


def make_backup(
    backup_folders=None,
    backup_dir="backups",
    keep_days=7,
    keep_per_day=10,
    create_dummy_if_empty=True,
    force_dummy=False,
):
    """
    Laver backup af valgte mapper/filer med timestamp i dato-undermappe og sletter gamle backups.
    Opretter dummy test-fil hvis intet andet findes, så workflow altid producerer noget.
    """
    if backup_folders is None:
        backup_folders = [
            "models",
            "logs",
            "tuner_cache",
            "data",
            "BotStatus.md",
            "CHANGELOG.md",
        ]

    if keep_days < 0 or keep_per_day < 0:
        raise ValueError("keep_days og keep_per_day skal være >= 0")

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
        if os.path.exists(item) and not force_dummy:
            try:
                dst = os.path.join(backup_path, os.path.basename(item))
                if os.path.isdir(item):
                    shutil.copytree(item, dst)
                else:
                    shutil.copy2(item, dst)
                print(f"✅ Backed up: {item}")
                found_any = True
            except Exception as e:
                print(f"❌ Kunne ikke backe op: {item}: {e}")
        else:
            print(f"⚠️ Advarsel: {item} findes ikke og blev ikke backet op.")

    # Dummy-fil hvis nødvendigt
    if (not found_any and create_dummy_if_empty) or force_dummy:
        dummy_path = os.path.join(backup_path, "dummy_backup.txt")
        with open(dummy_path, "w") as f:
            f.write("Ingen af de forventede filer/mapper fandtes - dummy backup.\n")
        print(f"🟡 Oprettede dummy-fil: {dummy_path}")

    # Ryd op i gamle backups
    try:
        cleanup_old_backups(backup_dir, keep_days, keep_per_day)
    except Exception as e:
        print(f"❌ Fejl under sletning af gamle backups: {e}")

    return backup_path


def cleanup_old_backups(backup_dir, keep_days=7, keep_per_day=10):
    """
    Sletter gamle backups:
    - Gem kun keep_days dage tilbage (minimum 1 dag).
    - Gem max keep_per_day backups pr. dag.
    Returnerer liste over slettede filer/mapper for testformål.
    """
    deleted_items = []

    if not os.path.exists(backup_dir):
        return deleted_items

    # Altid behold mindst én dags backups
    keep_days = max(1, keep_days)

    # Ryd op i backups pr. dag
    for datedir in os.listdir(backup_dir):
        day_path = os.path.join(backup_dir, datedir)
        if os.path.isdir(day_path):
            backups = [d for d in os.listdir(day_path) if str(d).startswith("backup_")]
            backups.sort(reverse=True)  # nyeste først
            for b in backups[keep_per_day:]:
                full_path = os.path.join(day_path, b)
                try:
                    shutil.rmtree(full_path)
                    deleted_items.append(full_path)
                    print(f"🗑️ Slettet gammel backup: {full_path}")
                except Exception as e:
                    print(f"❌ Kunne ikke slette: {full_path}: {e}")

    # Ryd op i dato-mapper, men aldrig den nyeste dag
    days = [
        d for d in os.listdir(backup_dir) if os.path.isdir(os.path.join(backup_dir, d))
    ]
    days.sort(reverse=True)
    for day_to_remove in days[keep_days:]:
        full_path = os.path.join(backup_dir, day_to_remove)
        try:
            shutil.rmtree(full_path)
            deleted_items.append(full_path)
            print(f"🗑️ Slettet gammel backup-dag: {full_path}")
        except Exception as e:
            print(f"❌ Kunne ikke slette dag: {full_path}: {e}")

    return deleted_items


if __name__ == "__main__":
    # CLI/direkte test
    try:
        path = make_backup(force_dummy=True)
        print(f"Test-backup oprettet i: {path}")
    except Exception as e:
        print(f"Fejl ved test-backup: {e}")
