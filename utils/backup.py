# backup.py
import os
import shutil
from datetime import datetime

def make_backup(backup_folders=None, backup_dir="backups", keep_last=10):
    """
    Laver backup af valgte mapper/filer med timestamp og sletter gamle backups.

    Args:
        backup_folders (list): Mapper/filer der skal tages backup af.
        backup_dir (str): Backupmappe.
        keep_last (int): Antal backups der skal beholdes (ældste slettes automatisk).

    Returns:
        backup_path (str): Sti til backup-mappen.
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

    # Lav backup-mappe med timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backup_path = os.path.join(backup_dir, f"backup_{timestamp}")
    os.makedirs(backup_path, exist_ok=True)

    for item in backup_folders:
        if os.path.exists(item):
            try:
                if os.path.isdir(item):
                    shutil.copytree(item, os.path.join(backup_path, item))
                else:
                    shutil.copy2(item, backup_path)
                print(f"✅ Backed up: {item}")
            except Exception as e:
                print(f"❌ Kunne ikke backe op: {item}: {e}")
        else:
            print(f"⚠️ Advarsel: {item} findes ikke og blev ikke backet op.")

    # Slet gamle backups automatisk hvis for mange
    cleanup_old_backups(backup_dir, keep_last)
    return backup_path

def cleanup_old_backups(backup_dir, keep_last=10):
    """
    Sletter gamle backup-mapper hvis der er flere end keep_last.
    """
    if not os.path.exists(backup_dir):
        return
    dirs = [d for d in os.listdir(backup_dir) if d.startswith("backup_")]
    dirs.sort(reverse=True)  # nyeste først
    for dir_to_remove in dirs[keep_last:]:
        full_path = os.path.join(backup_dir, dir_to_remove)
        try:
            if os.path.isdir(full_path):
                shutil.rmtree(full_path)
                print(f"🗑️ Slettet gammel backup: {full_path}")
        except Exception as e:
            print(f"❌ Kunne ikke slette: {full_path}: {e}")

# Test direkte
if __name__ == "__main__":
    make_backup()
