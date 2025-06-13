# tests/test_backup.py
import unittest
from utils.backup import make_backup
import os

class TestBackup(unittest.TestCase):
    def test_make_backup_creates_folder(self):
        # Brug de nye parametre fra make_backup
        path = make_backup(keep_days=1, keep_per_day=1)
        print("\n=== DEBUG: path returned from make_backup ===")
        print(path)
        print("EXISTS:", os.path.exists(path))
        print("ISDIR:", os.path.isdir(path))
        print("LISTDIR (backups):", os.listdir("backups"))

        # Her tjekker vi både om path eksisterer og er en mappe
        self.assertTrue(os.path.exists(path), f"Path does not exist: {path}")
        self.assertTrue(os.path.isdir(path), f"Path is not a directory: {path}")

if __name__ == "__main__":
    unittest.main()

