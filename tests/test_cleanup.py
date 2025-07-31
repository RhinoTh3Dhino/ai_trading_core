import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(str(PROJECT_ROOT)))
import unittest
import os
import shutil
from utils.backup import cleanup_old_backups


class TestCleanupBackups(unittest.TestCase):
    def setUp(self):
        # Opret test-backup-mapper med 5 dage, 3 backups pr. dag
        self.backup_dir = "test_backups"
        os.makedirs(self.backup_dir, exist_ok=True)
        for i in range(5):
            day_dir = os.path.join(self.backup_dir, f"2025-05-2{i}")
            os.makedirs(day_dir, exist_ok=True)
            for j in range(3):
                backup_dir = os.path.join(day_dir, f"backup_0{i}-0{j}-00")
                os.makedirs(backup_dir, exist_ok=True)

    def tearDown(self):
        # Slet hele test-backup-mappen efter hver test
        shutil.rmtree(self.backup_dir)

    def test_cleanup_removes_old_days_and_limits_per_day(self):
        # Behold kun 2 dage og max 2 backups pr. dag
        cleanup_old_backups(self.backup_dir, keep_days=2, keep_per_day=2)
        days = [d for d in os.listdir(self.backup_dir)]
        self.assertEqual(len(days), 2)  # Kun 2 dage tilbage
        for day in days:
            backups = [b for b in os.listdir(os.path.join(self.backup_dir, day))]
            self.assertLessEqual(len(backups), 2)  # Max 2 backups pr. dag


if __name__ == "__main__":
    unittest.main()
