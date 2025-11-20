# tests/test_botstatus.py
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(str(PROJECT_ROOT)))
import os
import unittest

from utils.botstatus import update_bot_status


class TestBotStatus(unittest.TestCase):
    def test_update_bot_status_creates_file(self):
        # Undlad emoji i test for at undgå encoding/Unicode-problemer på Windows
        update_bot_status(status="Test", backup_path="dummy_path", error_msg="Ingen")
        with open("BotStatus.md", encoding="utf-8") as f:
            content = f.read()
        print("DEBUG content:", repr(content))
        # Søg efter "Test" (uden emoji) for at sikre det virker i alle miljøer
        self.assertIn("Test", content)


if __name__ == "__main__":
    unittest.main()
