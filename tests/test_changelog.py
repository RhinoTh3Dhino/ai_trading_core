import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(str(PROJECT_ROOT)))
import unittest
import os
from utils.changelog import append_to_changelog


class TestChangelog(unittest.TestCase):
    TEST_FILE = "test_CHANGELOG.md"

    def setUp(self):
        # Opretter (eller overskriver) en tom testfil inden hver test
        with open(self.TEST_FILE, "w", encoding="utf-8") as f:
            f.write("")

    def tearDown(self):
        # Sletter testfilen efter hver test
        if os.path.exists(self.TEST_FILE):
            os.remove(self.TEST_FILE)

    def test_append_to_changelog_adds_entry(self):
        entry = "Test ændring til changelog"
        append_to_changelog(entry, changelog_file=self.TEST_FILE)
        with open(self.TEST_FILE, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn(entry, content)

    def test_append_to_changelog_multiple_entries(self):
        entry1 = "Ændring entry1"
        entry2 = "Ændring entry2"
        append_to_changelog(entry1, changelog_file=self.TEST_FILE)
        append_to_changelog(entry2, changelog_file=self.TEST_FILE)
        with open(self.TEST_FILE, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn(entry1, content)
        self.assertIn(entry2, content)


if __name__ == "__main__":
    unittest.main()
