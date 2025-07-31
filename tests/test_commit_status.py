# tests/test_commit_status.py

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(str(PROJECT_ROOT)))
import sys
import os
import unittest
from unittest.mock import patch

# Tilføj projekt-roden til sys.path så Python kan finde update_bot_status.py
sys.path.insert(0, str(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))))

from update_bot_status import commit_bot_status


class TestCommitBotStatus(unittest.TestCase):
    @patch("subprocess.run")
    def test_commit_bot_status_runs_git_commands(self, mock_run):
        commit_bot_status()
        # Saml alle kaldte kommandoer som lister (argumenterne til subprocess.run)
        called_cmds = [list(call[0][0]) for call in mock_run.call_args_list]
        # Tjek at de vigtige git-kommandoer er blevet kaldt
        self.assertIn(["git", "add", "BotStatus.md"], called_cmds)
        self.assertIn(["git", "commit", "-m", "Auto-update BotStatus.md"], called_cmds)
        self.assertIn(["git", "push"], called_cmds)
        # Mindst 3 git-kommandoer bør være kaldt (kan være flere)
        self.assertGreaterEqual(len(called_cmds), 3)


if __name__ == "__main__":
    unittest.main()
