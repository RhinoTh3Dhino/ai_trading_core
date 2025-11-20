from pathlib import Path

from dotenv import load_dotenv


def load_env():
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env", override=False)
