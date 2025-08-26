# -*- coding: utf-8 -*-
"""
Tilpassede undtagelser for botten.
"""

__all__ = [
    "EmptyCSVError",
    "MissingColumnsError",
    "InvalidConfigError",
    "APIError",
    "BackupError",
]


class EmptyCSVError(Exception):
    """Kastes når en CSV er tom."""
    pass


class MissingColumnsError(Exception):
    """Kastes når krævede kolonner mangler i CSV."""
    pass


class InvalidConfigError(Exception):
    """Kastes ved ugyldig konfiguration."""
    pass


class APIError(Exception):
    """Kastes ved API-fejl (timeout/HTTP/JSON)."""
    pass


class BackupError(Exception):
    """Kastes ved backup/restore-fejl."""
    pass
