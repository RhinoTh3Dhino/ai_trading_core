# -*- coding: utf-8 -*-
"""
CSV-hjælpefunktioner uden eksterne afhængigheder.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .errors import EmptyCSVError, MissingColumnsError


def safe_read_csv(
    path: str | Path,
    required_columns: Optional[Iterable[str]] = None,
    encoding: str = "utf-8",
    delimiter: Optional[str] = None,
    skip_blank_rows: bool = True,
) -> List[Dict[str, str]]:
    """
    Læser en CSV og returnerer liste af rækker (dicts).
    Kaster:
      - EmptyCSVError for tom fil
      - MissingColumnsError hvis en eller flere 'required_columns' mangler

    Parametre:
      path: filsti
      required_columns: liste/iterable af kolonnenavne, der skal findes (case-sensitivt)
      encoding: standard 'utf-8'
      delimiter: hvis angivet, bruges denne delimiter (ellers sniff/fallback til ',')
      skip_blank_rows: hvis True, skipper rækker hvor alle felter er tomme
    """
    p = Path(path)
    required = list(required_columns or [])

    with p.open("r", encoding=encoding, newline="") as fh:
        content = fh.read()
        if not content.strip():
            raise EmptyCSVError("CSV-filen er tom.")

        # Find delimiter
        try:
            first_non_empty = next(
                (ln for ln in content.splitlines() if ln.strip()), ""
            )
            sniffed = (
                csv.Sniffer().sniff(first_non_empty) if first_non_empty else csv.excel
            )
            used_delimiter = delimiter or sniffed.delimiter
        except Exception:
            used_delimiter = delimiter or ","

        fh.seek(0)
        reader = csv.DictReader(fh, delimiter=used_delimiter, skipinitialspace=True)

        # Normalisér header (fjern evt. UTF-8 BOM på første kolonnenavn)
        raw_header = reader.fieldnames or []
        header = [h.lstrip("\ufeff") if isinstance(h, str) else "" for h in raw_header]

        # Valider krævede kolonner (case-sensitivt for forudsigelighed)
        missing = [c for c in required if c not in header]
        if missing:
            raise MissingColumnsError(f"Manglende kolonner: {missing}")

        # Læs rækker og map til normaliseret header (så BOM ikke smitter nøglerne)
        rows: List[Dict[str, str]] = []
        for row in reader:
            # Skipper helt blanke rækker
            if skip_blank_rows and all(
                (v is None) or (str(v).strip() == "") for v in row.values()
            ):
                continue
            # Map ved indeks så vi kan erstatte nøgler med den rensede header
            normalized: Dict[str, str] = {}
            for i, col in enumerate(header):
                # reader.fieldnames kan indeholde BOM; brug indeks for at slå op i row
                key_raw = raw_header[i] if i < len(raw_header) else col
                normalized[col] = row.get(key_raw)
            rows.append(normalized)

        return rows
