# Persistens & filhygiejne (Fase 4)
- Mappestruktur under outputs/
- Navngivning: {SYMBOL}_{TF}_{ARTIFACT}_{VERSION}_{YYYYMMDD}[_HHMMSS].ext
- Latest-pegepinde: symlink eller kopi på Windows
- Rotation: tools/rotate_outputs.py (default keep-værdier i koden)
- Backup: tools/make_archive.py → archives/*.zip + CI-artifact
- Gendannelse: unzip archives/* til repo-rod
