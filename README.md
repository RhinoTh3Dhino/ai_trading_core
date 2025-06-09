# AI Trading Core

Dette projekt er fundamentet for en avanceret AI trading bot.

# test af Trigger CI workflow

# 📦 Standard Commit Guide & Ekstra Tips

## 1. Tilføj ændringer
git add .

## 2. Commit med beskrivende besked (brug konventioner)
git commit -m "feat: Tilføjet backup-test, dagsmappe-backup og auto-oprydning"

## 3. (Ekstra) Skriv en mere detaljeret besked
# Tryk ENTER efter din commit-besked for at tilføje flere linjer:
#
# feat: Tilføjet backup-test, dagsmappe-backup og auto-oprydning
#
# - Tilføjet unittest for backup-funktion
# - Backup-mappe nu med dagsstruktur
# - Automatisk dummy-fil og oprydning af gamle backups
# - Forbedret teststruktur (tests/test_backup.py)
# 
# [refs #nummer hvis du bruger GitHub Issues]

## 4. Push til korrekt branch (fx dev)
git push origin ai_bot_dev

---------------------------------------------
# 🧠 Ekstra tip:
- Commit ofte, men med mening: Hver commit skal kunne forklares!
- Brug branches konsekvent (fx ai_bot_dev, ai_bot_test, ai_trading_pro)
- Husk at merge dev → test → prod når du har testet!
- Skriv altid “hvorfor” i din commit-besked – ikke kun “hvad”
- Brug GitHub Actions til auto-test og auto-backup (du har allerede CI workflows)
- Review evt. dine commits på github.com før merge til prod
- Brug CHANGELOG.md – gerne automatisk hvis muligt
- Hold .env og credentials ude af git (brug .gitignore)
- Commit både kode og tests!
- Tag evt. screenshot af workflows/commits til dokumentation

---------------------------------------------




## CI/CD Milestones og Merge-flow

1. **Løbende udvikling i `ai_bot_dev`**
    - Alt nyt udvikles og testes i dev-branchen.
2. **Milestone: Merge til TEST**
    - Når delmål/kritiske funktioner er testet og stabile → merge til `ai_bot_test` via Pull Request.
    - Kør alle unittests og CI/CD workflows på test.
3. **Milestone: Merge til PROD**
    - Når ALT er godkendt på test, og der er grønt lys på CI → merge til `ai_trading_pro`.
    - Opdater changelog, tag version, og lav release.
4. **Ekstra:**
    - Feature freeze før prod-merge.
    - Automatisk backup og notifikationer.
    - Dokumentér alle større merges i BotStatus.md.



### Robust Test Plan for AI Trading Bot

1. Fejlhåndtering af kritiske funktioner
    - Backup: Fejl i os.makedirs, shutil.copy2, shutil.copytree
    - Status/Changelog: Fejl i open()
    - Telegram: Fejl i send_telegram_message

2. Telegram-funktion med Mock
    - Korrekt POST payload og endpoint
    - Fejlhåndtering af API-fejl og exceptions

3. .env/config-indlæsning
    - Test korrekte/manglende variabler
    - Forventede advarsler eller fejlhåndtering

4. Edge-cases for cleanup/backup
    - Cleanup uden gamle backups
    - Backup uden enkelte eller alle mapper/filer




