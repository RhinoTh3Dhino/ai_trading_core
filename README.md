# AI Trading Core

Dette projekt er fundamentet for en avanceret AI trading bot.



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
