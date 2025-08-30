from utils.alerts import AlertManager, AlertThresholds

def test_alert_cooldown():
    am = AlertManager(AlertThresholds(dd_pct=5, winrate_min=90, profit_pct=1, cooldown_s=3600), allow_alerts=True)
    msgs=[]
    def send(kind, text): msgs.append((kind,text))
    am.on_equity(100); am.on_equity(94.9)  # ~ -5.1% DD
    am.evaluate_and_notify(send)
    am.evaluate_and_notify(send)  # anden gang skal ikke sende grundet cooldown
    assert len(msgs) == 1
