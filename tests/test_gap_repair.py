from data.gap_repair import rest_catchup


def test_rest_catchup_signature_only():
    # smoke: funktion kan kaldes (ikke hitting real API i CI uden key)
    assert callable(rest_catchup)
