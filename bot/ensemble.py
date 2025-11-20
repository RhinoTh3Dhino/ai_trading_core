# bot/ensemble.py


def ensemble_predict(predictions):
    """
    Returnerer majoritetssignalet fra en liste af forudsigelser.
    Hvis der er tie eller tom liste, returneres 0 (HOLD).
    predictions: list af ints, fx [1, 0, 1, -1]
    Return: int (1=BUY, 0=HOLD, -1=SELL)
    """
    if not predictions:
        return 0  # HOLD hvis tom liste

    # Tæl forekomster af hver signaltype
    votes = {}
    for p in predictions:
        votes[p] = votes.get(p, 0) + 1

    # Find majoriteten (højeste stemmetal)
    max_votes = max(votes.values())
    majorities = [k for k, v in votes.items() if v == max_votes]

    # Hvis der er én majoritet, returnér den
    if len(majorities) == 1:
        return majorities[0]
    # Hvis tie (fx [1, -1] eller [1, 0, -1]), returnér 0 (HOLD)
    return 0


# (Valgfrit) Udvid med vægtet voting – eksempel:
def weighted_ensemble_predict(predictions, weights):
    """
    Vægtet voting: hver prediction vægtes med weights[i].
    predictions: list af ints
    weights: list af floats, samme længde som predictions
    Return: int (1=BUY, 0=HOLD, -1=SELL)
    """
    if not predictions or not weights or len(predictions) != len(weights):
        return 0
    weighted = {}
    for p, w in zip(predictions, weights):
        weighted[p] = weighted.get(p, 0) + w
    max_weight = max(weighted.values())
    majorities = [k for k, v in weighted.items() if v == max_weight]
    if len(majorities) == 1:
        return majorities[0]
    return 0


# Eksempel på brug
if __name__ == "__main__":
    print(ensemble_predict([1, 1, 0, 1, 0]))  # -> 1
    print(ensemble_predict([0, 0, 1, -1, 0]))  # -> 0
    print(ensemble_predict([-1, -1, 0, -1, 1]))  # -> -1
    print(ensemble_predict([1, -1, 1, -1]))  # -> 0 (tie)
    print(ensemble_predict([]))  # -> 0 (empty)

    # Vægtet voting-eksempel
    print(weighted_ensemble_predict([1, -1, 1], [0.7, 0.2, 0.5]))  # -> 1
