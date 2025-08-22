# ensemble/ensemble_predict.py

import numpy as np


def _to_1d_int_array(x):
    """Ensret input til 1D np.array af dtype int8."""
    return np.asarray(x).flatten().astype(np.int8)


def ensemble_predict(
    ml_preds,
    dl_preds,
    rule_preds=None,
    extra_preds=None,
    weights=None,
    voting="majority",
    debug=False,
):
    """
    Kombinér ML, DL (og evt. rule-based og ekstra) predictions til ensemble-voting.

    Args:
        ml_preds (array-like): ML predictions (0/1 eller -1/1)
        dl_preds (array-like): DL predictions (0/1 eller -1/1)
        rule_preds (array-like, optional): Rule-based signal (0/1, -1/1 eller None)
        extra_preds (array-like | list[array-like], optional):
            Enten et enkelt ekstra signal-array ELLER en liste/tuple af ekstra signal-arrays.
        weights (list[float], optional): Weights pr. model/signal
        voting (str): 'majority', 'weighted', 'sum'
        debug (bool): Udskriv ekstra information

    Returns:
        np.ndarray: Ensemble-voting output (0/1; hvis input indeholder -1/1 kan 'sum' give -1/1 før binarisering)
    """
    # Saml alle predictions i én liste
    all_preds = [ml_preds, dl_preds]
    if rule_preds is not None:
        all_preds.append(rule_preds)

    # --- Robust håndtering af extra_preds: tillad både enkelt array og liste/tuple af arrays
    if extra_preds is not None:
        if isinstance(extra_preds, (list, tuple)):
            all_preds.extend(extra_preds)
        else:
            # Ét enkelt array/sekvens → wrap i liste
            all_preds.append(extra_preds)

    # Ensret alle til 1D np.array (int8)
    all_preds = [_to_1d_int_array(p) for p in all_preds]

    # Valider identisk længde
    n = len(all_preds[0])
    for arr in all_preds:
        if len(arr) != n:
            raise ValueError(
                f"Alle input-arrays skal have samme længde! ({[len(a) for a in all_preds]})"
            )

    preds = np.column_stack(all_preds)

    if debug:
        print("[Ensemble] Input matrix shape:", preds.shape)
        print("[Ensemble] Input matrix (head):\n", preds[:5])

    # Default weights
    if weights is None:
        weights = [1.0] * preds.shape[1]
    weights = np.asarray(weights, dtype=np.float32)

    if len(weights) != preds.shape[1]:
        raise ValueError(
            f"Antal weights ({len(weights)}) matcher ikke antal input ({preds.shape[1]})"
        )

    # Normaliser til 0/1 hvis der er -1/1 input
    input_has_neg = np.any(preds == -1)
    preds_bin = (preds > 0).astype(int) if input_has_neg else preds.copy()

    # === VOTING LOGIK ===
    if voting == "majority":
        # Hver model giver en stemme (evt. vægtet). Tærskel = 50% af sum(weights).
        votes = np.round(preds_bin * weights).astype(int)
        summed = np.sum(votes, axis=1)
        threshold = 0.5 * np.sum(weights)
        if debug:
            print(f"[Ensemble] Weights: {weights}")
            print(f"[Ensemble] Summed votes (head): {summed[:10]}")
            print(f"[Ensemble] Threshold: {threshold}")
        result = (summed > threshold).astype(int)

    elif voting == "weighted":
        # Brug vægtet gennemsnit af de binære stemmer
        score = np.average(preds_bin, axis=1, weights=weights)
        if debug:
            print(f"[Ensemble] Weighted scores (head): {score[:10]}")
        result = (score > 0.5).astype(int)

    elif voting == "sum":
        # Samlet signal: sum(preds * weights) – bevarer evt. -1/1 logik i input
        summed = np.sum(preds * weights, axis=1)
        if debug:
            print(f"[Ensemble] Summed signals (head): {summed[:10]}")
        # Output 0/1 for consistency med de fleste pipelines
        result = (summed > 0).astype(int)

    else:
        raise ValueError("Ukendt voting-type: %s" % voting)

    # Hvis du senere ønsker -1/1 output, kan det håndteres her.
    # (P.t. returnerer vi konsekvent 0/1, hvilket matcher tests og typiske pipelines.)
    return result


# === CLI-test ===
if __name__ == "__main__":
    # Dummy test for både 0/1 og -1/1 logik
    ml = np.array([1, 0, 1, 0])
    dl = np.array([1, 1, 0, 0])
    rsi = np.array([0, 0, 1, 1])

    print("Majority voting:", ensemble_predict(ml, dl, rsi, voting="majority", debug=True))
    print(
        "Weighted voting:",
        ensemble_predict(ml, dl, rsi, weights=[1, 1, 0.7], voting="weighted", debug=True),
    )
    print(
        "Sum voting:",
        ensemble_predict(ml, dl, rsi, weights=[1, 1, 1], voting="sum", debug=True),
    )

    # Test med -1/1
    print(
        "Majority voting -1/1:",
        ensemble_predict(
            np.array([1, -1, 1, -1]),
            np.array([1, 1, -1, -1]),
            np.array([-1, -1, 1, 1]),
            voting="majority",
            debug=True,
        ),
    )
