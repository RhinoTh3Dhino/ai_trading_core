# ensemble/ensemble_predict.py

import numpy as np

def ensemble_predict(
    ml_preds, 
    dl_preds, 
    rule_preds=None, 
    extra_preds=None, 
    weights=None, 
    voting="majority", 
    debug=False
):
    """
    Kombinér ML, DL (og evt. rule-based og ekstra) predictions til ensemble-voting.

    Args:
        ml_preds (array-like): ML predictions (0/1 eller -1/1)
        dl_preds (array-like): DL predictions (0/1 eller -1/1)
        rule_preds (array-like, optional): Rule-based signal (0/1, -1/1 eller None)
        extra_preds (list, optional): Liste over ekstra signal-arrays (fx andre modeller)
        weights (list, optional): Weights pr. model/signal
        voting (str): 'majority', 'weighted', 'sum'
        debug (bool): Udskriv ekstra information

    Returns:
        np.array: Ensemble-voting output (0/1 eller -1/1 afhængig af input)
    """
    # Saml alle predictions i én liste
    all_preds = [ml_preds, dl_preds]
    if rule_preds is not None:
        all_preds.append(rule_preds)
    if extra_preds is not None:
        all_preds.extend(extra_preds)
    
    # Ensret alle til 1D np.array, cast til int8
    all_preds = [np.asarray(p).flatten().astype(np.int8) for p in all_preds]
    n = len(all_preds[0])
    for arr in all_preds:
        if len(arr) != n:
            raise ValueError(f"Alle input-arrays skal have samme længde! ({[len(a) for a in all_preds]})")
    preds = np.column_stack(all_preds)

    if debug:
        print("[Ensemble] Ensemble input matrix:\n", preds)

    # Default weights
    if weights is None:
        weights = [1.0] * preds.shape[1]
    weights = np.array(weights, dtype=np.float32)
    if len(weights) != preds.shape[1]:
        raise ValueError(f"Antal weights ({len(weights)}) matcher ikke antal input ({preds.shape[1]})")
    
    # Normaliser til 0/1 hvis der er -1/1 input
    input_has_neg = np.any(preds == -1)
    if input_has_neg:
        preds_bin = (preds > 0).astype(int)
    else:
        preds_bin = preds.copy()

    # === VOTING LOGIK ===
    if voting == "majority":
        # Klassisk: hver model tæller 1 (eller vægt), output = 1 hvis >50% stemmer for
        votes = np.round(preds_bin * weights).astype(int)
        summed = np.sum(votes, axis=1)
        threshold = 0.5 * np.sum(weights)
        if debug:
            print(f"[Ensemble] Weights: {weights}, summed votes: {summed}, threshold: {threshold}")
        result = (summed > threshold).astype(int)
    elif voting == "weighted":
        # Brug vægtet gennemsnit af alle modeller
        score = np.average(preds_bin, axis=1, weights=weights)
        if debug:
            print(f"[Ensemble] Weighted scores: {score}")
        result = (score > 0.5).astype(int)
    elif voting == "sum":
        # Samlet signal: sum(preds*weights), brug evt. -1/1 logik
        summed = np.sum(preds * weights, axis=1)
        if debug:
            print(f"[Ensemble] Summed signals (inkl. -1/1): {summed}")
        result = (summed > 0).astype(int)
    else:
        raise ValueError("Ukendt voting-type: %s" % voting)

    # Hvis original input var -1/1, tillad -1/1 output (valgfrit)
    if input_has_neg and not np.all(np.isin(result, [0,1])):
        result = np.where(result == 0, -1, 1)
    return result

# === CLI-test ===
if __name__ == "__main__":
    # Dummy test for både 0/1 og -1/1 logik
    ml = np.array([1, 0, 1, 0])
    dl = np.array([1, 1, 0, 0])
    rsi = np.array([0, 0, 1, 1])
    print("Majority voting:", ensemble_predict(ml, dl, rsi, voting="majority", debug=True))
    print("Weighted voting:", ensemble_predict(ml, dl, rsi, weights=[1,1,0.7], voting="weighted", debug=True))
    print("Sum voting:", ensemble_predict(ml, dl, rsi, weights=[1,1,1], voting="sum", debug=True))
    # Test med -1/1
    print("Majority voting -1/1:", ensemble_predict(np.array([1,-1,1,-1]), np.array([1,1,-1,-1]), np.array([-1,-1,1,1]), voting="majority", debug=True))
