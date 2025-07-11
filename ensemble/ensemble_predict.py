# ensemble/ensemble_predict.py

import numpy as np

def ensemble_predict(ml_preds, dl_preds, rule_preds=None, weights=None, voting="majority", debug=False):
    """
    Kombinér ML, DL (og evt. rule-based) prediction arrays til ensemble-voting.
    
    Args:
        ml_preds (array-like): scikit-learn/klassisk ML predictions (0/1)
        dl_preds (array-like): PyTorch DL predictions (0/1)
        rule_preds (array-like, optional): Rule-based signal (0/1), fx RSI eller MACD
        weights (list, optional): Liste med weights til hver model (fx [1.0, 1.0, 0.7])
        voting (str): 'majority' eller 'weighted'
        debug (bool): Udskriv ekstra information
        
    Returns:
        np.array: Ensemble-voting output (0/1)
    """
    # --- Input-validering ---
    all_preds = [ml_preds, dl_preds]
    if rule_preds is not None:
        all_preds.append(rule_preds)
    # Ensret til arrays
    all_preds = [np.asarray(p).flatten() for p in all_preds]
    n = len(all_preds[0])
    for arr in all_preds:
        if len(arr) != n:
            raise ValueError(f"Alle input-arrays skal have samme længde! ({[len(a) for a in all_preds]})")
    preds = np.column_stack(all_preds)
    if debug:
        print("[Ensemble] Ensemble input matrix:\n", preds)
    if weights is None:
        weights = [1.0] * preds.shape[1]

    # --- Majority voting ---
    if voting == "majority":
        votes = np.round(preds * weights).astype(int)
        summed = np.sum(votes, axis=1)
        threshold = 0.5 * np.sum(weights)
        if debug:
            print(f"[Ensemble] Weights: {weights}, summed votes: {summed}, threshold: {threshold}")
        return (summed > threshold).astype(int)
    # --- Weighted average voting ---
    elif voting == "weighted":
        score = np.average(preds, axis=1, weights=weights)
        if debug:
            print(f"[Ensemble] Weighted scores: {score}")
        return (score > 0.5).astype(int)
    else:
        raise ValueError("Ukendt voting-type: %s" % voting)

# === Eksempel på CLI-test (kan fjernes i produktion) ===
if __name__ == "__main__":
    # Dummy test
    ml = np.array([1, 0, 1, 0])
    dl = np.array([1, 1, 0, 0])
    rsi = np.array([0, 0, 1, 1])
    print("Majority voting:", ensemble_predict(ml, dl, rsi, voting="majority", debug=True))
    print("Weighted voting:", ensemble_predict(ml, dl, rsi, weights=[1,1,0.7], voting="weighted", debug=True))
