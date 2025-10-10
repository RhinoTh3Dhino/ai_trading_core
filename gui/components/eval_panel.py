import json
from pathlib import Path

import streamlit as st


def render(eval_path="outputs/evals/last_eval.json"):
    st.subheader("LLM-evaluering (smoke)")
    p = Path(eval_path)
    if not p.exists():
        st.info("Ingen evaluering fundet endnu.")
        return
    data = json.loads(p.read_text(encoding="utf-8"))
    c1, c2, c3 = st.columns(3)
    c1.metric("Edge Score", data["edge_score"])
    c2.metric("Action", data["actionability"])
    c3.metric("Confidence", f"{data['confidence']:.2f}")
    st.json(data)
