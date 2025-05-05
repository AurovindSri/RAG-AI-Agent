#!/usr/bin/env python3
"""
Evaluate non‑RAG vs. RAG answers with BERTScore (per‑question & averages).

Folder layout expected:
    gold.json                     # {"Q1": "correct answer", ...}
    rag_comparison_results.csv    # must contain columns: qid, rag_answer, baseline_answer
Run:
    python bert_eval.py
"""

import json, pandas as pd
from bert_score import score

# ── config ──────────────────────────────────────────────────────────────────── #
CSV_PATH   = "rag_comparison_results.csv"
GOLD_PATH  = "gold.json"
MODEL_NAME = "microsoft/deberta-large-mnli"   # strong English model
LANG       = "en"                             # BERTScore language flag
OUT_CSV    = "bertscore_results.csv"

# If your CSV has different column names, edit here:
QID_COL        = "qid"
RAG_COL        = "rag_answer"
BASELINE_COL   = "baseline_answer"
# ────────────────────────────────────────────────────────────────────────────── #

def main():
    # 1️⃣  load files
    df   = pd.read_csv(CSV_PATH)
    gold = json.load(open(GOLD_PATH))

    # 2️⃣  ensure alignment / order
    if set(df[QID_COL]) != set(gold):
        raise ValueError("Mismatch between QIDs in CSV and gold.json")

    g_list   = [gold[qid] for qid in df[QID_COL]]
    rag_list = df[RAG_COL].fillna("").tolist()
    bl_list  = df[BASELINE_COL].fillna("").tolist()

    # 3️⃣  compute BERTScore F1 arrays (precision/recall ignored for brevity)
    print("Computing BERTScore … this may take 10‑20 s on CPU")
    _, _, rag_f = score(rag_list, g_list, lang=LANG, model_type=MODEL_NAME)
    _, _, bl_f  = score(bl_list,  g_list, lang=LANG, model_type=MODEL_NAME)

    # 4️⃣  attach to dataframe
    df["rag_bertscore"]       = rag_f.numpy()             # 0‑1
    df["baseline_bertscore"]  = bl_f.numpy()
    df["rag_grade_0‑10"]      = (df.rag_bertscore      * 10).round(2)
    df["baseline_grade_0‑10"] = (df.baseline_bertscore * 10).round(2)

    # 5️⃣  console report
    print("\nPer‑question BERTScore (F1 → 0‑10 scale)")
    print(df[[QID_COL, "rag_grade_0‑10", "baseline_grade_0‑10"]]
            .to_string(index=False))

    print("\nAverages")
    print(f"  RAG      : {df['rag_bertscore'].mean():.3f}  ({df['rag_grade_0‑10'].mean():.2f}/10)")
    print(f"  non‑RAG  : {df['baseline_bertscore'].mean():.3f}  ({df['baseline_grade_0‑10'].mean():.2f}/10)")

    # 6️⃣  save enriched CSV
    df.to_csv(OUT_CSV, index=False)
    print(f"\nDetailed results written to {OUT_CSV}")

if __name__ == "__main__":
    main()