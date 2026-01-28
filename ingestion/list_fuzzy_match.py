import pandas as pd
import re
from rapidfuzz import fuzz, distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Normalization utilities
# -----------------------------

LEGAL_SUFFIXES = [
    "ltd", "limited", "inc", "llc", "corp", "corporation", "co", "company",
    "gmbh", "sarl", "sa", "bv", "plc", "oy", "ag", "kg"
]

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    tokens = [t for t in tokens if t not in LEGAL_SUFFIXES]

    return " ".join(tokens)

def acronym(text: str) -> str:
    tokens = text.split()
    return "".join(t[0] for t in tokens if t)

# -----------------------------
# Similarity functions
# -----------------------------

def containment_score(a: str, b: str) -> float:
    if a in b or b in a:
        return min(len(a), len(b)) / max(len(a), len(b))
    return 0.0

def combined_score(a_raw, b_raw, a_norm, b_norm, a_acr, b_acr):
    token_sim = fuzz.token_set_ratio(a_norm, b_norm) / 100
    char_sim = distance.JaroWinkler.similarity(a_norm, b_norm)
    contain = containment_score(a_norm, b_norm)
    acr_sim = 1.0 if a_acr == b_norm or b_acr == a_norm or a_acr == b_acr else 0.0

    score = (
        0.35 * token_sim +
        0.25 * char_sim +
        0.20 * contain +
        0.20 * acr_sim
    )
    return round(score, 4)

# -----------------------------
# Blocking strategy
# -----------------------------

def blocking_key(text: str) -> str:
    return text[:4]  # first 4 chars

# -----------------------------
# Main matching function
# -----------------------------

def match_companies(csv_a, csv_b, output_csv):

    df_a = pd.read_csv(csv_a)
    df_b = pd.read_csv(csv_b)

    col_a = df_a.columns[0]
    col_b = df_b.columns[0]

    df_a["raw"] = df_a[col_a].astype(str)
    df_b["raw"] = df_b[col_b].astype(str)

    df_a["norm"] = df_a["raw"].apply(normalize)
    df_b["norm"] = df_b["raw"].apply(normalize)

    df_a["acr"] = df_a["norm"].apply(acronym)
    df_b["acr"] = df_b["norm"].apply(acronym)

    df_a["block"] = df_a["norm"].apply(blocking_key)
    df_b["block"] = df_b["norm"].apply(blocking_key)

    candidates = df_a.merge(df_b, on="block", suffixes=("_a", "_b"))

    results = []

    for _, row in candidates.iterrows():
        score = combined_score(
            row["raw_a"], row["raw_b"],
            row["norm_a"], row["norm_b"],
            row["acr_a"], row["acr_b"]
        )

        if score >= 0.6:  # adjustable threshold
            results.append({
                "company_a": row["raw_a"],
                "company_b": row["raw_b"],
                "score": score
            })

    result_df = pd.DataFrame(results)
    result_df.sort_values("score", ascending=False, inplace=True)

    # Keep best match per company_a
    result_df = result_df.groupby("company_a", as_index=False).first()

    result_df.to_csv(output_csv, index=False)
    print(f"Saved {len(result_df)} matches to {output_csv}")

# -----------------------------
# CLI usage
# -----------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("Usage: python match_companies.py file_a.csv file_b.csv output.csv")
        sys.exit(1)

    match_companies(sys.argv[1], sys.argv[2], sys.argv[3])
