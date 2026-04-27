import pandas as pd

def check_bias(df, sensitive_col, decision_col):
    groups = df[sensitive_col].unique()

    if len(groups) != 2:
        return {"error": "Need exactly 2 groups (e.g., Male/Female)"}



    # Calculate approval rates for each group
    g1_name = groups[0]
    g2_name = groups[1]
    
    rate1 = df[df[sensitive_col] == g1_name][decision_col].mean()
    rate2 = df[df[sensitive_col] == g2_name][decision_col].mean()



    # Disparate Impact Ratio
    # We divide the lower rate by the higher rate
    ratio = min(rate1, rate2) / max(rate1, rate2) if max(rate1, rate2) > 0 else 1.0

    return {
        "group1": g1_name,
        "group2": g2_name,
        "rate1": rate1,
        "rate2": rate2,
        "ratio": ratio,
        "biased": ratio < 0.8  # True if below 80%
    }