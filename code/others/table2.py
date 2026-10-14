# -*- coding: utf-8 -*-
"""
Reproduce Table 2 (Means of Key Variables) of Card & Krueger (1994 AER).
Reads ./data/processed/cleaned_main.csv (read-only).
Fix: Percentage dummy variables (wage share, bonus) multiply by 100.
PENTREE no longer multiplied by 10.
"""
import os
import numpy as np
import pandas as pd

# 相对路径配置，工作目录：C:\Users\LHK\Desktop\Li_Haoke_3036763933
PROC = os.path.join("data", "processed")
OUTPUT = os.path.join("output")
os.makedirs(OUTPUT, exist_ok=True)

df = pd.read_csv(os.path.join(PROC, "cleaned_main.csv"))
# ---------------------------------------------------------------------------
# helper functions
# ---------------------------------------------------------------------------
def mean_se(s):
    s = pd.Series(s).dropna().astype(float)
    n = len(s)
    if n == 0:
        return np.nan, np.nan, 0
    return float(s.mean()), float(s.std(ddof=1)/np.sqrt(n)), n
def welch_t(s1, s2):
    s1 = pd.Series(s1).dropna()
    s2 = pd.Series(s2).dropna()
    n1, n2 = len(s1), len(s2)
    if n1<2 or n2<2:
        return np.nan
    v1, v2 = s1.var(ddof=1), s2.var(ddof=1)
    se = np.sqrt(v1/n1 + v2/n2)
    return float((s1.mean() - s2.mean()) / se)
# ---------------------------------------------------------------------------
# derived variables
# ---------------------------------------------------------------------------
def pct(numer, denom):
    return np.where(denom>0, numer/denom*100, np.nan)
df["pctft"] = pct(df["empft"], df["empft"] + df["emppt"])
df["pctft2"] = pct(df["empft2"], df["empft2"] + df["emppt2"])
# wage dummy: use ±0.01 tolerance
def wage_flag(col, target_val, eps=0.01):
    s = pd.to_numeric(df[col], errors="coerce")
    cond = (s >= target_val - eps) & (s <= target_val + eps)
    return np.where(s.notna(), cond.astype(float), np.nan)
df["wage_st_425"] = wage_flag("wage_st", 4.25)
df["wage_st2_425"] = wage_flag("wage_st2", 4.25)
df["wage_st2_505"] = wage_flag("wage_st2", 5.05)
# chain / ownership dummies
df["bk"] = (df["chain"] ==1).astype(float)
df["kfc"] = (df["chain"] ==2).astype(float)
df["rr"] = (df["chain"] ==3).astype(float)
df["wendys"] = (df["chain"] ==4).astype(float)
df["coowned"] = (df["co_owned"] ==1).astype(float)
# ---------------------------------------------------------------------------
# ROW definition:
# ("SectionFlag", Label, nj_col, pa_col, is_percentage, dec_mean, dec_se, t_dash, note)
# is_percentage=True → multiply mean by 100
# ---------------------------------------------------------------------------
ROWS = [
    ("S", "1. Distribution of Store Types (percentages):", None, None, None, None, None, None, None),
    ("r", "a. Burger King", "bk", "bk", True, 1, 1, None, None),
    ("r", "b. KFC", "kfc", "kfc", True, 1, 1, None, None),
    ("r", "c. Roy Rogers", "rr", "rr", True, 1, 1, None, None),
    ("r", "d. Wendy's", "wendys", "wendys", True, 1, 1, None, None),
    ("r", "e. Company-owned", "coowned", "coowned", True, 1, 1, None, None),
    ("S", "2. Means in Wave 1:", None, None, None, None, None, None, None),
    ("r", "a. FTE employment", "fte1", "fte1", False, 1, 2, None, None),
    ("r", "b. Percentage full-time employees", "pctft", "pctft", False, 1, 1, None, None),
    ("r", "c. Starting wage", "wage_st", "wage_st", False, 2, 2, None, None),
    ("r", "d. Wage = $4.25 (percentage)", "wage_st_425", "wage_st_425", True, 1, 1, None, None),
    ("r", "e. Price of full meal", "pentree", "pentree", False, 2, 2, None, None),
    ("r", "f. Hours open (weekday)", "hrsopen", "hrsopen", False, 1, 1, None, None),
    ("r", "g. Recruiting bonus", "bonus", "bonus", True, 1, 1, None, None),
    ("S", "3. Means in Wave 2:", None, None, None, None, None, None, None),
    ("r", "a. FTE employment", "fte2", "fte2", False, 1, 2, None, None),
    ("r", "b. Percentage full-time employees", "pctft2", "pctft2", False, 1, 1, None, None),
    ("r", "c. Starting wage", "wage_st2", "wage_st2", False, 2, 2, None, None),
    ("r", "d. Wage = $4.25 (percentage)", "wage_st2_425", "wage_st2_425", True, 1, 1, "dash", None),
    ("r", "e. Wage = $5.05 (percentage)", "wage_st2_505", "wage_st2_505", True, 1, 1, None, None),
    ("r", "f. Price of full meal", "pentree2", "pentree2", False, 2, 2, None, None),
    ("r", "g. Hours open (weekday)", "hrsopen2", "hrsopen2", False, 1, 1, None, None),
    ("r", "h. Recruiting bonus", None, None, None, None, None, None, "bonus2 (wave-2 recruiting bonus) is not present in cleaned_main.csv"),
]
# ---------------------------------------------------------------------------
# Print table + capture output to save md
# ---------------------------------------------------------------------------
import sys
from io import StringIO
old_stdout = sys.stdout
sys.stdout = StringIO()

print("TABLE 2 - MEANS OF KEY VARIABLES")
print("="*80)
print(f"{'Variable':<38}{'NJ':>16}{'PA':>16}{'t^a':>9}")
print("-"*80)
for row in ROWS:
    flag, label, nj_col, pa_col, is_pct, dec_m, dec_se, t_dash, note = row
    if flag == "S":
        print(label)
        continue
    if nj_col is None:
        print(f"{label:<38}{'-':>16}{'-':>16}{'-':>9}   <- {note}")
        continue
    # extract group data
    s_nj = df.loc[df["nj"] == 1, nj_col]
    s_pa = df.loc[df["nj"] == 0, pa_col]
    mnj, senj, _ = mean_se(s_nj)
    mpa, sepa, _ = mean_se(s_pa)
    tval = np.nan if t_dash=="dash" else welch_t(s_nj, s_pa)
    if is_pct:
        mnj = mnj *100
        senj = senj *100
        mpa = mpa *100
        sepa = sepa *100
    str_nj = f"{mnj:.{dec_m}f} ({senj:.{dec_se}f})"
    str_pa = f"{mpa:.{dec_m}f} ({sepa:.{dec_se}f})"
    str_t = "\u2014" if (t_dash=="dash" or pd.isna(tval)) else f"{tval:.1f}"
    print(f"{label:<38}{str_nj:>16}{str_pa:>16}{str_t:>9}")
print("-"*80)
print("Notes: Standard errors are in parentheses. t^a is a two-sample unequal variance t-test for equality of means.")
print("\nDone.")

# save markdown
output_text = sys.stdout.getvalue()
sys.stdout = old_stdout
with open(os.path.join(OUTPUT, "table2_result.md"), "w", encoding="utf-8") as f:
    f.write(output_text)
print("\n✅ Table2 result saved to output/table2_result.md")

