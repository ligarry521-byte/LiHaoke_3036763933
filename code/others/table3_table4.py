# -*- coding: utf-8 -*-
"""
Table3 & Table4 replication for Card & Krueger (1994 AER)
Only generate markdown table file, saved to outputs/tables/
"""
import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from io import StringIO

# 读取全局路径（run_all.py调用时会自动传入，单独运行也兼容）
PROC_DIR = os.path.join("data", "processed")
TABLE_DIR = os.path.join("outputs", "tables")
os.makedirs(TABLE_DIR, exist_ok=True)

def run_table3_table4():
    print("\n" + "="*60)
    print("STEP 3 / 5 : Table 3 & Table 4")
    print("="*60)
    main = pd.read_csv(os.path.join(PROC_DIR, "cleaned_main.csv"))
    robust = pd.read_csv(os.path.join(PROC_DIR, "cleaned_robust.csv"))
    def mean_se(s):
        s = pd.Series(s).dropna()
        n = int(s.shape[0])
        if n == 0:
            return np.nan, np.nan, 0
        return float(s.mean()), float(s.std(ddof=1) / np.sqrt(n)), n
    def diff_se(s1, s2):
        s1 = s1.astype(float)
        s2 = s2.astype(float)
        both = s1.notna() & s2.notna()
        n1 = int(s1.notna().sum())
        n2 = int(s2.notna().sum())
        no = int(both.sum())
        if n1 == 0 or n2 == 0:
            return np.nan
        v1 = s1.var(ddof=1) / n1
        v2 = s2.var(ddof=1) / n2
        cov = float(s1[both].cov(s2[both])) if no > 1 else 0.0
        var = v1 + v2 - 2.0 * cov * no / (n1 * n2)
        return float(np.sqrt(max(var, 0.0)))
    def group_rows(mask):
        m = main[mask]
        r = robust[mask]
        out = {}
        m1, s1, n1 = mean_se(m["fte1"])
        m2, s2, n2 = mean_se(m["fte2"])
        out["before"] = (m1, s1, n1)
        out["after"] = (m2, s2, n2)
        out["change"] = (m2 - m1, diff_se(m["fte1"], m["fte2"]), None)
        bal = m[m["balanced_sample"] == 1]
        d4 = bal["fte2"] - bal["fte1"]
        out["bal"] = mean_se(d4)
        balr = r[r["balanced_sample"] == 1]
        d5 = balr["fte2"] - balr["fte1"]
        out["temp0"] = mean_se(d5)
        return out
    wage = main["wage_st"].round(2)
    nj = main["nj"] == 1
    masks = {
        "PA":             main["nj"] == 0,
        "NJ":             nj,
        "NJ w=4.25":      nj & (wage == 4.25),
        "NJ 4.26-4.99":   nj & (wage >= 4.26) & (wage <= 4.99),
        "NJ >=5.00":      nj & (wage >= 5.00),
    }
    G = {name: group_rows(msk) for name, msk in masks.items()}
    def diff_group(ga, gb):
        return {
            k: (ga[k][0] - gb[k][0], np.sqrt(ga[k][1] ** 2 + gb[k][1] ** 2), None)
            for k in ["before", "after", "change", "bal", "temp0"]
        }
    G["NJ-PA"] = diff_group(G["NJ"], G["PA"])
    G["low-high"] = diff_group(G["NJ w=4.25"], G["NJ >=5.00"])
    G["mid-high"] = diff_group(G["NJ 4.26-4.99"], G["NJ >=5.00"])
    ROWS = ["before", "after", "change", "bal", "temp0"]
    ROW_LABELS = [
        "1. FTE before (all available)",
        "2. FTE after  (all available)",
        "3. Change in mean FTE",
        "4. Change, balanced sample",
        "5. Change, temp-closed = 0",
    ]
    COLS = ["PA", "NJ", "NJ-PA", "NJ w=4.25", "NJ 4.26-4.99", "NJ >=5.00",
            "low-high", "mid-high"]
    def fmt(tup):
        m, s, _ = tup
        if np.isnan(m):
            return "      -      "
        return f"{m:7.2f} ({s:.2f})"
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    print("=" * 100)
    print("TABLE 3  -  AVERAGE EMPLOYMENT PER STORE BEFORE AND AFTER THE RISE")
    print("           IN THE NEW JERSEY MINIMUM WAGE")
    print("=" * 100)
    header = f"{'Variable':<32}" + "".join(f"{c:>20}" for c in COLS)
    print(header)
    print("-" * len(header))
    for rlab, r in zip(ROW_LABELS, ROWS):
        line = f"{rlab:<32}" + "".join(f"{fmt(G[c][r]):>20}" for c in COLS)
        print(line)
    print("-" * len(header))
    print("\nNJ wage-subgroup observation counts (wave-1 wage):")
    for name in ["NJ w=4.25", "NJ 4.26-4.99", "NJ >=5.00"]:
        print(f"  {name:<16}: N = {int(masks[name].sum())}")
    print("\n" + "=" * 100)
    print("TABLE 3 CROSS-CHECK vs PAPER (PA / NJ / NJ-PA columns)")
    print("=" * 100)
    paper = {
        "before": {"PA": (23.33, 1.35), "NJ": (20.44, 0.51), "NJ-PA": (-2.89, 1.44)},
        "after":  {"PA": (21.17, 0.94), "NJ": (21.03, 0.52), "NJ-PA": (-0.14, 1.07)},
        "change": {"PA": (-2.16, 1.25), "NJ": (0.59, 0.54),  "NJ-PA": (2.76, 1.36)},
        "bal":    {"PA": (-2.28, 1.25), "NJ": (0.47, 0.48),  "NJ-PA": (2.75, 1.35)},
        "temp0":  {"PA": (-2.28, 1.25), "NJ": (0.23, 0.48),  "NJ-PA": (2.51, 1.35)},
    }
    print(f"{'row':<10}{'col':<8}{'paper':>16}{'mine':>16}{'d_mean':>10}{'d_se':>9}")
    for rlab, r in zip(ROW_LABELS, ROWS):
        for c in ["PA", "NJ", "NJ-PA"]:
            pm, ps = paper[r][c]
            mm, ms, _ = G[c][r]
            print(f"{rlab.split('.')[0]:<10}{c:<8}{pm:>9.2f} ({ps:.2f})"
                  f"{mm:>9.2f} ({ms:.2f})"
                  f"{mm - pm:>+10.2f}{ms - ps:>+9.2f}")
    did_m, did_s, _ = G["NJ-PA"]["change"]
    print("\n>>> Difference-in-differences (Table 3, row 3, NJ-PA):")
    print(f"    mine = {did_m:.4f} ({did_s:.4f})   |   paper = 2.76 (1.36), t = 2.03")
    # ===== Table 4 =====
    reg = main[main["balanced_sample"] == 1].copy()
    reg = reg[reg["wage_st"].notna() &
              (reg["wage_st2"].notna() | (reg["closed"] == 1))].copy()
    reg["chain2"] = (reg["chain"] == 2).astype(int)
    reg["chain3"] = (reg["chain"] == 3).astype(int)
    reg["chain4"] = (reg["chain"] == 4).astype(int)
    y = reg["delta_fte"].astype(float)
    print("\n" + "=" * 100)
    print("TABLE 4  -  REDUCED-FORM MODELS FOR CHANGE IN EMPLOYMENT")
    print("=" * 100)
    print(f"Sample size            : {len(reg)}  (paper: 357)")
    print(f"Dep-var mean / sd      : {y.mean():.3f} / {y.std(ddof=1):.3f}"
          f"  (paper: -0.237 / 8.825)")
    def ols_model(key, cols):
        X = reg[cols].astype(float)
        X = sm.add_constant(X)
        fit = sm.OLS(y, X).fit()
        return fit.params[key], fit.bse[key], np.sqrt(fit.mse_resid)
    models = {
        "(i)":   ("nj",  ["nj"]),
        "(ii)":  ("nj",  ["nj", "chain2", "chain3", "chain4", "co_owned"]),
        "(iii)": ("gap", ["gap"]),
        "(iv)":  ("gap", ["gap", "chain2", "chain3", "chain4", "co_owned"]),
        "(v)":   ("gap", ["gap", "chain2", "chain3", "chain4", "co_owned",
                          "southj", "centralj", "pa1", "pa2"]),
    }
    paper_t4 = {
        "(i)":   (2.33, 1.19),
        "(ii)":  (2.30, 1.20),
        "(iii)": (15.65, 6.08),
        "(iv)":  (14.92, 6.21),
        "(v)":   (11.91, 7.39),
    }
    print(f"\n{'model':<6}{'key':<5}{'coef':>9}{'se':>8}{'paper coef':>12}"
          f"{'paper se':>10}{'d_coef':>9}{'d_se':>8}{'resid SE':>10}")
    for m, (key, cols) in models.items():
        coef, se, resid_se = ols_model(key, cols)
        pc, ps = paper_t4[m]
        print(f"{m:<6}{key:<5}{coef:>9.3f}{se:>8.3f}{pc:>12.2f}{ps:>10.2f}"
              f"{coef - pc:>+9.2f}{se - ps:>+8.2f}{resid_se:>10.3f}")
    print("\nDone.")
    output_text = sys.stdout.getvalue()
    sys.stdout = old_stdout
    print(output_text)
    out_path = os.path.join(TABLE_DIR, "table3_table4_result.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output_text)
    print(f"✅ Saved: {out_path}")

if __name__ == "__main__":
    run_table3_table4()
