# -*- coding: utf-8 -*-
"""
All-in-one replication script for Card & Krueger (1994 AER)
Pipeline: data cleaning → Table 2 → Table 3 & 4 → Figure 1 → Heterogeneous DID
Run from repository root folder: Li_Haoke_3036763933/
Command: python all_in_one.py
"""
import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from io import StringIO

# 强制 UTF-8 输出，解决 Windows GBK 编码报错
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding="utf-8")

# ===================== 全局路径配置 =====================
RAW_DIR = os.path.join("data", "raw")
PROC_DIR = os.path.join("data", "processed")
OUTPUT_DIR = os.path.join("output")
RAW_CSV = os.path.join(RAW_DIR, "public.csv")

# 自动创建文件夹
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== 1. 数据清洗 =====================
def run_clean():
    print("\n" + "="*60)
    print("STEP 1 / 5 : Data Cleaning")
    print("="*60)

    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(f"Raw data missing! Expected at {RAW_CSV}")

    df = pd.read_csv(RAW_CSV)
    df.columns = [c.strip().lower() for c in df.columns]
    print(f"[load] raw shape: {df.shape}")

    NUMERIC_COLS = [
        "empft", "emppt", "nmgrs", "wage_st",
        "empft2", "emppt2", "nmgrs2", "wage_st2",
    ]
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["state", "status2", "chain", "co_owned"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 构建核心指标
    df["nj"] = df["state"].astype("Int64")
    df["closed"] = (df["status2"] == 3).astype(int)
    df["temp_close"] = (df["status2"].isin([2, 4, 5])).astype(int)

    # FTE 计算
    df["fte1"] = df["empft"] + df["nmgrs"] + 0.5 * df["emppt"]
    df["fte2_raw"] = df["empft2"] + df["nmgrs2"] + 0.5 * df["emppt2"]
    df["fte2"] = df["fte2_raw"].copy()
    df.loc[df["closed"] == 1, "fte2"] = 0.0
    df.loc[df["temp_close"] == 1, "fte2"] = np.nan

    df["delta_fte"] = df["fte2"] - df["fte1"]

    # GAP 变量
    cond_pa = df["nj"] == 0
    cond_high = (df["nj"] == 1) & (df["wage_st"] >= 5.05)
    df["gap"] = np.select(
        [cond_pa, cond_high],
        [0.0, 0.0],
        default=(5.05 - df["wage_st"]) / df["wage_st"],
    )

    df["balanced_sample"] = (~df["fte1"].isna()) & (~df["delta_fte"].isna())

    # 保存主数据集
    MAIN_OUT = os.path.join(PROC_DIR, "cleaned_main.csv")
    df.to_csv(MAIN_OUT, index=False)
    print(f"[save] {MAIN_OUT}  ({len(df)} rows)")

    # 稳健性数据集
    df_robust = df.copy()
    df_robust.loc[df_robust["temp_close"] == 1, "fte2"] = 0.0
    df_robust["delta_fte"] = df_robust["fte2"] - df_robust["fte1"]
    df_robust["balanced_sample"] = (
        (~df_robust["fte1"].isna()) & (~df_robust["delta_fte"].isna())
    )
    ROBUST_OUT = os.path.join(PROC_DIR, "cleaned_robust.csv")
    df_robust.to_csv(ROBUST_OUT, index=False)
    print(f"[save] {ROBUST_OUT}  ({len(df_robust)} rows)")

    # 校验输出
    print("\n=== Store-closing counts (sanity check vs paper) ===")
    print("closed stores    :", int(df["closed"].sum()), "(paper: 6)")
    print("temp-close stores:", int(df["temp_close"].sum()), "(paper: 4)")

    summary = df.groupby("nj")[["fte1", "fte2", "wage_st", "wage_st2"]].mean()
    print("\n=== Cross-check vs paper Table 2 ===")
    reference = {
        "fte1":    {"NJ": 20.4,  "PA": 23.3},
        "fte2":    {"NJ": 21.0,  "PA": 21.2},
        "wage_st": {"NJ": 4.61,  "PA": 4.62},
        "wage_st2":{"NJ": 5.08,  "PA": 4.62},
    }
    print(f"{'var':<9}{'state':<4}{'paper':>9}{'mine':>10}{'diff':>9}")
    for var in ["fte1", "fte2", "wage_st", "wage_st2"]:
        for st in ["NJ", "PA"]:
            p = reference[var][st]
            m = summary.loc[1 if st=="NJ" else 0, var]
            print(f"{var:<9}{st:<4}{p:>9.2f}{m:>10.4f}{m - p:>+9.3f}")

    print("\n✅ Step 1 finished: data cleaning complete.")
    return df, df_robust

# ===================== 2. Table 2 =====================
def run_table2():
    print("\n" + "="*60)
    print("STEP 2 / 5 : Table 2 - Means of Key Variables")
    print("="*60)

    df = pd.read_csv(os.path.join(PROC_DIR, "cleaned_main.csv"))

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

    def pct(numer, denom):
        return np.where(denom>0, numer/denom*100, np.nan)

    df["pctft"] = pct(df["empft"], df["empft"] + df["emppt"])
    df["pctft2"] = pct(df["empft2"], df["empft2"] + df["emppt2"])

    def wage_flag(col, target_val, eps=0.01):
        s = pd.to_numeric(df[col], errors="coerce")
        cond = (s >= target_val - eps) & (s <= target_val + eps)
        return np.where(s.notna(), cond.astype(float), np.nan)

    df["wage_st_425"] = wage_flag("wage_st", 4.25)
    df["wage_st2_425"] = wage_flag("wage_st2", 4.25)
    df["wage_st2_505"] = wage_flag("wage_st2", 5.05)

    df["bk"] = (df["chain"] ==1).astype(float)
    df["kfc"] = (df["chain"] ==2).astype(float)
    df["rr"] = (df["chain"] ==3).astype(float)
    df["wendys"] = (df["chain"] ==4).astype(float)
    df["coowned"] = (df["co_owned"] ==1).astype(float)

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

    # 捕获输出保存文件
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
        str_t = "-" if (t_dash=="dash" or pd.isna(tval)) else f"{tval:.1f}"
        print(f"{label:<38}{str_nj:>16}{str_pa:>16}{str_t:>9}")
    print("-"*80)
    print("Notes: Standard errors are in parentheses. t^a is a two-sample unequal variance t-test for equality of means.")
    print("\nDone.")

    output_text = sys.stdout.getvalue()
    sys.stdout = old_stdout
    print(output_text)

    out_path = os.path.join(OUTPUT_DIR, "table2_result.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output_text)
    print(f"✅ Saved: {out_path}")

# ===================== 3. Table 3 & Table 4 =====================
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

    out_path = os.path.join(OUTPUT_DIR, "table3_table4_result.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output_text)
    print(f"✅ Saved: {out_path}")

# ===================== 4. Figure 1 =====================
def run_figure1():
    print("\n" + "="*60)
    print("STEP 4 / 5 : Figure 1 - Wage Distribution")
    print("="*60)

    df = pd.read_csv(os.path.join(PROC_DIR, "cleaned_main.csv"))

    edges = np.array([4.25, 4.35, 4.45, 4.55, 4.65, 4.75, 4.85, 4.95,
                      5.05, 5.15, 5.25, 5.35, 5.45, 5.55])
    centers = (edges[:-1] + edges[1:]) / 2.0

    nj = df["nj"] == 1.0
    pa = df["nj"] == 0.0

    def pct_by_state(wage_col):
        w = pd.to_numeric(df[wage_col], errors="coerce")
        wnj = w[nj].dropna()
        wpa = w[pa].dropna()
        nj_pct = np.histogram(wnj, bins=edges)[0] / float(wnj.shape[0]) * 100.0
        pa_pct = np.histogram(wpa, bins=edges)[0] / float(wpa.shape[0]) * 100.0
        return nj_pct, pa_pct

    nj1, pa1 = pct_by_state("wage_st")
    nj2, pa2 = pct_by_state("wage_st2")

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 10
    bar_w = 0.038
    offset = 0.020
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 8.5), sharex=True)

    for ax, njp, pap, title, ymax, yticks in [
        (axes[0], nj1, pa1, "February 1992", 35, [0, 10, 20, 30]),
        (axes[1], nj2, pa2, "November 1992", 90, [0, 20, 40, 60, 80]),
    ]:
        ax.bar(centers - offset, njp, width=bar_w,
               color="black", edgecolor="black", linewidth=0.5)
        ax.bar(centers + offset, pap, width=bar_w,
               color="white", edgecolor="black", linewidth=0.5, hatch="///")
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_ylim(0, ymax)
        ax.set_yticks(yticks)
        ax.set_xlim(4.20, 5.60)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    axes[1].set_xticks(edges)
    axes[1].set_xticklabels([f"{e:.2f}" for e in edges], rotation=0, fontsize=7)
    plt.setp(axes[0].get_xticklabels(), visible=False)
    plt.setp(axes[0].get_xticklines(), visible=False)

    fig.text(0.02, 0.52, "Percent of Stores", rotation=90,
             va="center", ha="center", fontsize=11)

    handles = [
        Patch(facecolor="black", edgecolor="black", label="New Jersey"),
        Patch(facecolor="white", edgecolor="black", hatch="///",
              label="Pennsylvania"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.065))
    fig.text(0.5, 0.012, "FIGURE 1. DISTRIBUTION OF STARTING WAGE RATES",
             ha="center", va="bottom", fontsize=11)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.14,
                        hspace=0.30)

    out_path = os.path.join(OUTPUT_DIR, "figure1.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {out_path}")

# ===================== 5. 异质性 DID =====================
def run_het_did():
    print("\n" + "="*60)
    print("STEP 5 / 5 : Heterogeneous DID by Wage Subgroup")
    print("="*60)

    main_df = pd.read_csv(os.path.join(PROC_DIR, "cleaned_main.csv"))
    robust_df = pd.read_csv(os.path.join(PROC_DIR, "cleaned_robust.csv"))

    GROUPS = {
        "L": ("Low: wage_st = 4.25",        lambda w: np.isclose(w, 4.25)),
        "M": ("Mid: 4.26 <= wage_st <= 4.99", lambda w: (w >= 4.26) & (w <= 4.99)),
        "H": ("High: wage_st >= 5.00",       lambda w: (w >= 5.00)),
    }

    def subgroup_mask(df):
        w = pd.to_numeric(df["wage_st"], errors="coerce")
        return {k: fn(w) for k, (_label, fn) in GROUPS.items()}

    def did_first_diff(df, mask):
        sub = df[mask & df["fte1"].notna() & df["fte2"].notna()].copy()
        n = len(sub)
        if n < 4:
            return (np.nan,) * 6 + (0, 0)
        dy = (sub["fte2"] - sub["fte1"]).astype(float)
        nj = sub["nj"].astype(float)
        X = sm.add_constant(nj)
        fit = sm.OLS(dy, X).fit()
        coef = float(fit.params["nj"])
        se = float(fit.bse["nj"])
        pval = float(fit.pvalues["nj"])
        ci_low, ci_high = fit.conf_int().loc["nj"].tolist()
        n_nj = int((nj == 1).sum())
        n_pa = int((nj == 0).sum())
        return coef, se, pval, n, ci_low, ci_high, n_nj, n_pa

    def run_sample(df, name):
        masks = subgroup_mask(df)
        rows = []
        estimates = {}
        for key, (label, _fn) in GROUPS.items():
            coef, se, pval, n, ci_low, ci_high, n_nj, n_pa = \
                did_first_diff(df, masks[key])
            rows.append({
                "Sample": name,
                "Subgroup definition": label,
                "N": n,
                "N_NJ": n_nj,
                "N_PA": n_pa,
                "DID coefficient": coef,
                "SE": se,
                "p-value": pval,
            })
            estimates[key] = {
                "coef": coef, "se": se, "pval": pval, "n": n,
                "ci_low": ci_low, "ci_high": ci_high,
            }
        return rows, estimates

    rows_main, est_main = run_sample(main_df, "Baseline")
    rows_robust, est_robust = run_sample(robust_df, "Robustness")

    md_lines = []
    md_lines.append("| Sample | Subgroup definition | N (NJ / PA) | DID coef | SE | p-value |")
    md_lines.append("|---|---|---|---|---|---|")
    def fmt_row(r):
        coef = "—" if np.isnan(r["DID coefficient"]) else f"{r['DID coefficient']:.3f}"
        se = "—" if np.isnan(r["SE"]) else f"{r['SE']:.3f}"
        pv = "—" if np.isnan(r["p-value"]) else f"{r['p-value']:.4f}"
        return (f"| {r['Sample']} | {r['Subgroup definition']} | "
                f"{r['N']} ({r['N_NJ']} / {r['N_PA']}) | {coef} | {se} | {pv} |")
    for r in rows_main + rows_robust:
        md_lines.append(fmt_row(r))
    table_text = "\n".join(md_lines)

    print("\n--- Markdown summary table ---\n")
    print(table_text)

    md_path = os.path.join(OUTPUT_DIR, "het_did_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Heterogeneous DID by pre-policy starting-wage subgroup\n\n")
        f.write(table_text + "\n")
    print(f"\n✅ Saved table: {md_path}")

    # 绘图
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 11
    keys = ["L", "M", "H"]
    coefs = np.array([est_main[k]["coef"] for k in keys])
    cis_low = np.array([est_main[k]["ci_low"] for k in keys])
    cis_high = np.array([est_main[k]["ci_high"] for k in keys])
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.errorbar(x, coefs,
                yerr=[coefs - cis_low, cis_high - coefs],
                fmt="o", color="black", ecolor="black", capsize=6,
                markersize=7, linewidth=1.2)
    for xi, c, lo, hi in zip(x, coefs, cis_low, cis_high):
        ax.text(xi, hi + 0.12, f"{c:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(["Low\n(wage_st = 4.25)",
                        "Mid\n(4.26–4.99)",
                        "High\n(wage_st ≥ 5.00)"], fontsize=10)
    ax.set_xlim(-0.5, 2.5)
    ax.set_xlabel("Pre-policy starting-wage subgroup (New Jersey)")
    ax.set_ylabel("DID coefficient (change in FTE)")
    ax.set_title("Heterogeneous treatment effects by starting wage (baseline sample)")
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, "het_did_plot.png")
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"✅ Saved plot: {plot_path}")

    # 结果解读
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    cL, cM, cH = (est_main["L"]["coef"], est_main["M"]["coef"], est_main["H"]["coef"])
    if cL <= cM <= cH:
        print("  -> Consistent with theory: DID is most negative for lowest-wage group.")
    else:
        print("  -> NOT consistent with competitive model prediction.")
        print("     Mirrors Card & Krueger's central finding: employment did not fall")
        print("     in the most-affected stores.")
    print("\n✅ Step 5 finished: heterogeneous DID complete.")

# ===================== 主入口 =====================
def main():
    print("🚀 Card & Krueger (1994) Full Replication Pipeline")
    print(f"📂 Working directory: {os.getcwd()}")

    run_clean()
    run_table2()
    run_table3_table4()
    run_figure1()
    run_het_did()

    print("\n" + "="*60)
    print("🎉 ALL 5 STEPS FINISHED SUCCESSFULLY!")
    print("="*60)
    print("📂 Cleaned datasets: ./data/processed/")
    print("📂 Tables & figures:  ./output/")

if __name__ == "__main__":
    main()