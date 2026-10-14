# -*- coding: utf-8 -*-
"""
Reproduce the data‑cleaning step of Card & Krueger (1994 AER, "Minimum Wages and
Employment") using raw survey file public.csv.
Rules follow the original paper (Table 2 / Table 3 notes):
  * FTE (full‑time‑equivalent) employment = full‑time workers (incl. managers)
        + 0.5 * part‑time workers.
      fte1     = empft + nmgrs + 0.5 * emppt
      fte2_raw = empft2 + nmgrs2 + 0.5 * emppt2
  * Permanently closed stores (closed == 1)  -> fte2 = 0   (KEEP, do not drop)
  * Temporarily closed stores (temp_close==1) -> fte2 = NaN in MAIN dataset
        (missing: excluded from baseline stats & regression); a SEPARATE copy
        df_robust sets fte2 = 0 only for the Table 3 row‑5 robustness check.
  * delta_fte = fte2 - fte1
  * GAP (proportional wage increase needed to reach the new $5.05 minimum):
        gap = 0                                    if nj == 0 (Pennsylvania)
        gap = 0                                    if nj == 1 and wage_st >= 5.05
        gap = (5.05 - wage_st) / wage_st           otherwise
  * balanced_sample = (~fte1.isna()) & (~delta_fte.isna())

Note: Run script from repository root folder Li_Haoke_3036763933/
"""
import os
import numpy as np
import pandas as pd


def run_clean():
    # ---------- Repository relative paths ----------
    RAW_DIR = os.path.join("data", "raw")
    PROC_DIR = os.path.join("data", "processed")
    RAW_CSV = os.path.join(RAW_DIR, "public.csv")

    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROC_DIR, exist_ok=True)

    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(f"Raw data missing! Expected at {RAW_CSV}. Please upload public.csv to data/raw/")

    # 1. Read and normalise column names
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

    # 2. Build key indicators
    df["nj"] = df["state"].astype("Int64")
    df["closed"] = (df["status2"] == 3).astype(int)
    df["temp_close"] = (df["status2"].isin([2, 4, 5])).astype(int)

    # 3. Compute FTE employment
    df["fte1"] = df["empft"] + df["nmgrs"] + 0.5 * df["emppt"]
    df["fte2_raw"] = df["empft2"] + df["nmgrs2"] + 0.5 * df["emppt2"]

    df["fte2"] = df["fte2_raw"].copy()
    df.loc[df["closed"] == 1, "fte2"] = 0.0
    df.loc[df["temp_close"] == 1, "fte2"] = np.nan

    # 4. delta_fte
    df["delta_fte"] = df["fte2"] - df["fte1"]

    # 5. GAP variable
    cond_pa = df["nj"] == 0
    cond_high = (df["nj"] == 1) & (df["wage_st"] >= 5.05)
    df["gap"] = np.select(
        [cond_pa, cond_high],
        [0.0, 0.0],
        default=(5.05 - df["wage_st"]) / df["wage_st"],
    )

    # 6. balanced_sample flag
    df["balanced_sample"] = (~df["fte1"].isna()) & (~df["delta_fte"].isna())

    # 7. Save main dataset
    MAIN_OUT = os.path.join(PROC_DIR, "cleaned_main.csv")
    df.to_csv(MAIN_OUT, index=False)
    print(f"[save] {MAIN_OUT}  ({len(df)} rows)")

    # 8. Robustness dataset: temp‑closed stores set fte2=0
    df_robust = df.copy()
    df_robust.loc[df_robust["temp_close"] == 1, "fte2"] = 0.0
    df_robust["delta_fte"] = df_robust["fte2"] - df_robust["fte1"]
    df_robust["balanced_sample"] = (
        (~df_robust["fte1"].isna()) & (~df_robust["delta_fte"].isna())
    )
    ROBUST_OUT = os.path.join(PROC_DIR, "cleaned_robust.csv")
    df_robust.to_csv(ROBUST_OUT, index=False)
    print(f"[save] {ROBUST_OUT}  ({len(df_robust)} rows)")

    # 9. Sanity checks
    print("\n=== Store‑closing counts (sanity check vs paper) ===")
    print("closed stores    :", int(df["closed"].sum()), "(paper: 6)")
    print("  closed by nj   :", df.groupby("nj")["closed"].sum().to_dict())
    print("temp‑close stores:", int(df["temp_close"].sum()), "(paper: 4)")
    print("  temp‑close by nj:", df.groupby("nj")["temp_close"].sum().to_dict())
    print("status2 counts   :", df["status2"].value_counts().sort_index().to_dict())

    # 10. Group means
    print("\n=== Group means by nj (nonmissing obs per variable) ===")
    summary = df.groupby("nj")[["fte1", "fte2", "wage_st", "wage_st2"]].mean()
    with pd.option_context("display.float_format", lambda x: f"{x:,.4f}"):
        print(summary)
    print("\nObservation counts per variable per group:")
    print(df.groupby("nj")[["fte1", "fte2", "wage_st", "wage_st2"]].count())

    # 11. Cross‑check against paper Table2
    print("\n=== Cross‑check vs paper Table 2 ===")
    reference = {
        "fte1":    {"NJ": 20.4,  "PA": 23.3},
        "fte2":    {"NJ": 21.0,  "PA": 21.2},
        "wage_st": {"NJ": 4.61,  "PA": 4.62},
        "wage_st2":{"NJ": 5.08,  "PA": 4.62},
    }
    my = {
        "fte1":    {"NJ": summary.loc[1, "fte1"],     "PA": summary.loc[0, "fte1"]},
        "fte2":    {"NJ": summary.loc[1, "fte2"],     "PA": summary.loc[0, "fte2"]},
        "wage_st": {"NJ": summary.loc[1, "wage_st"],  "PA": summary.loc[0, "wage_st"]},
        "wage_st2":{"NJ": summary.loc[1, "wage_st2"], "PA": summary.loc[0, "wage_st2"]},
    }
    print(f"{'var':<9}{'state':<4}{'paper':>9}{'mine':>10}{'diff':>9}")
    for var in ["fte1", "fte2", "wage_st", "wage_st2"]:
        for st in ["NJ", "PA"]:
            p, m = reference[var][st], my[var][st]
            print(f"{var:<9}{st:<4}{p:>9.2f}{m:>10.4f}{m - p:>+9.3f}")
    print("\n[clean_data] Finished.")
    return df, df_robust


if __name__ == "__main__":
    # 直接运行本脚本就执行清洗
    run_clean()
