# -*- coding: utf-8 -*-
""" Heterogeneous DID extension for Card & Krueger (1994 AER). Estimates the treatment effect of the New Jersey minimum-wage rise on the change in full-time-equivalent employment (Delta FTE = fte2 - fte1), separately within three pre-policy starting-wage subgroups of New Jersey, exactly the three bins used in the original paper's Table 3:
    Group L (low)    : wage_st == 4.25
    Group M (middle) : 4.26 <= wage_st <= 4.99
    Group H (high)   : wage_st >= 5.00
DESIGN (confirmed with the user):
    For each wage bin we keep BOTH New Jersey (treatment) and Pennsylvania
    (control) stores whose wave-1 starting wage falls in that bin, and run the
    canonical two-period difference-in-differences specification
        FTE_it = b0 + b1*nj_i + b2*post_t + b3*(nj_i*post_t) + e_it
    where `nj` is the state dummy (1 = New Jersey, 0 = Pennsylvania) and
    `post` is the wave dummy (0 = wave 1, 1 = wave 2).  The interaction
    coefficient b3 is the DID estimator.
    NOTE on the two equivalent implementations:  the coefficient b3 on
    (nj*post) in the *levels* (long/panel) regression is numerically identical
    to the coefficient on `nj` in the *first-difference* regression
        (fte2 - fte1)_i = a0 + a1*nj_i + u_i,
    which is exactly Table 4 model (i).  We therefore report b3 but estimate
    it with the first-difference regression, whose ordinary (non-robust) OLS
    standard error matches the paper's Table 4 and is the correct SE for a
    two-period DID (each store contributes a single first-difference
    observation).  A long-format cross-check is also printed to demonstrate
    b3 (levels, cluster-robust by store) == a1 (first-difference OLS).
SAMPLES:
    * baseline    -> ./data/processed/cleaned_main.csv
                     (temporarily-closed stores are missing in wave 2, so they
                      are dropped from the balanced panel)
    * robustness  -> ./data/processed/cleaned_robust.csv
                     (temporarily-closed stores are coded fte2 = 0)
This script does NOT re-run the original Table 3 / Table 4 code (reproduce_tables.py); it is an independent heterogeneous extension.
"""
import os
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =====路径配置（仅修改输出文件夹，业务逻辑完全不变）=====
PROC = os.path.join("data", "processed")
TABLE_DIR = os.path.join("outputs", "tables")
FIG_DIR = os.path.join("outputs", "figures")
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

MAIN = os.path.join(PROC, "cleaned_main.csv")     # baseline sample
ROBUST = os.path.join(PROC, "cleaned_robust.csv")  # robustness sample
# ---------------------------------------------------------------------------
# subgroup definitions (pre-policy starting wage, wave 1)
# ---------------------------------------------------------------------------
GROUPS = {
    "L": ("Low: wage_st = 4.25",        lambda w: np.isclose(w, 4.25)),
    "M": ("Mid: 4.26 <= wage_st <= 4.99", lambda w: (w >= 4.26) & (w <= 4.99)),
    "H": ("High: wage_st >= 5.00",       lambda w: (w >= 5.00)),
}

def subgroup_mask(df):
    """Return a dict {key: boolean mask} for the three wage bins."""
    w = pd.to_numeric(df["wage_st"], errors="coerce")
    return {k: fn(w) for k, (_label, fn) in GROUPS.items()}

def did_first_diff(df, mask):
    """First-difference DID within a subgroup (identical to Table 4 model i).
    Returns (coef, se, pval, n, ci_low, ci_high, n_nj, n_pa).
      coef = a1 = DID == b3 (the nj*post interaction in levels).
    """
    sub = df[mask & df["fte1"].notna() & df["fte2"].notna()].copy()
    n = len(sub)
    if n < 4:
        return (np.nan,) * 6 + (0, 0)
    dy = (sub["fte2"] - sub["fte1"]).astype(float)      # Delta FTE
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

def did_levels_crosscheck(df, mask):
    """Long-format (levels) DID: FTE on nj, post, nj*post, clustered by store.
    Confirms b3 == a1 (the first-difference estimate).  Cluster-robust SEs by
    store in a two-period panel equal the first-difference OLS SE.
    """
    sub = df[mask & df["fte1"].notna() & df["fte2"].notna()].reset_index(drop=True)
    n = len(sub)
    if n < 4:
        return (np.nan,) * 4
    nj = sub["nj"].astype(float).to_numpy()
    ids = np.repeat(np.arange(n), 2)
    post = np.tile([0.0, 1.0], n)
    fte = np.empty(2 * n)
    fte[0::2] = sub["fte1"].astype(float).to_numpy()
    fte[1::2] = sub["fte2"].astype(float).to_numpy()
    X = np.column_stack([np.ones(2 * n), np.repeat(nj, 2), post,
                         np.repeat(nj, 2) * post])
    fit = sm.OLS(fte, X).fit(cov_type="cluster", cov_kwds={"groups": ids})
    coef = float(fit.params[3])   # b3 = nj*post
    se = float(fit.bse[3])
    pval = float(fit.pvalues[3])
    return coef, se, pval, n

def run_sample(df, name):
    """Run the three subgroup DIDs on one sample and return rows + estimates."""
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
        # cross-check: levels (nj*post) == first-difference (nj)
        lc_coef, lc_se, lc_pval, lc_n = did_levels_crosscheck(df, masks[key])
        if not np.isnan(lc_coef):
            print(f"[check {name} {key}] first-diff b={coef:.4f} (se {se:.4f})  "
                  f"| levels nj*post b={lc_coef:.4f} (clustered se {lc_se:.4f})")
    return rows, estimates

def main():
    main_df = pd.read_csv(MAIN)
    robust_df = pd.read_csv(ROBUST)
    print("=" * 78)
    print("HETEROGENEOUS DID BY PRE-POLICY WAGE SUBGROUP (Card & Krueger 1994)")
    print("Dependent variable: Delta FTE = fte2 - fte1")
    print("Estimator: b3 = nj*post (== nj coefficient in the first-difference")
    print("           regression, i.e. Table 4 model (i) run within each bin)")
    print("=" * 78)
    rows_main, est_main = run_sample(main_df, "Baseline")
    rows_robust, est_robust = run_sample(robust_df, "Robustness")
    # ------------------------------------------------------------------
    # consolidated markdown table
    # ------------------------------------------------------------------
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
    # also persist the table for the report appendix
    md_path = os.path.join(TABLE_DIR, "het_did_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Heterogeneous DID by pre-policy starting-wage subgroup\n\n")
        f.write(table_text + "\n")
    print(f"\n[saved] {md_path}")
    # ------------------------------------------------------------------
    # plot: point estimates + 95% CI (baseline sample)
    # ------------------------------------------------------------------
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 11
    keys = ["L", "M", "H"]
    labels = [GROUPS[k][0] for k in keys]
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
        ax.text(xi, hi + 0.12, f"{c:.2f}", ha="center", va="bottom",
                fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(["Low\n(wage_st = 4.25)",
                        "Mid\n(4.26–4.99)",
                        "High\n(wage_st ≥ 5.00)"], fontsize=10)
    ax.set_xlim(-0.5, 2.5)
    ax.set_xlabel("Pre-policy starting-wage subgroup (New Jersey)")
    ax.set_ylabel("DID coefficient (change in FTE)")
    ax.set_title("Heterogeneous treatment effects by starting wage "
                 "(baseline sample)")
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    plot_path = os.path.join(FIG_DIR, "het_did_plot.png")
    fig.savefig(plot_path, dpi=300)
    print(f"[saved] {plot_path}")
    # ------------------------------------------------------------------
    # plain-text interpretation for the report
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("INTERPRETATION")
    print("=" * 78)
    print(
        "Theory: the minimum-wage rise hits the lowest-wage stores hardest "
        "because the\nmandated increase is largest there.  Group L (wage_st = "
        "$4.25) faces the full\n$0.80 rise to the new $5.05 minimum, Group M "
        "a partial rise, and Group H\n(wage_st >= $5.00) essentially no "
        "binding increase.  The standard competitive-\nlabour-demand prediction "
        "is therefore a DID coefficient that is most negative for\nGroup L and "
        "closest to zero for Group H.\n"
    )
    for k in keys:
        c = est_main[k]
        print(f"  {GROUPS[k][0]:<28} DID = {c['coef']:+.3f}  "
              f"(SE {c['se']:.3f}, p = {c['pval']:.4f}, N = {c['n']})")
    # automated assessment of the theory prediction (baseline sample)
    cL, cM, cH = (est_main["L"]["coef"], est_main["M"]["coef"],
                  est_main["H"]["coef"])
    print("\nAssessment (baseline):")
    if cL <= cM <= cH:
        print("  -> Consistent with the theory prediction: the DID is most "
              "negative for the\n     lowest-wage group and increases "
              "(becomes less negative / more positive)\n     toward the "
              "high-wage group.")
    else:
        print("  -> NOT monotonically consistent with the competitive-model "
              "prediction:\n     the lowest-wage group does not show the most "
              "negative DID.  This mirrors\n     Card & Krueger's central "
              "finding that employment did not fall in the\n     most-affected "
              "stores, and is consistent with their monopsony / demand-shock\n"
              "     interpretation.")
    print(
        "\nRobustness (temp-closed stores coded fte2 = 0):\n"
        "  " + "  |  ".join(
            f"{k}: {est_robust[k]['coef']:+.3f} (p={est_robust[k]['pval']:.3f})"
            for k in keys)
    )
    print("\nDone.")

if __name__ == "__main__":
    main()


