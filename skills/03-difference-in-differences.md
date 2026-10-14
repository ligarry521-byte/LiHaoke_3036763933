# Skill 3 — Difference-in-Differences & Regression Analysis

## What this skill is

Estimating causal effects from a natural experiment using the difference-in-differences
(DiD) design and reduced-form regression, including standard errors for the
pre/post comparison and grouped (subgroup) treatment effects.

## How it was applied in this project

Implemented in [`code/others/table3_table4.py`](../code/others/table3_table4.py)
(Tables 3 & 4) and [`code/others/het_did.py`](../code/others/het_did.py) (the
extension).

### Table 3 — the DiD estimate

- Computes mean FTE employment per store **before** and **after** the New Jersey
  minimum-wage rise, for Pennsylvania, New Jersey, and New Jersey wage subgroups.
- Computes the **change** in mean FTE and the difference-in-differences
  `(NJ_after − NJ_before) − (PA_after − PA_before)`, with standard errors that account
  for the two-period structure.
- Also reports the change on the **balanced sample** and under the **temp-closed = 0**
  robustness convention.

### Table 4 — reduced-form regression

- Regresses the change in employment `delta_fte` on the treatment indicators using
  ordinary least squares (via `statsmodels`):
  - models (i)–(ii) use the New Jersey dummy `nj`;
  - models (iii)–(v) use the wage-gap variable `gap = (5.05 − wage_st)/wage_st`;
  - later specifications add chain, company-ownership, and region controls.
- Reproduces the paper's coefficients, standard errors, and the sample size (357).

### Extension — heterogeneous DiD

- Splits stores into three pre-policy starting-wage bins (low / mid / high) and
  estimates the DiD coefficient within each bin.
- Uses the two-period equivalence: the coefficient on `nj × post` in the levels
  regression equals the coefficient on `nj` in the first-difference regression
  `(fte2 − fte1)_i = a0 + a1·nj_i + u_i`. Both are computed, with a cluster-robust
  levels specification printed as a cross-check.

## Key techniques / tools

- Two-period difference-in-differences with the first-difference equivalence.
- `statsmodels` OLS (`sm.OLS`, `fit()`, coefficients, standard errors, p-values).
- Balanced-sample and robustness variants.
- Subgroup / heterogeneous treatment-effect estimation.
- Standard-error construction for pre/post differences.

## Evidence

- Outputs: `outputs/tables/table3_table4_result.md`, `outputs/tables/het_did_summary.md`.
- The DiD estimate `2.75 (1.31)` reproduces the paper's `2.76 (1.36)`; the Table 4
  coefficients match to two decimal places (e.g. model (i): 2.326 vs. 2.33), and the
  sample size matches exactly (357).
