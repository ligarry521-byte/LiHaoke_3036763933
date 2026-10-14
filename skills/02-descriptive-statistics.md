# Skill 2 — Descriptive Statistics & Hypothesis Testing

## What this skill is

Summarising a dataset with group means and standard errors, and formally comparing
group means across two independent samples using two-sample hypothesis tests.

## How it was applied in this project

Implemented in [`code/others/table2.py`](../code/others/table2.py), reproducing
**Table 2 — Means of Key Variables** of the paper. For each variable, the script
computes the mean and standard error separately for New Jersey and Pennsylvania, and
tests the equality of means across the two states:

- **Store-type distribution** (Burger King, KFC, Roy Rogers, Wendy's, company-owned),
  reported as percentages.
- **Wave-1 and Wave-2 means** of FTE employment, percentage full-time employees,
  starting wage, share of stores at $4.25 (and $5.05 in wave 2), full-meal price,
  weekday opening hours, and recruiting bonus.
- **Two-sample tests** — Welch's (unequal-variance) two-sample *t*-test
  `t = (mean_NJ − mean_PA) / sqrt(se_NJ² + se_PA²)` for every row, reported in the
  `t^a` column.

A notable detail handled here is the **scaling of percentage variables**: proportion
variables stored in `[0, 1]` (store-type shares, full-time share, wage = $4.25 share,
recruiting-bonus share) are multiplied by 100 to be reported as percentage points,
while dollar-valued variables (the meal price `pentree`) are left unscaled.

## Key techniques / tools

- `pandas` groupwise summaries (`groupby("nj")`, `mean`, `std(ddof=1)`).
- Standard error of the mean `SE = s / sqrt(n)`.
- Welch's two-sample *t*-test.
- Percentage vs. dollar-unit scaling of variables.

## Evidence

- Output: `outputs/tables/table2_result.md`.
- Reproduced values match the paper (e.g. Wave-1 FTE: NJ 20.4 (0.51), PA 23.3 (1.35);
  Wave-2 starting wage: NJ 5.08, PA 4.62), confirming the summarisation and the
  tests are correct.
