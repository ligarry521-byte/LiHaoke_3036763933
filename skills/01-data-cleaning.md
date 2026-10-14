# Skill 1 — Data Cleaning & Preprocessing

## What this skill is

Turning a raw, messy survey dataset into a clean, analysis-ready dataset: reading and
standardising columns, coercing types, handling missing values, constructing derived
variables, and validating the result against known external facts.

## How it was applied in this project

Implemented in [`code/others/clean_data.py`](../code/others/clean_data.py) (Step 1 of
the pipeline). The raw file `data/raw/public.csv` (410 restaurants, 46 columns,
uppercase names) is transformed into two analysis-ready datasets in
`data/processed/`:

1. **Column standardisation** — all column names are stripped and lowercased.
2. **Type coercion** — numeric columns (`empft`, `emppt`, `nmgrs`, `wage_st`, and
   their wave-2 counterparts) and categorical codes (`state`, `status2`, `chain`,
   `co_owned`) are converted with `pd.to_numeric`, treating non-numeric values as
   missing.
3. **Indicator construction** — `nj` (1 = New Jersey, 0 = Pennsylvania), `closed`
   (status 3), and `temp_close` (statuses 2/4/5) are built from raw codes.
4. **Derived variables** —
   - full-time-equivalent employment `fte = empft + nmgrs + 0.5 * emppt`;
   - `fte2` with business closures coded as `0` and temporary closures coded as
     missing;
   - change in employment `delta_fte = fte2 - fte1`;
   - the wage-gap variable `gap = (5.05 - wage_st) / wage_st` for New Jersey stores
     above the old minimum, `0` otherwise.
5. **Two sample definitions** —
   - `cleaned_main.csv` (baseline): temporarily-closed stores are missing in wave 2;
   - `cleaned_robust.csv` (robustness): temporarily-closed stores are coded `fte2 = 0`.
   - Both carry a `balanced_sample` flag for stores with non-missing `fte1` and
     `delta_fte`.
6. **Validation / sanity checks** — the script prints store-closing counts (6 closed,
   4 temporarily-closed, matching the paper), group means by state, and a line-by-line
   cross-check of group means against the paper's Table 2.

## Key techniques / tools

- `pandas` (`read_csv`, `to_numeric`, `groupby`, boolean masking, `np.select`).
- Missing-value conventions (`.isna()`, `.notna()`).
- Robustness-vs-baseline sample construction.
- Encoding handling (UTF-8 stdout for cross-platform runs).

## Evidence

- Input: `data/raw/public.csv` (410 rows × 46 columns).
- Outputs: `data/processed/cleaned_main.csv` and `data/processed/cleaned_robust.csv`.
- The reproduced Table 2 group means (e.g. NJ Wave-1 FTE = 20.44, PA = 23.33) match
  the paper exactly, confirming the cleaning is correct.
