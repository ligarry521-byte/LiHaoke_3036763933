# Skill 5 — Reproducible Research & Pipeline Automation

## What this skill is

Organising a multi-step analysis so that the entire result set can be regenerated from
raw inputs with a single, deterministic command — the core of reproducible research.

## How it was applied in this project

Implemented in [`code/run_all.py`](../code/run_all.py), which wraps the five analysis
steps into one pipeline:

1. Data cleaning → `data/processed/cleaned_main.csv`, `cleaned_robust.csv`;
2. Table 2 → `outputs/tables/table2_result.md`;
3. Tables 3 & 4 → `outputs/tables/table3_table4_result.md`;
4. Figure 1 → `outputs/figures/figure1.png`;
5. Heterogeneous DiD → `outputs/tables/het_did_summary.md`,
   `outputs/figures/het_did_plot.png`.

The pipeline is made reproducible through:

- **Dependency ordering** — each step runs only after the data it needs exists.
- **Centralised path configuration** — raw/processed/output directories are defined
  once and shared by all steps, resolved relative to the repository root.
- **Automatic directory creation** — output folders are created if missing.
- **Cross-platform robustness** — UTF-8 stdout is forced, so the scripts run on
  Windows/macOS/Linux without manual setup.
- **A documented, single entry point** — `python code/run_all.py` from the repository
  root regenerates everything; each step is also available as a standalone script in
  `code/others/`.
- **Built-in verification** — the scripts print cross-checks against the paper's
  published values (Table 2 means, Table 3 DiD, Table 4 coefficients) so a wrong
  result is immediately visible.

## Key techniques / tools

- Modular functions with a clear pipeline order (`run_clean`, `run_table2`, …).
- `os.makedirs(..., exist_ok=True)` and shared path constants.
- UTF-8 output handling (`sys.stdout.reconfigure`).
- Self-documenting output (console progress + Markdown/PNG files).

## Evidence

- Running `python code/run_all.py` regenerates all seven output files, and the
  console reports `ALL 5 STEPS FINISHED SUCCESSFULLY!` together with the paper
  cross-checks. See `README.md` for the full reproducibility instructions.
