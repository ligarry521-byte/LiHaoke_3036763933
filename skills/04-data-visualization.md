# Skill 4 — Data Visualization

## What this skill is

Communicating empirical results graphically: reproducing a published figure, and
building a clear coefficient plot for an original analysis.

## How it was applied in this project

### Figure 1 — wage distribution

Implemented in [`code/others/figure1.py`](../code/others/figure1.py). The figure is a
two-panel histogram of the distribution of starting wage rates for New Jersey (black)
and Pennsylvania (hatched white) stores, in February 1992 and November 1992:

- wage values are binned into the paper's intervals (4.25 … 5.55) and converted to
  **percent of stores**;
- bars are side-by-side, with New Jersey in solid black and Pennsylvania in hatched
  white, matching the paper's black-and-white style;
- the figure uses a serif font, hidden top/right spines, a shared x-axis, a rotated
  y-axis label, and a two-entry legend, then is exported at 300 dpi.

### Heterogeneous DiD plot

In [`code/others/het_did.py`](../code/others/het_did.py), a coefficient plot shows the
estimated DiD effect for the three pre-policy wage subgroups, with 95% confidence
interval error bars, a reference line at zero, annotated point estimates, and a
zero-hidden grid.

## Key techniques / tools

- `matplotlib` (`subplots`, `bar`, `errorbar`, `axhline`, `legend`, `savefig`).
- Histogram binning with `numpy.histogram` and percentage scaling.
- Publication-style styling (serif fonts, spine removal, hatched fills, 300-dpi export).

## Evidence

- Outputs: `outputs/figures/figure1.png` and `outputs/figures/het_did_plot.png`.
- `figure1.png` reproduces the paper's Figure 1 layout (two waves, NJ vs. PA, same
  binning and axis limits).
