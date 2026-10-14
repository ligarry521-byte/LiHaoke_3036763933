# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os

# =====路径配置（适配你的项目）=====
PROC = os.path.join("data", "processed")
OUTPUT = os.path.join("output")
os.makedirs(OUTPUT, exist_ok=True)

df = pd.read_csv(os.path.join(PROC, "cleaned_main.csv"))

edges = np.array([4.25, 4.35, 4.45, 4.55, 4.65, 4.75, 4.85, 4.95,
                  5.05, 5.15, 5.25, 5.35, 5.45, 5.55])
centers = (edges[:-1] + edges[1:]) / 2.0

# 修正：数据集变量为nj，不是state
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

# 输出图片到output文件夹
out_path = os.path.join(OUTPUT, "figure1.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print("saved figure 1 to:", out_path)
