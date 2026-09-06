"""Redraw method_d_vs_method_c.png from saved per-flow data (no pipeline /
LLM re-run). Reads method_d_vs_method_c.csv.

JUROR's per-flow rating is an integer (0/1/2 here) while Method C's value is a
continuous mean, so a y=x scatter collapses every flow onto three horizontal
lines. We instead group flows by their integer JUROR rating and show the
distribution of Method C means within each level (points sized by transport
mass; diamonds = group mean +/- 1 SD), which makes the monotonic agreement
legible without overplotting.
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(HERE / "method_d_vs_method_c.csv")))
j = np.array([int(r["method_d_rating"]) for r in rows])       # JUROR integer rating
c = np.array([float(r["method_c_mean"]) for r in rows])       # Method C mean within flow
mass = np.array([float(r["mass"]) for r in rows])
sp = spearmanr(j, c).statistic
pe = pearsonr(j, c).statistic

rng = np.random.default_rng(42)
levels = sorted(set(j))
colors = plt.cm.viridis(np.linspace(0.15, 0.8, len(levels)))

fig, ax = plt.subplots(figsize=(7, 6))
means = []
for lvl, col in zip(levels, colors):
    m = j == lvl
    xs = lvl + rng.uniform(-0.16, 0.16, size=m.sum())
    ax.scatter(xs, c[m], s=40 + mass[m] * 1400, color=col, alpha=0.7,
               edgecolor="white", linewidth=0.6, zorder=3)
    mu, sd = c[m].mean(), c[m].std()
    means.append(mu)
    ax.errorbar(lvl, mu, yerr=sd, fmt="D", color="black", ms=9, capsize=5,
                elinewidth=1.4, zorder=4)
    ax.annotate(f"n={m.sum()}", (lvl, ax.get_ylim()[1]), xytext=(lvl, 2.32),
                ha="center", fontsize=9, color="#444")

# monotonic trend connecting the group means
ax.plot(levels, means, color="#888", ls="--", lw=1.3, zorder=2)

ax.set_xticks(levels)
ax.set_xlim(-0.5, max(levels) + 0.5)
ax.set_ylim(-0.3, 2.5)
ax.set_xlabel("JUROR flow-level rating (single LLM call, 0--4)")
ax.set_ylabel("Method C mean pair rating within flow")
ax.set_title(f"JUROR flow rating vs Method C mean\n"
             f"Spearman $\\rho$ = {sp:.3f},  Pearson r = {pe:.3f}  (n={len(rows)} flows)")
ax.grid(axis="y", alpha=0.3)
# size legend (transport mass)
for mref, lab in [(0.01, "mass 0.01"), (0.06, "0.06"), (0.12, "0.12")]:
    ax.scatter([], [], s=40 + mref * 1400, color="#999", alpha=0.7,
               edgecolor="white", label=lab)
ax.legend(title="point size $\\propto$ flow mass", loc="lower right",
          fontsize=8, title_fontsize=8, frameon=True)
fig.tight_layout()
fig.savefig(HERE / "method_d_vs_method_c.png", dpi=150)
plt.close(fig)
print(f"Redrew method_d_vs_method_c.png  (grouped; Spearman={sp:.3f}, "
      f"group means={[round(m,2) for m in means]})")
