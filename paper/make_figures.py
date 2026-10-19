"""Figures for the latent-coherence write-up. Numbers are copied from
`lceval.py analyze --run runs/main --planned-models 7` (analysis_7models)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "pdf.fonttype": 42,
})

# model: (mean residual, lo, hi), (persistent, lo, hi), noise, reasoning tokens
D = {
    "GPT-6.1 Sol":      ((0.083, 0.060, 0.109), (0.083, 0.018, 0.128), 0.163, 400.2),
    "Claude Opus 5.5":  ((0.115, 0.073, 0.169), (0.000, 0.000, 0.138), 0.342, 499.0),
    "DeepSeek V4-Pro":  ((0.148, 0.095, 0.210), (0.267, 0.103, 0.398), 0.302, 16872.2),
    "Grok 4.7":         ((0.206, 0.147, 0.271), (0.206, 0.000, 0.316), 0.506, 7236.0),
    "Gemini 3.8 Flash": ((0.255, 0.185, 0.337), (0.278, 0.104, 0.397), 0.502, 4408.7),
    "GLM 5.3":          ((0.260, 0.202, 0.328), (0.108, 0.000, 0.249), 0.526, 6732.9),
    "Claude Haiku 5.5": ((0.371, 0.272, 0.486), (0.345, 0.150, 0.482), 0.641, 1116.7),
}
models = list(D)[::-1]          # best at top
ys = range(len(models))


def dotci(ax, idx, title, xlab, color):
    for y, m in zip(ys, models):
        est, lo, hi = D[m][idx]
        sig = lo > 0
        ax.plot([lo, hi], [y, y], color=color, lw=2, solid_capstyle="round", zorder=2)
        ax.plot([est], [y], marker="o", ms=5.5, mfc=color if sig or idx == 0 else "white",
                mec=color, mew=1.5, zorder=3)
    ax.set_yticks(list(ys))
    ax.set_ylim(-0.6, len(models) - 0.4)
    ax.set_xlim(-0.012, 0.52)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(xlab)
    ax.set_title(title, loc="left", fontsize=9, color=INK)


fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True, gridspec_kw={"wspace": 0.08})
dotci(a, 0, "(a) Mean residual (90 latent items, one wording)", "mean absolute residual (log-odds)", BLUE)
dotci(b, 1, "(b) Persistent residual (repeated items)", "persistent residual (log-odds)", BLUE)
a.set_yticklabels(models)
fig.savefig("figures/residuals.pdf", bbox_inches="tight")

# reasoning length vs residual
fig, ax = plt.subplots(figsize=(3.3, 2.4))
offsets = {"GPT-6.1 Sol": (6, -9), "Claude Opus 5.5": (6, 2), "Claude Haiku 5.5": (6, -3),
           "Gemini 3.8 Flash": (-7, 0), "GLM 5.3": (6, 6), "Grok 4.7": (6, -7),
           "DeepSeek V4-Pro": (0, -30)}
for m, ((est, lo, hi), _, _, tok) in D.items():
    ax.plot([tok, tok], [lo, hi], color=BLUE, lw=1.5, alpha=0.45, solid_capstyle="round")
    ax.plot([tok], [est], "o", ms=5.5, color=BLUE, mec="white", mew=1)
    dx, dy = offsets[m]
    ax.annotate(m.replace("Claude ", ""), (tok, est), xytext=(dx, dy), textcoords="offset points",
                ha="left" if dx > 0 else ("right" if dx < 0 else "center"), fontsize=7, color=INK2)
ax.set_xscale("log")
ax.set_xlim(250, 40000)
ax.set_ylim(0, 0.52)
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
ax.set_xlabel("mean reasoning tokens per answer (log scale)")
ax.set_ylabel("mean absolute residual (log-odds)")
ax.grid(color=GRID, lw=0.6)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.savefig("figures/reasoning.pdf", bbox_inches="tight")
print("ok")
