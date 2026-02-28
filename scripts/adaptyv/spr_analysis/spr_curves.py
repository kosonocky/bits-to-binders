import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(DATA_DIR, "raw_data")
FIT_DIR = os.path.join(DATA_DIR, "fit_data")
OUT_DIR = DATA_DIR

CONCENTRATIONS = [10.0, 31.6, 100.0, 316.2, 1000.0]
REPLICATES = [1, 2, 3]

# Muted plasma-like palette: navy → blue-violet → dusty mauve → warm tan → amber
PALETTE = ["#0F708E", "#2A8CC7", "#78ADE8", "#BEC7F5", "#FFDCEC"]
colors = {c: PALETTE[i] for i, c in enumerate(CONCENTRATIONS)}

T_MAX = 600.0  # trim after this

for rep in REPLICATES:
    fig, ax = plt.subplots(figsize=(5, 5))

    for conc in CONCENTRATIONS:
        label = f"{conc}"
        color = colors[conc]

        # Raw data (trimmed to T_MAX)
        raw_file = os.path.join(RAW_DIR, f"2383_{rep}_{conc}.csv")
        raw = pd.read_csv(raw_file)
        mask_raw = raw["t"] <= T_MAX
        ax.plot(raw["t"][mask_raw], raw["y"][mask_raw],
                color=color, alpha=1.0, linewidth=2.0, label=label, zorder=1)

        # Fitted curve — detect vertical jump and split
        fit_file = os.path.join(FIT_DIR, f"2383_{rep}_{conc}.csv")
        fit = pd.read_csv(fit_file)
        mask_fit = fit["t"] <= T_MAX
        t = fit["t"][mask_fit].values
        y = fit["y"][mask_fit].values

        dy = np.abs(np.diff(y))
        mask_jump = t[:-1] > 200
        jump_idx = np.where(mask_jump)[0][np.argmax(dy[mask_jump])]

        ax.plot(t[:jump_idx+1], y[:jump_idx+1], color="black", linestyle="--", linewidth=1.2, zorder=2)
        ax.plot(t[jump_idx+1:], y[jump_idx+1:], color="black", linestyle="--", linewidth=1.2, zorder=2)

    ax.axvline(x=300, color="gray", linestyle="-", linewidth=1.2)

    ax.set_xlim(0, T_MAX)
    # ax.set_ylim(0, 30)
    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("Response units", fontsize=14)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(title="[Binder] (nM)", frameon=False, fontsize=14, title_fontsize=14,
              bbox_to_anchor=(0.5, -0.12), loc="upper center", ncol=5,
              handlelength=1.0, handletextpad=0.4, columnspacing=0.8)
    ax.text(0.65, 0.5, r"$\it{K}$$_D$ = 643 nM", transform=ax.transAxes,
            fontsize=14, verticalalignment="top")

    # hide top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"spr_curves_rep{rep}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved {out_path}")
