

import json, sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
except ImportError:
    print("Run:  pip install matplotlib numpy")
    sys.exit(1)

RESULTS_DIR = Path("results")
PLOTS_DIR   = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Colour palette
C_RED    = "#E53935"
C_GREEN  = "#43A047"
C_BLUE   = "#1565C0"
C_PURPLE = "#6A1B9A"
C_BG     = "#FAFAFA"
C_GRID   = "#E0E0E0"


def load():
    p1 = RESULTS_DIR / "phase1_baseline_results.json"
    p2 = RESULTS_DIR / "phase2_private_results.json"
    for p in [p1, p2]:
        if not p.exists():
            print(f"Missing {p}. Run phase1 and phase2 first.")
            sys.exit(1)
    return json.loads(p1.read_text()), json.loads(p2.read_text())


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(C_BG)
    ax.grid(True, color=C_GRID, linestyle="--", linewidth=0.7, zorder=0)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _ensure_entities(r):
    if "avg_entities_per_response" not in r:
        qs = r.get("queries", [])
        r["avg_entities_per_response"] = round(
            sum(q.get("pii_leaked", 0) for q in qs) / max(len(qs), 1), 2
        )
    return r




def plot_tradeoff(p1, p2):
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(C_BG)

    combos = [
        ("hotpotqa",    "HotpotQA",         "o", C_BLUE,   C_PURPLE),
        ("pii_masking", "PII-masking-200k",  "s", C_BLUE,   C_PURPLE),
    ]

    for ds_key, ds_label, marker, hc, rc in combos:
        for fw_key, fw_label, color in [("hipporag","HippoRAG-2",hc),
                                         ("raganything","RAG-Anything",rc)]:
            b = _ensure_entities(p1[ds_key][fw_key])
            p = _ensure_entities(p2[ds_key][fw_key])

        
            ax.scatter(b["avg_f1"], b["pii_leakage_rate"],
                       marker=marker, s=150, color=color, alpha=0.35,
                       edgecolors=color, linewidths=1.0, zorder=3)
         
            ax.scatter(p["avg_f1"], p["pii_leakage_rate"],
                       marker=marker, s=210, color=color, alpha=1.0,
                       edgecolors="black", linewidths=1.2, zorder=4)
          
            dx = p["avg_f1"]         - b["avg_f1"]
            dy = p["pii_leakage_rate"]- b["pii_leakage_rate"]
            if abs(dx) > 0.001 or abs(dy) > 0.001:
                ax.annotate(
                    "", xy=(p["avg_f1"], p["pii_leakage_rate"]),
                    xytext=(b["avg_f1"], b["pii_leakage_rate"]),
                    arrowprops=dict(arrowstyle="->", color=color,
                                    lw=1.8, connectionstyle="arc3,rad=0.2"),
                    zorder=5,
                )
            # Label
            ax.annotate(f"{fw_label}\n({ds_label})",
                        xy=(p["avg_f1"], p["pii_leakage_rate"]),
                        xytext=(7, 4), textcoords="offset points",
                        fontsize=7.5, color=color)

    # Ideal zone
    ax.axvspan(0.25, 1.0, alpha=0.03, color=C_GREEN, zorder=1)
    ax.axhspan(0.0,  0.3, alpha=0.03, color=C_GREEN, zorder=1)
    ax.text(0.97, 0.05, "✅ Ideal Zone",
            transform=ax.transAxes, fontsize=8, color=C_GREEN,
            ha="right", style="italic")

    legend_els = [
        mpatches.Patch(color=C_BLUE,   label="HippoRAG-2"),
        mpatches.Patch(color=C_PURPLE, label="RAG-Anything"),
        plt.scatter([], [], marker="o", color="grey", s=60, label="HotpotQA"),
        plt.scatter([], [], marker="s", color="grey", s=60, label="PII-masking-200k"),
        plt.scatter([], [], s=70,  alpha=0.35, color="grey", label="Baseline"),
        plt.scatter([], [], s=120, alpha=1.0,  color="grey", label="Private (after masking)"),
    ]
    ax.legend(handles=legend_els, fontsize=8, loc="upper left", framealpha=0.9)
    _style(ax,
           "Privacy–Utility Trade-off\n(arrows show improvement after PII masking)",
           "Answer Quality → F1 Score  (higher is better)",
           "Privacy Risk → PII Leakage Rate  (lower is better)")

    plt.tight_layout()
    out = PLOTS_DIR / "privacy_utility_tradeoff.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {out}")
    return out



def plot_pii_bars(p1, p2):
    labels = ["HotpotQA\nHippoRAG-2", "HotpotQA\nRAG-Anything",
              "PII-masking\nHippoRAG-2", "PII-masking\nRAG-Anything"]
    keys   = [("hotpotqa","hipporag"), ("hotpotqa","raganything"),
              ("pii_masking","hipporag"), ("pii_masking","raganything")]

    base_total = [p1[ds][fw]["total_pii_leaked"]           for ds,fw in keys]
    priv_total = [p2[ds][fw]["total_pii_leaked"]           for ds,fw in keys]
    base_avg   = [_ensure_entities(p1[ds][fw])["avg_entities_per_response"] for ds,fw in keys]
    priv_avg   = [_ensure_entities(p2[ds][fw])["avg_entities_per_response"] for ds,fw in keys]

    x     = np.arange(len(labels))
    w     = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(C_BG)

    def annotate_bars(ax, bars, vals, fmt="{}", color=None):
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.15,
                    fmt.format(v), ha="center", va="bottom",
                    fontsize=9, fontweight="bold",
                    color=color or "black")

  
    b1 = ax1.bar(x-w/2, base_total, w, label="Baseline", color=C_RED,   alpha=0.85, zorder=3)
    b2 = ax1.bar(x+w/2, priv_total, w, label="Private",  color=C_GREEN, alpha=0.85, zorder=3)
    annotate_bars(ax1, b1, base_total)
    annotate_bars(ax1, b2, priv_total, color=C_GREEN)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8)
    ax1.legend(fontsize=9)
    _style(ax1, "Total PII Instances Leaked\n(Baseline vs Privacy-Preserving)",
           "Framework × Dataset", "PII Instances")

    b3 = ax2.bar(x-w/2, base_avg, w, label="Baseline", color=C_RED,   alpha=0.85, zorder=3)
    b4 = ax2.bar(x+w/2, priv_avg, w, label="Private",  color=C_GREEN, alpha=0.85, zorder=3)
    annotate_bars(ax2, b3, base_avg, fmt="{:.2f}")
    annotate_bars(ax2, b4, priv_avg, fmt="{:.2f}", color=C_GREEN)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8)
    ax2.legend(fontsize=9)
    _style(ax2, "Avg PII Entities Per Response\n(Baseline vs Privacy-Preserving)",
           "Framework × Dataset", "Avg Entities Exposed")

    plt.tight_layout()
    out = PLOTS_DIR / "pii_leakage_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {out}")
    return out




def plot_f1(p1, p2):
    
    fw_labels = ["HippoRAG-2", "RAG-Anything"]
    fw_keys   = ["hipporag",   "raganything"]

    b_f1 = [p1["hotpotqa"][k]["avg_f1"] for k in fw_keys]
    p_f1 = [p2["hotpotqa"][k]["avg_f1"] for k in fw_keys]
    b_em = [p1["hotpotqa"][k]["avg_em"] for k in fw_keys]
    p_em = [p2["hotpotqa"][k]["avg_em"] for k in fw_keys]

    x = np.arange(len(fw_labels))
    w = 0.18
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(C_BG)

    bars = [
        ax.bar(x-1.5*w, b_f1, w, label="Baseline F1",  color=C_RED,   alpha=0.85, zorder=3),
        ax.bar(x-0.5*w, p_f1, w, label="Private F1",   color=C_GREEN, alpha=0.85, zorder=3),
        ax.bar(x+0.5*w, b_em, w, label="Baseline EM",  color=C_RED,   alpha=0.45, hatch="//", zorder=3),
        ax.bar(x+1.5*w, p_em, w, label="Private EM",   color=C_GREEN, alpha=0.45, hatch="//", zorder=3),
    ]
    vals = [b_f1, p_f1, b_em, p_em]
    for bar_group, vs in zip(bars, vals):
        for bar, v in zip(bar_group, vs):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.004,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)


    for i, (bf, pf) in enumerate(zip(b_f1, p_f1)):
        delta = pf - bf
        c = C_GREEN if delta >= -0.05 else "orange"
        ax.annotate(f"Δ={delta:+.3f}",
                    xy=(x[i]-0.5*w, pf),
                    xytext=(0, 14), textcoords="offset points",
                    ha="center", fontsize=8, color=c, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.8))

    ax.set_xticks(x); ax.set_xticklabels(fw_labels, fontsize=11)
    max_val = max(b_f1 + p_f1 + b_em + p_em)
    ax.set_ylim(0, min(1.0, max_val * 1.35))
    ax.legend(fontsize=9)
    _style(ax,
           "Answer Quality: Baseline vs Privacy-Preserving\n(HotpotQA — F1 and Exact Match)",
           "RAG Framework", "Score")

    plt.tight_layout()
    out = PLOTS_DIR / "f1_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {out}")
    return out




def main():
    from colorama import init, Fore, Style
    init(autoreset=True)
    print(f"\n{Fore.CYAN}Generating privacy–utility trade-off plots...{Style.RESET_ALL}\n")
    p1, p2 = load()
    plot_tradeoff(p1, p2)
    plot_pii_bars(p1, p2)
    plot_f1(p1, p2)
    print(f"\n{Fore.GREEN}✅ All 3 plots saved to {PLOTS_DIR}/{Style.RESET_ALL}")
    print(f"  → privacy_utility_tradeoff.png")
    print(f"  → pii_leakage_comparison.png")
    print(f"  → f1_comparison.png")
    print(f"\n{Fore.YELLOW}Include these charts in your presentation slides!{Style.RESET_ALL}\n")


if __name__ == "__main__":
    main()
