"""
QKD Protocol Visualization Dashboard.

Two dashboard functions:

  show_dashboard()      — Single-run summary (9 panels on a 3x3 grid)
  show_analysis_dashboard() — Key-rate analysis (4 panels: distance sweep,
                              QBER sweep, Eve attack, finite-key epsilon)

All data is passed in via plain dicts / result objects so the visualiser is
completely decoupled from the protocol implementation.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


# ===================================================================
# Colour palette (colour-blind friendly)
# ===================================================================

C_ALICE   = "#2196F3"   # blue
C_BOB     = "#4CAF50"   # green
C_EVE     = "#F44336"   # red
C_KEY     = "#FF9800"   # orange
C_SECURE  = "#00BCD4"   # teal
C_LEAKED  = "#9E9E9E"   # grey
C_DECOY   = "#AB47BC"   # purple
C_FINITE  = "#26A69A"   # green-teal
C_BG      = "#FAFAFA"
C_DARK    = "#212121"

SERIES_COLORS = [C_ALICE, C_EVE, C_DECOY, C_FINITE, C_KEY, C_BOB, C_SECURE]


# ===================================================================
# Dashboard 1: Single-run protocol summary  (3x3 grid)
# ===================================================================

def show_dashboard(stats: dict[str, Any], save_path: str | None = None) -> None:
    """
    Render a 3x3 panel dashboard for a single QKD run.

    Panels:
      Row 1: Key waterfall | QBER bars | Protocol flow
      Row 2: Basis sifting | Security analysis | Bit randomness
      Row 3: CASCADE detail | EC efficiency | Finite-key summary
    """
    fig = plt.figure(figsize=(19, 14), facecolor=C_BG)
    fig.suptitle(
        "BB84 Quantum Key Distribution — Protocol Dashboard",
        fontsize=17, fontweight="bold", color=C_DARK, y=0.98,
    )

    gs = GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.34,
                  left=0.06, right=0.96, top=0.93, bottom=0.04)

    _plot_key_waterfall(fig.add_subplot(gs[0, 0]), stats)
    _plot_error_rates(fig.add_subplot(gs[0, 1]), stats)
    _plot_protocol_flow(fig.add_subplot(gs[0, 2]), stats)
    _plot_basis_match(fig.add_subplot(gs[1, 0]), stats)
    _plot_security_analysis(fig.add_subplot(gs[1, 1]), stats)
    _plot_bit_randomness(fig.add_subplot(gs[1, 2]), stats)
    _plot_cascade_detail(fig.add_subplot(gs[2, 0]), stats)
    _plot_ec_efficiency(fig.add_subplot(gs[2, 1]), stats)
    _plot_finite_key_summary(fig.add_subplot(gs[2, 2]), stats)

    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight", facecolor=C_BG)
        print(f"Dashboard saved to {save_path}")
    plt.show()


# ===================================================================
# Dashboard 2: Key-rate analysis (2x2 grid)
# ===================================================================

def show_analysis_dashboard(
    distance_sweep=None,
    qber_sweep=None,
    eve_sweep=None,
    epsilon_data=None,
    save_path: str | None = None,
) -> None:
    """
    Render a 2x2 analysis dashboard.

    Args:
        distance_sweep: KeyRateSweepResult from sweep_distance().
        qber_sweep:     KeyRateSweepResult from sweep_qber().
        eve_sweep:      KeyRateSweepResult from sweep_eve_interception().
        epsilon_data:   list of (epsilon, key_rate) tuples from key_rate_vs_epsilon().
    """
    fig = plt.figure(figsize=(16, 11), facecolor=C_BG)
    fig.suptitle(
        "BB84 QKD — Security & Key-Rate Analysis",
        fontsize=17, fontweight="bold", color=C_DARK, y=0.97,
    )

    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30,
                  left=0.08, right=0.96, top=0.91, bottom=0.07)

    if distance_sweep is not None:
        _plot_key_rate_sweep(fig.add_subplot(gs[0, 0]), distance_sweep,
                             title="Secret Key Rate vs Distance",
                             log_y=True)

    if qber_sweep is not None:
        _plot_key_rate_sweep(fig.add_subplot(gs[0, 1]), qber_sweep,
                             title="Secret Key Rate vs QBER",
                             log_y=False)

    if eve_sweep is not None:
        _plot_eve_analysis(fig.add_subplot(gs[1, 0]), eve_sweep)

    if epsilon_data is not None:
        _plot_epsilon_sensitivity(fig.add_subplot(gs[1, 1]), epsilon_data)

    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight", facecolor=C_BG)
        print(f"Analysis dashboard saved to {save_path}")
    plt.show()


# ===================================================================
# Row 1 panels (existing, refined)
# ===================================================================

def _plot_key_waterfall(ax: plt.Axes, s: dict) -> None:
    stages = ["Raw", "Sifted", "Corrected", "Amplified", "AES-256"]
    values = [
        s.get("raw_bits", 0),
        s.get("sifted_bits", 0),
        s.get("corrected_bits", 0),
        s.get("amplified_bits", 0),
        s.get("aes_key_bits", 256),
    ]
    colors = ["#90CAF9", "#64B5F6", "#42A5F5", "#1E88E5", C_SECURE]

    bars = ax.bar(stages, values, color=colors, edgecolor="white", linewidth=1.2)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02,
                str(v), ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_title("Key Length by Stage (bits)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Bits")
    ax.set_ylim(0, max(values) * 1.18)
    ax.tick_params(axis="x", rotation=30)
    _style_ax(ax)


def _plot_error_rates(ax: plt.Axes, s: dict) -> None:
    before = s.get("qber_before", 0) * 100
    after  = s.get("qber_after", 0) * 100
    threshold = s.get("error_threshold", 0.11) * 100

    x = np.array([0, 1])
    bars = ax.bar(x, [before, after], width=0.5,
                  color=[C_EVE, C_SECURE], edgecolor="white", linewidth=1.2)
    ax.axhline(threshold, color="#FF5722", ls="--", lw=1.5,
               label=f"Security threshold ({threshold:.0f}%)")

    ax.set_xticks(x)
    ax.set_xticklabels(["Before CASCADE", "After CASCADE"])
    ax.set_ylabel("Error Rate (%)")
    ax.set_title("Quantum Bit Error Rate", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(before, threshold) * 1.4 + 1)
    ax.legend(fontsize=8, loc="upper right")

    for bar, v in zip(bars, [before, after]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{v:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    _style_ax(ax)


def _plot_protocol_flow(ax: plt.Axes, s: dict) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Protocol Flow", fontsize=11, fontweight="bold")

    # Alice
    alice_box = mpatches.FancyBboxPatch(
        (0.3, 2), 2.4, 2, boxstyle="round,pad=0.2",
        facecolor=C_ALICE, edgecolor="white", linewidth=2, alpha=0.9)
    ax.add_patch(alice_box)
    ax.text(1.5, 3, "ALICE", ha="center", va="center",
            fontsize=12, fontweight="bold", color="white")

    # Bob
    bob_box = mpatches.FancyBboxPatch(
        (7.3, 2), 2.4, 2, boxstyle="round,pad=0.2",
        facecolor=C_BOB, edgecolor="white", linewidth=2, alpha=0.9)
    ax.add_patch(bob_box)
    ax.text(8.5, 3, "BOB", ha="center", va="center",
            fontsize=12, fontweight="bold", color="white")

    # Quantum channel
    ax.annotate("", xy=(7.1, 3.5), xytext=(2.9, 3.5),
                arrowprops=dict(arrowstyle="-|>", color=C_KEY, lw=2.5))
    ax.text(5, 3.85, "Quantum Channel", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color=C_KEY)
    ax.text(5, 3.45, "|0\u27E9  |1\u27E9  |+\u27E9  |-\u27E9", ha="center",
            fontsize=8, color=C_DARK, family="monospace")

    # Classical channel
    ax.annotate("", xy=(7.1, 2.4), xytext=(2.9, 2.4),
                arrowprops=dict(arrowstyle="<|-|>", color=C_LEAKED, lw=1.8))
    ax.text(5, 2.0, "Classical Channel", ha="center", va="top",
            fontsize=9, color=C_LEAKED)
    ax.text(5, 1.55, "bases / parities / PA seed", ha="center",
            fontsize=7, color=C_LEAKED, style="italic")

    # Eve
    if s.get("eavesdropper_active"):
        eve_box = mpatches.FancyBboxPatch(
            (3.8, 4.5), 2.4, 1.2, boxstyle="round,pad=0.15",
            facecolor=C_EVE, edgecolor="white", linewidth=2, alpha=0.85)
        ax.add_patch(eve_box)
        ax.text(5, 5.1, "EVE", ha="center", va="center",
                fontsize=11, fontweight="bold", color="white")
        ax.annotate("", xy=(5, 4.5), xytext=(5, 3.9),
                    arrowprops=dict(arrowstyle="-|>", color=C_EVE, lw=1.5, ls="--"))

    # Status badge
    status = "SECURE" if s.get("qber_before", 0) < s.get("error_threshold", 0.11) else "COMPROMISED"
    badge_col = C_SECURE if status == "SECURE" else C_EVE
    badge = mpatches.FancyBboxPatch(
        (3.5, 0.2), 3, 0.9, boxstyle="round,pad=0.15",
        facecolor=badge_col, edgecolor="white", linewidth=1.5)
    ax.add_patch(badge)
    ax.text(5, 0.65, status, ha="center", va="center",
            fontsize=11, fontweight="bold", color="white")


# ===================================================================
# Row 2 panels
# ===================================================================

def _plot_basis_match(ax: plt.Axes, s: dict) -> None:
    match   = s.get("basis_match_count", 50)
    mismatch = s.get("basis_mismatch_count", 50)
    total = match + mismatch
    eff = match / total * 100 if total > 0 else 50.0
    sizes = [match, mismatch]
    labels = [f"Matched\n({match})", f"Discarded\n({mismatch})"]
    colors = [C_ALICE, "#E0E0E0"]

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, textprops={"fontsize": 9},
        wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    for at in autotexts:
        at.set_fontweight("bold")

    # Show biased label if efficiency > 55%
    bias_label = "Biased" if eff > 55 else "Standard"
    ax.set_title(f"Basis Sifting ({bias_label}, {eff:.0f}%)",
                 fontsize=11, fontweight="bold")


def _plot_security_analysis(ax: plt.Axes, s: dict) -> None:
    total = s.get("sifted_bits", 100)
    leaked = s.get("cascade_leaked", 0)
    secure = s.get("amplified_bits", 0)
    discarded = max(0, total - leaked - secure)

    cats = ["Key Composition"]
    ax.barh(cats, [secure], color=C_SECURE, label=f"Secure ({secure})")
    ax.barh(cats, [leaked], left=[secure], color=C_LEAKED, label=f"Leaked ({leaked})")
    ax.barh(cats, [discarded], left=[secure + leaked],
            color="#FFCDD2", label=f"Discarded ({discarded})")

    ax.set_xlim(0, total * 1.05)
    ax.set_xlabel("Bits")
    ax.set_title("Security Analysis", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    _style_ax(ax)


def _plot_bit_randomness(ax: plt.Axes, s: dict) -> None:
    key = s.get("final_key", [])
    if not key:
        ax.text(0.5, 0.5, "No key data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Key Bit Distribution", fontsize=11, fontweight="bold")
        return

    zeros = key.count(0)
    ones  = key.count(1)
    bars = ax.bar(["0", "1"], [zeros, ones], color=[C_ALICE, C_KEY],
                  edgecolor="white", linewidth=1.2, width=0.5)
    ax.axhline(len(key) / 2, color=C_LEAKED, ls="--", lw=1, label="Ideal (50%)")

    for bar, v in zip(bars, [zeros, ones]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + len(key) * 0.01,
                str(v), ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title("Key Bit Distribution", fontsize=11, fontweight="bold")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    _style_ax(ax)


# ===================================================================
# Row 3 panels (new)
# ===================================================================

def _plot_cascade_detail(ax: plt.Axes, s: dict) -> None:
    """Per-pass CASCADE statistics bar chart."""
    pass_stats = s.get("cascade_pass_stats", [])
    if not pass_stats:
        # Fallback: show summary text
        ax.axis("off")
        ax.set_title("CASCADE Detail", fontsize=11, fontweight="bold")
        lines = [
            f"Errors corrected: {s.get('errors_corrected', '?')}",
            f"Parity bits leaked: {s.get('cascade_leaked', '?')}",
            f"Passes: {s.get('cascade_passes', '?')}",
            f"EC efficiency (f): {s.get('ec_efficiency', '?'):.2f}"
                if isinstance(s.get('ec_efficiency'), (int, float)) else "",
        ]
        ax.text(0.5, 0.5, "\n".join(l for l in lines if l),
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, family="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#E3F2FD"))
        return

    passes = [f"Pass {ps['pass_number']}" for ps in pass_stats]
    errors = [ps["errors_found"] for ps in pass_stats]
    parities = [ps["parity_bits_exchanged"] for ps in pass_stats]

    x = np.arange(len(passes))
    w = 0.35
    ax.bar(x - w/2, errors, w, label="Errors found", color=C_EVE, edgecolor="white")
    ax.bar(x + w/2, parities, w, label="Parities exchanged", color=C_LEAKED, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(passes, fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title("CASCADE Per-Pass Detail", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    _style_ax(ax)


def _plot_ec_efficiency(ax: plt.Axes, s: dict) -> None:
    """Gauge-style display of EC efficiency factor."""
    f_ec = s.get("ec_efficiency", 1.16)
    if not isinstance(f_ec, (int, float)):
        f_ec = 1.16

    ax.set_xlim(0, 3)
    ax.set_ylim(0, 2)
    ax.axis("off")
    ax.set_title("Error Correction Efficiency", fontsize=11, fontweight="bold")

    # Background arc
    theta = np.linspace(math.pi, 0, 100)
    r = 0.8
    cx, cy = 1.5, 0.6
    ax.plot(cx + r * np.cos(theta), cy + r * np.sin(theta),
            color="#E0E0E0", lw=12, solid_capstyle="round")

    # Colored arc (proportion: f=1.0 is best=full green, f=2.0 is worst=full red)
    frac = min(max((f_ec - 1.0) / 0.5, 0), 1.0)  # 0..1 maps 1.0..1.5
    fill_angle = math.pi * (1 - frac)
    theta_fill = np.linspace(math.pi, fill_angle, 50)
    color = C_SECURE if f_ec < 1.15 else (C_KEY if f_ec < 1.25 else C_EVE)
    ax.plot(cx + r * np.cos(theta_fill), cy + r * np.sin(theta_fill),
            color=color, lw=12, solid_capstyle="round")

    # Value text
    ax.text(cx, cy - 0.05, f"f = {f_ec:.3f}", ha="center", va="center",
            fontsize=16, fontweight="bold", color=C_DARK)
    ax.text(cx, cy - 0.35, "Shannon limit: f = 1.000", ha="center",
            fontsize=8, color=C_LEAKED)

    # Labels
    ax.text(cx - r - 0.1, cy - 0.1, "1.0", fontsize=8, ha="center", color=C_SECURE)
    ax.text(cx + r + 0.1, cy - 0.1, "1.5+", fontsize=8, ha="center", color=C_EVE)

    # Secure key rate
    rate = s.get("secure_key_rate_per_pulse", None)
    if rate is not None and isinstance(rate, (int, float)):
        ax.text(cx, cy + 0.85, f"Key rate: {rate:.2e} bits/pulse",
                ha="center", fontsize=9, fontweight="bold", color=C_DARK,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9"))


def _plot_finite_key_summary(ax: plt.Axes, s: dict) -> None:
    """Text panel showing finite-key security parameters."""
    ax.axis("off")
    ax.set_title("Finite-Key Security", fontsize=11, fontweight="bold")

    fk = s.get("finite_key", {})
    eps = fk.get("epsilon_security", s.get("epsilon_security", "N/A"))
    eps_str = f"{eps:.1e}" if isinstance(eps, float) else str(eps)
    qber_ub = fk.get("qber_upper_bound", None)
    qber_ub_str = f"{qber_ub:.4f}" if isinstance(qber_ub, float) else "N/A"
    secure_len = fk.get("secure_key_length", s.get("amplified_bits", "?"))
    min_N = fk.get("min_block_size", "N/A")

    channel_dist = s.get("channel_distance_km", None)
    sifting_eff = s.get("sifting_efficiency", None)
    decoy = s.get("decoy_enabled", False)

    lines = [
        ("Security param (eps)", eps_str),
        ("QBER upper bound", qber_ub_str),
        ("Secure key length", f"{secure_len} bits"),
        ("Min block for pos. rate", f"{min_N:,}" if isinstance(min_N, int) else str(min_N)),
    ]
    if channel_dist is not None:
        lines.insert(0, ("Channel distance", f"{channel_dist} km"))
    if sifting_eff is not None:
        lines.insert(1, ("Sifting efficiency", f"{sifting_eff:.1%}"))
    if decoy:
        lines.append(("Decoy states", "ENABLED"))

    y = 0.92
    for label, value in lines:
        ax.text(0.05, y, label + ":", transform=ax.transAxes,
                fontsize=9, color=C_LEAKED, fontweight="bold")
        ax.text(0.95, y, value, transform=ax.transAxes,
                fontsize=9, color=C_DARK, ha="right", family="monospace")
        y -= 0.13

    # Colour-coded security badge
    if isinstance(eps, float):
        if eps <= 1e-10:
            badge_text, badge_col = "ULTRA-SECURE", C_SECURE
        elif eps <= 1e-6:
            badge_text, badge_col = "HIGH SECURITY", C_BOB
        else:
            badge_text, badge_col = "MODERATE", C_KEY
    else:
        badge_text, badge_col = "ASYMPTOTIC", C_LEAKED

    badge = mpatches.FancyBboxPatch(
        (0.25, 0.01), 0.5, 0.1, boxstyle="round,pad=0.02",
        facecolor=badge_col, edgecolor="white", linewidth=1.2,
        transform=ax.transAxes)
    ax.add_patch(badge)
    ax.text(0.5, 0.06, badge_text, transform=ax.transAxes,
            ha="center", va="center", fontsize=9, fontweight="bold", color="white")


# ===================================================================
# Analysis dashboard panels
# ===================================================================

def _plot_key_rate_sweep(ax: plt.Axes, sweep, title: str, log_y: bool = True) -> None:
    """Generic line plot for key-rate sweeps."""
    for i, (label, y_vals) in enumerate(sweep.series):
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        # Filter out zeros for log scale
        y_arr = np.array(y_vals, dtype=float)
        x_arr = np.array(sweep.x_values, dtype=float)

        if log_y:
            mask = y_arr > 0
            if mask.any():
                ax.semilogy(x_arr[mask], y_arr[mask], "-o", markersize=3,
                            color=color, label=label, linewidth=1.8)
        else:
            ax.plot(x_arr, y_arr, "-o", markersize=3,
                    color=color, label=label, linewidth=1.8)

    ax.set_xlabel(sweep.x_label, fontsize=10)
    ax.set_ylabel("Key Rate (bits/pulse)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    _style_ax(ax)


def _plot_eve_analysis(ax: plt.Axes, sweep) -> None:
    """Dual-axis plot: key rate and QBER vs Eve interception rate."""
    x = np.array(sweep.x_values, dtype=float)
    key_rates = np.array(sweep.series[0][1], dtype=float)
    qbers = np.array(sweep.series[1][1], dtype=float)

    # Key rate on left axis
    color1 = C_SECURE
    ax.plot(x, key_rates, "-o", markersize=3, color=color1,
            label="Key Rate", linewidth=2)
    ax.set_xlabel(sweep.x_label, fontsize=10)
    ax.set_ylabel("Key Rate (bits/pulse)", color=color1, fontsize=10)
    ax.tick_params(axis="y", labelcolor=color1)

    # QBER on right axis
    ax2 = ax.twinx()
    color2 = C_EVE
    ax2.plot(x, qbers, "--s", markersize=3, color=color2,
             label="QBER", linewidth=1.8)
    ax2.set_ylabel("QBER (%)", color=color2, fontsize=10)
    ax2.tick_params(axis="y", labelcolor=color2)

    # Threshold line
    ax2.axhline(11.0, color="#FF5722", ls=":", lw=1.5, alpha=0.7)
    ax2.text(x[-1], 11.5, "11% threshold", fontsize=7, color="#FF5722", ha="right")

    # Find abort point
    abort_idx = np.argmax(qbers > 11.0) if np.any(qbers > 11.0) else None
    if abort_idx is not None and abort_idx > 0:
        ax.axvline(x[abort_idx], color=C_EVE, ls="--", lw=1, alpha=0.5)
        ax.text(x[abort_idx] + 1, max(key_rates) * 0.9, "ABORT",
                fontsize=8, color=C_EVE, fontweight="bold")

    ax.set_title("Eavesdropper Impact Analysis", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.2)
    _style_ax(ax)


def _plot_epsilon_sensitivity(ax: plt.Axes, epsilon_data: list[tuple[float, float]]) -> None:
    """Key rate vs security parameter epsilon."""
    epsilons = [e for e, _ in epsilon_data]
    rates = [r for _, r in epsilon_data]

    ax.semilogx(epsilons, rates, "-o", markersize=5, color=C_DECOY,
                linewidth=2, markeredgecolor="white", markeredgewidth=0.5)

    ax.set_xlabel("Security Parameter (epsilon)", fontsize=10)
    ax.set_ylabel("Key Rate (bits/pulse)", fontsize=10)
    ax.set_title("Finite-Key: Rate vs Security Tightness", fontsize=11, fontweight="bold")

    # Annotate sweet spots
    if rates:
        # Highlight 1e-10 region
        for eps, rate in epsilon_data:
            if abs(math.log10(eps) - (-10)) < 0.5 and rate > 0:
                ax.annotate(f"eps={eps:.0e}\nR={rate:.2e}",
                           xy=(eps, rate), xytext=(eps * 50, rate * 1.3),
                           fontsize=7, color=C_DARK,
                           arrowprops=dict(arrowstyle="->", color=C_LEAKED, lw=0.8))
                break

    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    _style_ax(ax)


# ===================================================================
# Styling
# ===================================================================

def _style_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)
