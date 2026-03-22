"""
S2P Touchstone File - S21 Peak-to-Peak Ripple Analysis
-------------------------------------------------------
Parses an .s2p file, computes the peak-to-peak variation in S21 magnitude (dB)
within a sliding 250 MHz window, and plots the result vs. frequency.

Usage:
    python s2p_ripple_analysis.py <path_to_file.s2p> [options]

Options:
    --window-mhz    Sliding window width in MHz (default: 250)
    --freq-step-mhz Expected frequency step in MHz (default: 50)
    --output        Save plot to file instead of displaying it (e.g. result.png)
"""

import argparse
import sys
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ---------------------------------------------------------------------------
# Touchstone .s2p parser
# ---------------------------------------------------------------------------

def parse_s2p(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse a 2-port Touchstone (.s2p) file.

    Returns
    -------
    frequencies : np.ndarray  — frequency values in Hz
    s21_db      : np.ndarray  — S21 magnitude in dB
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    freq_unit_mult = 1.0        # multiplier to convert to Hz
    data_format    = "MA"       # MA | DB | RI
    z0             = 50.0       # reference impedance (informational only)
    frequencies    = []
    s_raw          = []         # raw parameter columns per frequency point

    with filepath.open("r") as fh:
        for raw_line in fh:
            line = raw_line.strip()

            # Skip blank lines and comment lines
            if not line or line.startswith("!"):
                continue

            # Option line  (starts with #)
            if line.startswith("#"):
                # Format: # <freq_unit> S <format> R <z0>
                tokens = line.upper().split()
                unit_map = {"HZ": 1e0, "KHZ": 1e3, "MHZ": 1e6, "GHZ": 1e9}
                for i, tok in enumerate(tokens):
                    if tok in unit_map:
                        freq_unit_mult = unit_map[tok]
                    if tok in ("MA", "DB", "RI"):
                        data_format = tok
                continue

            # Strip inline comments
            line = re.split(r"\s*!", line)[0].strip()
            if not line:
                continue

            nums = list(map(float, line.split()))
            if not nums:
                continue

            # Each .s2p data line: freq  S11  S21  S12  S22
            # Each S-param is 2 values (mag+angle, dB+angle, or re+im).
            # A single frequency point may span multiple lines (rare but valid).
            if len(nums) >= 1 and not s_raw:
                # Start of a new frequency point
                frequencies.append(nums[0])
                s_raw.append(nums[1:])
            else:
                # Check if this is a new frequency (first token differs from last)
                # A simple heuristic: if total accumulated values for last point < 8
                # this is a continuation line; otherwise it's a new frequency row.
                if len(s_raw[-1]) < 8:
                    s_raw[-1].extend(nums)
                else:
                    frequencies.append(nums[0])
                    s_raw.append(nums[1:])

    frequencies = np.array(frequencies) * freq_unit_mult   # convert to Hz

    # Extract S21 (index 1 in the parameter order: S11, S21, S12, S22)
    # Each parameter occupies 2 values → S21 starts at offset 2 (0-based)
    s21_db = np.empty(len(frequencies))
    for i, row in enumerate(s_raw):
        if len(row) < 4:
            raise ValueError(
                f"Line {i+1}: expected at least 4 values after frequency, got {len(row)}"
            )
        val_a, val_b = row[2], row[3]

        if data_format == "DB":
            s21_db[i] = val_a                           # already in dB
        elif data_format == "MA":
            s21_db[i] = 20.0 * np.log10(max(abs(val_a), 1e-300))
        elif data_format == "RI":
            mag = np.sqrt(val_a**2 + val_b**2)
            s21_db[i] = 20.0 * np.log10(max(mag, 1e-300))
        else:
            raise ValueError(f"Unknown data format: {data_format}")

    return frequencies, s21_db


# ---------------------------------------------------------------------------
# Sliding-window peak-to-peak analysis
# ---------------------------------------------------------------------------

def sliding_peak_to_peak(
    frequencies: np.ndarray,
    s21_db: np.ndarray,
    window_hz: float,
    freq_step_hz: float,
) -> np.ndarray:
    """
    For every frequency point f_i, compute max(S21_dB) - min(S21_dB) over a
    window of width window_hz centred on f_i.

    Edge handling — the window always spans exactly n_pts points:
      - Interior points  : symmetric, [f_i - half_window, f_i + half_window]
      - Near low  edge   : window shifts right so it starts at f_min
      - Near high edge   : window shifts left  so it ends   at f_max

    The transition from forward-looking to symmetric to backward-looking
    occurs naturally once the centre point is more than half a window width
    away from each edge.

    Returns
    -------
    ptp : np.ndarray, same length as frequencies
        Peak-to-peak ripple in dB at each frequency point.
    """
    n_pts = int(round(window_hz / freq_step_hz)) + 1   # total points in window
    half  = n_pts // 2                                  # points to each side
    n     = len(s21_db)
    ptp   = np.empty(n)

    for i in range(n):
        start = i - half
        end   = start + n_pts      # exclusive

        # Clamp to array bounds, shifting the whole window rather than truncating
        if start < 0:
            start = 0
            end   = n_pts
        if end > n:
            end   = n
            start = n - n_pts

        ptp[i] = s21_db[start:end].max() - s21_db[start:end].min()

    return ptp


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(
    frequencies: np.ndarray,
    s21_db: np.ndarray,
    ptp: np.ndarray,
    window_mhz: float,
    limit_db: float | None,
    output_path: str | None,
) -> None:
    freq_ghz = frequencies / 1e9
    valid    = np.ones(len(ptp), dtype=bool)   # all points now have a value

    fig, axes = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={"hspace": 0.08}
    )
    fig.suptitle(
        f"S21 Analysis — {window_mhz:.0f} MHz Sliding Window",
        fontsize=14, fontweight="bold", y=0.98
    )

    # --- Top plot: S21 magnitude ---
    ax0 = axes[0]
    ax0.plot(freq_ghz, s21_db, color="#1f77b4", linewidth=1.0, label="S21 magnitude")
    ax0.set_ylabel("S21 Magnitude (dB)", fontsize=11)
    ax0.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax0.legend(loc="upper right", fontsize=9)
    ax0.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # --- Bottom plot: peak-to-peak ripple ---
    ax1 = axes[1]
    ax1.plot(
        freq_ghz[valid], ptp[valid],
        color="#d62728", linewidth=1.0,
        label=f"Peak-to-peak within {window_mhz:.0f} MHz window"
    )
    ax1.fill_between(freq_ghz[valid], 0, ptp[valid], alpha=0.15, color="#d62728")
    if limit_db is not None:
        ax1.axhline(
            limit_db, color="#ff7f0e", linewidth=1.5,
            linestyle="--", label=f"Limit: {limit_db:.2f} dB"
        )
        ax1.legend(loc="upper right", fontsize=9)
    ax1.set_ylabel("Peak-to-Peak Ripple (dB)", fontsize=11)
    ax1.set_xlabel("Frequency (GHz)", fontsize=11)
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_ylim(bottom=0)
    ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # Shared x-axis formatting
    for ax in axes:
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(axis="both", which="both", labelsize=9)

    # Annotate max ripple point
    if valid.any():
        peak_idx = np.nanargmax(ptp)
        ax1.annotate(
            f"Max: {ptp[peak_idx]:.2f} dB @ {freq_ghz[peak_idx]:.3f} GHz",
            xy=(freq_ghz[peak_idx], ptp[peak_idx]),
            xytext=(10, 10), textcoords="offset points",
            fontsize=8, color="#d62728",
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.8),
        )

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze S21 peak-to-peak ripple from a .s2p touchstone file."
    )
    parser.add_argument("s2p_file", help="Path to the .s2p touchstone file")
    parser.add_argument(
        "--window-mhz", type=float, default=250.0,
        help="Sliding window width in MHz (default: 250)"
    )
    parser.add_argument(
        "--freq-step-mhz", type=float, default=50.0,
        help="Frequency step size in MHz (default: 50)"
    )
    parser.add_argument(
        "--limit-db", type=float, default=None,
        help="Draw a horizontal limit line at this ripple level in dB (e.g. 0.5)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save plot to this file path instead of displaying (e.g. result.png)"
    )
    args = parser.parse_args()

    window_hz    = args.window_mhz    * 1e6
    freq_step_hz = args.freq_step_mhz * 1e6

    print(f"Parsing: {args.s2p_file}")
    frequencies, s21_db = parse_s2p(args.s2p_file)

    print(f"  Frequency range : {frequencies[0]/1e9:.4f} – {frequencies[-1]/1e9:.4f} GHz")
    print(f"  Number of points: {len(frequencies)}")
    print(f"  S21 range       : {s21_db.min():.2f} dB  to  {s21_db.max():.2f} dB")

    # Validate frequency step
    actual_steps = np.diff(frequencies)
    median_step  = np.median(actual_steps)
    if not np.isclose(median_step, freq_step_hz, rtol=0.05):
        print(
            f"\n  WARNING: Median frequency step in file ({median_step/1e6:.2f} MHz) "
            f"differs from --freq-step-mhz ({args.freq_step_mhz:.0f} MHz). "
            "Ripple window may be inaccurate — consider updating --freq-step-mhz."
        )

    ptp = sliding_peak_to_peak(frequencies, s21_db, window_hz, freq_step_hz)
    valid_ptp = ptp

    print(f"\n  Window size     : {args.window_mhz:.0f} MHz  ({int(round(window_hz/freq_step_hz))+1} points)")
    if valid_ptp.size:
        peak_idx = np.nanargmax(ptp)
        print(f"  Max ripple      : {valid_ptp.max():.3f} dB  @ {frequencies[peak_idx]/1e9:.4f} GHz")
        print(f"  Mean ripple     : {valid_ptp.mean():.3f} dB")
        print(f"  Median ripple   : {np.median(valid_ptp):.3f} dB")

    plot_results(frequencies, s21_db, ptp, args.window_mhz, args.limit_db, args.output)


if __name__ == "__main__":
    main()
