"""
S2P Touchstone File - S21 Group Delay Peak-to-Peak Analysis
------------------------------------------------------------
Parses an .s2p file, derives S21 group delay via numerical differentiation
of the unwrapped phase, computes the peak-to-peak variation in group delay (ns)
within a sliding window, and plots the result vs. frequency.

Group delay is defined as:
    τ(f) = -dφ/dω  =  -dφ / (2π · df)

where φ is the unwrapped phase of S21 in radians.

Usage:
    python s2p_groupdelay_analysis.py <path_to_file.s2p> [options]

Options:
    --window-mhz    Sliding window width in MHz (default: 250)
    --freq-step-mhz Expected frequency step in MHz (default: 50)
    --limit-ns      Draw a horizontal limit line at this group delay variation in ns (e.g. 0.5)
    --output        Save plot to file instead of displaying it (e.g. result.png)
"""

import argparse
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
    frequencies  : np.ndarray — frequency values in Hz
    group_delay  : np.ndarray — S21 group delay in nanoseconds
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    freq_unit_mult = 1.0        # multiplier to convert to Hz
    data_format    = "MA"       # MA | DB | RI
    frequencies    = []
    s_raw          = []         # raw parameter columns per frequency point

    with filepath.open("r") as fh:
        for raw_line in fh:
            line = raw_line.strip()

            # Skip blank lines and comment lines
            if not line or line.startswith("!"):
                continue

            # Option line (starts with #)
            if line.startswith("#"):
                # Format: # <freq_unit> S <format> R <z0>
                tokens = line.upper().split()
                unit_map = {"HZ": 1e0, "KHZ": 1e3, "MHZ": 1e6, "GHZ": 1e9}
                for tok in tokens:
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
            if not s_raw:
                frequencies.append(nums[0])
                s_raw.append(nums[1:])
            else:
                if len(s_raw[-1]) < 8:
                    s_raw[-1].extend(nums)
                else:
                    frequencies.append(nums[0])
                    s_raw.append(nums[1:])

    frequencies = np.array(frequencies) * freq_unit_mult   # convert to Hz

    # Extract S21 phase (radians)
    # S21 is the 2nd parameter: offset 2–3 in the per-row value list (0-based)
    phase_rad = np.empty(len(frequencies))
    for i, row in enumerate(s_raw):
        if len(row) < 4:
            raise ValueError(
                f"Row {i+1}: expected at least 4 values after frequency, got {len(row)}"
            )
        val_a, val_b = row[2], row[3]

        if data_format in ("MA", "DB"):
            # val_b is the angle in degrees
            phase_rad[i] = np.deg2rad(val_b)
        elif data_format == "RI":
            # val_a = real, val_b = imaginary
            phase_rad[i] = np.arctan2(val_b, val_a)
        else:
            raise ValueError(f"Unknown data format: {data_format}")

    # Unwrap phase to remove 2π discontinuities before differentiating
    phase_unwrapped = np.unwrap(phase_rad)

    # Group delay: τ = -dφ/dω = -dφ / (2π · df)
    # np.gradient uses central differences internally, with one-sided at the edges
    dphi = np.gradient(phase_unwrapped, frequencies)   # dφ/df  (rad/Hz)
    group_delay_s  = -dphi / (2.0 * np.pi)            # seconds
    group_delay_ns = group_delay_s * 1e9               # nanoseconds

    return frequencies, group_delay_ns


# ---------------------------------------------------------------------------
# Sliding-window peak-to-peak analysis
# ---------------------------------------------------------------------------

def sliding_peak_to_peak(
    frequencies: np.ndarray,
    group_delay_ns: np.ndarray,
    window_hz: float,
    freq_step_hz: float,
) -> np.ndarray:
    """
    For every frequency point f_i, collect all samples in the window
    [f_i, f_i + window_hz) and compute max(GD) - min(GD).

    The window is *forward-looking* from each point. Near the end of the
    frequency range, the window shrinks naturally rather than being dropped.

    Returns
    -------
    ptp : np.ndarray, same length as frequencies
        Peak-to-peak group delay variation in ns at each frequency point.
    """
    n_pts_in_window = int(round(window_hz / freq_step_hz)) + 1  # inclusive
    n = len(group_delay_ns)
    ptp = np.empty(n)

    for i in range(n):
        end_idx = i + n_pts_in_window          # exclusive upper bound
        window_data = group_delay_ns[i:end_idx]  # slices to end of array naturally
        ptp[i] = window_data.max() - window_data.min()

    return ptp


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(
    frequencies: np.ndarray,
    group_delay_ns: np.ndarray,
    ptp: np.ndarray,
    window_mhz: float,
    limit_ns: float | None,
    output_path: str | None,
) -> None:
    freq_ghz = frequencies / 1e9

    fig, axes = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={"hspace": 0.08}
    )
    fig.suptitle(
        f"S21 Group Delay Analysis — {window_mhz:.0f} MHz Sliding Window",
        fontsize=14, fontweight="bold", y=0.98
    )

    # --- Top plot: group delay ---
    ax0 = axes[0]
    ax0.plot(freq_ghz, group_delay_ns, color="#1f77b4", linewidth=1.0, label="S21 group delay")
    ax0.set_ylabel("Group Delay (ns)", fontsize=11)
    ax0.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax0.legend(loc="upper right", fontsize=9)
    ax0.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # --- Bottom plot: peak-to-peak group delay variation ---
    ax1 = axes[1]
    ax1.plot(
        freq_ghz, ptp,
        color="#d62728", linewidth=1.0,
        label=f"Peak-to-peak within {window_mhz:.0f} MHz window"
    )
    ax1.fill_between(freq_ghz, 0, ptp, alpha=0.15, color="#d62728")
    if limit_ns is not None:
        ax1.axhline(
            limit_ns, color="#ff7f0e", linewidth=1.5,
            linestyle="--", label=f"Limit: {limit_ns:.2f} ns"
        )
    ax1.set_ylabel("Peak-to-Peak Group Delay Variation (ns)", fontsize=11)
    ax1.set_xlabel("Frequency (GHz)", fontsize=11)
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_ylim(bottom=0)
    ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # Shared x-axis formatting
    for ax in axes:
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(axis="both", which="both", labelsize=9)

    # Annotate max variation point
    peak_idx = int(np.argmax(ptp))
    ax1.annotate(
        f"Max: {ptp[peak_idx]:.3f} ns @ {freq_ghz[peak_idx]:.3f} GHz",
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
        description="Analyze S21 group delay peak-to-peak variation from a .s2p touchstone file."
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
        "--limit-ns", type=float, default=None,
        help="Draw a horizontal limit line at this group delay variation in ns (e.g. 0.5)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save plot to this file path instead of displaying (e.g. result.png)"
    )
    args = parser.parse_args()

    window_hz    = args.window_mhz    * 1e6
    freq_step_hz = args.freq_step_mhz * 1e6

    print(f"Parsing: {args.s2p_file}")
    frequencies, group_delay_ns = parse_s2p(args.s2p_file)

    print(f"  Frequency range   : {frequencies[0]/1e9:.4f} – {frequencies[-1]/1e9:.4f} GHz")
    print(f"  Number of points  : {len(frequencies)}")
    print(f"  Group delay range : {group_delay_ns.min():.3f} ns  to  {group_delay_ns.max():.3f} ns")

    # Validate frequency step
    actual_steps = np.diff(frequencies)
    median_step  = np.median(actual_steps)
    if not np.isclose(median_step, freq_step_hz, rtol=0.05):
        print(
            f"\n  WARNING: Median frequency step in file ({median_step/1e6:.2f} MHz) "
            f"differs from --freq-step-mhz ({args.freq_step_mhz:.0f} MHz). "
            "Window size may be inaccurate — consider updating --freq-step-mhz."
        )

    ptp = sliding_peak_to_peak(frequencies, group_delay_ns, window_hz, freq_step_hz)

    print(f"\n  Window size       : {args.window_mhz:.0f} MHz  ({int(round(window_hz/freq_step_hz))+1} points)")
    peak_idx = int(np.argmax(ptp))
    print(f"  Max variation     : {ptp.max():.3f} ns  @ {frequencies[peak_idx]/1e9:.4f} GHz")
    print(f"  Mean variation    : {ptp.mean():.3f} ns")
    print(f"  Median variation  : {np.median(ptp):.3f} ns")

    plot_results(frequencies, group_delay_ns, ptp, args.window_mhz, args.limit_ns, args.output)


if __name__ == "__main__":
    main()
