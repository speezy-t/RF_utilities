#!/usr/bin/env python3
"""
smooth_s2p_ri.py  –  Smooth noisy S2P files in Real/Imaginary format.

Workflow:
  1. Read the RI-format .s2p file.
  2. Convert every S-parameter to dB magnitude + phase (degrees).
  3. Smooth magnitude directly; smooth phase with unwrap → filter → re-wrap
     (identical method to smooth_s2p_v2.py).
  4. Write the result as a DB-format .s2p file.

The output is always DB format regardless of the input format, which is more
convenient for most VNA post-processing tools.

Usage:
    python smooth_s2p_ri.py input.s2p [output.s2p]
                            [--window 21] [--poly 3]
                            [--method savgol|gaussian]
                            [--sigma 100]
                            [--freq-range 27.5 31.0]
                            [--report]

The script errors if the file is not RI format (use smooth_s2p_v2.py for
MA or DB input files).

Dependencies:
    pip install numpy scipy matplotlib
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_s2p(filepath: str):
    """Read a Touchstone S2P file.

    Returns
    -------
    comments : list[str]
    options  : dict  (freq_unit, param, fmt, z0, raw_line)
    data     : np.ndarray shape (N, 9)
        Columns: freq, col1, col2, ... col8  (meaning depends on fmt)
    """
    comments = []
    options = {"freq_unit": "GHz", "param": "S", "fmt": "RI", "z0": 50.0, "raw_line": None}
    rows = []

    with open(filepath) as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            stripped = line.strip()
            if stripped.startswith("!"):
                comments.append(line)
                continue
            if stripped.startswith("#"):
                options["raw_line"] = line
                toks = stripped.upper().split()
                for i, tok in enumerate(toks):
                    if tok in ("HZ", "KHZ", "MHZ", "GHZ"):
                        options["freq_unit"] = tok.capitalize()
                    elif tok in ("S", "Y", "Z", "H", "G"):
                        options["param"] = tok
                    elif tok in ("MA", "DB", "RI"):
                        options["fmt"] = tok
                    elif tok == "R":
                        try:
                            options["z0"] = float(toks[i + 1])
                        except (IndexError, ValueError):
                            pass
                continue
            if not stripped:
                continue
            nums = [float(x) for x in stripped.split()]
            if nums:
                rows.append(nums)

    flat = [v for row in rows for v in row]
    if len(flat) % 9 != 0:
        raise ValueError(
            f"Expected a multiple of 9 values in the data section, got {len(flat)}. "
            "Is this a valid S2P file?"
        )
    data = np.array(flat, dtype=float).reshape(-1, 9)
    return comments, options, data


# ---------------------------------------------------------------------------
# Format conversion
# ---------------------------------------------------------------------------

def ri_to_db(data: np.ndarray) -> np.ndarray:
    """Convert an RI-format data array to DB/angle format.

    Input  columns: freq, Re(S11), Im(S11), Re(S21), Im(S21), ...
    Output columns: freq, dB(S11), ang(S11), dB(S21), ang(S21), ...
    """
    out = data.copy()
    for rc, ic in [(1, 2), (3, 4), (5, 6), (7, 8)]:
        re = data[:, rc]
        im = data[:, ic]
        mag_lin = np.sqrt(re ** 2 + im ** 2)
        mag_lin = np.where(mag_lin == 0.0, 1e-30, mag_lin)
        out[:, rc] = 20.0 * np.log10(mag_lin)          # dB magnitude
        out[:, ic] = np.rad2deg(np.arctan2(im, re))    # phase in degrees
    return out


# ---------------------------------------------------------------------------
# Smoothing  (identical logic to smooth_s2p_v2.py)
# ---------------------------------------------------------------------------

def _auto_window(n: int, window: int, poly: int):
    """Shrink window and poly order if the dataset is smaller than the window."""
    if window >= n:
        window = n - 1 if n % 2 == 0 else n
    if window % 2 == 0:
        window += 1
    if poly >= window:
        poly = window - 1
    return window, poly


def smooth_db_data(
    data: np.ndarray,
    window: int = 21,
    poly: int = 3,
    method: str = "savgol",
    sigma_mhz: float = 100.0,
    freq_range=None,
    freq_unit: str = "GHz",
) -> np.ndarray:
    """
    Smooth DB/angle S-parameter data with phase unwrapping.

    Odd-indexed columns (1, 3, 5, 7) are dB magnitudes – filtered directly.
    Even-indexed columns (2, 4, 6, 8) are phase angles – unwrapped to a
    continuous signal, filtered, then re-wrapped to [-180, 180].

    Parameters
    ----------
    data       : (N, 9) array in DB/angle format
    window     : Savitzky-Golay window length (odd, > poly)
    poly       : Savitzky-Golay polynomial order
    method     : 'savgol' or 'gaussian'
    sigma_mhz  : Gaussian kernel width in MHz; converted to points from the
                 file's actual step size so behaviour is independent of
                 sweep density.
    freq_range : (f_start_GHz, f_stop_GHz) or None for the full sweep
    freq_unit  : frequency unit from the option line ('MHz', 'GHz', etc.)
    """
    n = data.shape[0]
    smoothed = data.copy()

    # Step size → sigma in points
    step_native = np.median(np.diff(data[:, 0]))
    freq_to_mhz = {"hz": 1e-6, "khz": 1e-3, "mhz": 1.0, "ghz": 1e3}.get(
        freq_unit.lower(), 1.0
    )
    step_mhz  = step_native * freq_to_mhz
    sigma_pts = sigma_mhz / step_mhz
    print(f"  [info] Step size: {step_mhz:.4f} MHz  →  sigma = {sigma_mhz:.1f} MHz = {sigma_pts:.2f} points")

    # Index range to smooth (optional band restriction)
    if freq_range is not None:
        f_lo = freq_range[0] * 1e3 / freq_to_mhz   # GHz → native unit
        f_hi = freq_range[1] * 1e3 / freq_to_mhz
        mask = (data[:, 0] >= f_lo) & (data[:, 0] <= f_hi)
        idxs = np.where(mask)[0]
        if len(idxs) < 4:
            print(f"  [warn] Only {len(idxs)} points in freq_range – smoothing full sweep instead.")
            idxs = np.arange(n)
    else:
        idxs = np.arange(n)

    w, p = _auto_window(len(idxs), window, poly)
    if w != window:
        print(f"  [info] Window reduced to {w} (poly {p}) to fit {len(idxs)} points.")

    for col in range(1, 9):
        y_full   = smoothed[:, col].copy()
        y        = y_full[idxs]
        is_phase = (col % 2 == 0)   # even cols = phase angle

        if is_phase:
            # Unwrap → smooth → re-wrap to [-180, 180]
            y_unwrapped = np.unwrap(np.deg2rad(y))
            if method == "savgol":
                y_smooth = savgol_filter(y_unwrapped, window_length=w, polyorder=p)
            else:
                y_smooth = gaussian_filter1d(y_unwrapped, sigma=sigma_pts)
            y_smooth_deg = np.rad2deg(y_smooth)
            y_full[idxs] = (y_smooth_deg + 180.0) % 360.0 - 180.0
        else:
            # dB magnitude – filter directly
            if method == "savgol":
                y_full[idxs] = savgol_filter(y, window_length=w, polyorder=p)
            else:
                y_full[idxs] = gaussian_filter1d(y, sigma=sigma_pts)

        smoothed[:, col] = y_full

    return smoothed


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def write_s2p_db(filepath: str, comments: list, options: dict,
                 data: np.ndarray, source_file: str) -> None:
    """Write a DB/angle Touchstone S2P file."""
    with open(filepath, "w") as fh:
        for c in comments:
            fh.write(c + "\n")
        fh.write(f"! Smoothed by smooth_s2p_ri.py  –  source: {source_file}\n")
        fh.write(
            f"# {options['freq_unit']} {options['param']} DB R {options['z0']:.0f}\n"
        )
        for row in data:
            freq = row[0]
            freq_str = f"{freq:.6e}" if freq >= 1e9 else f"{freq:.6f}"
            pairs = " ".join(f"{row[i]:>14.6e} {row[i+1]:>14.6e}" for i in range(1, 9, 2))
            fh.write(f"{freq_str}  {pairs}\n")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate_report(db_orig: np.ndarray, db_smoothed: np.ndarray,
                    options: dict, output_png: str, freq_range=None) -> None:
    """Two-row × 4-col PNG: dB magnitude (row 0) and phase (row 1) per S-param."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("  [warn] matplotlib not found – skipping report.")
        return

    fu = options.get("freq_unit", "GHz").lower()
    to_ghz = {"hz": 1e-9, "khz": 1e-6, "mhz": 1e-3, "ghz": 1.0}.get(fu, 1.0)
    freq_ghz = db_orig[:, 0] * to_ghz

    if freq_range:
        mask = (freq_ghz >= freq_range[0]) & (freq_ghz <= freq_range[1])
    else:
        mask = np.ones(len(freq_ghz), dtype=bool)

    f         = freq_ghz[mask]
    params    = ["S11", "S21", "S12", "S22"]
    col_pairs = [(1, 2), (3, 4), (5, 6), (7, 8)]

    fig = plt.figure(figsize=(16, 9), facecolor="#0e1117")
    fig.suptitle(
        "S2P RI → DB Smoothing Report  –  smooth_s2p_ri.py",
        color="white", fontsize=13, y=0.995
    )
    gs = GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.33)

    def style(ax, title, ylabel):
        ax.set_facecolor("#1a1d27")
        ax.set_title(title, color="white", fontsize=9, pad=5)
        ax.set_xlabel("Freq (GHz)", color="#aaaacc", fontsize=7)
        ax.set_ylabel(ylabel, color="#aaaacc", fontsize=7)
        ax.tick_params(colors="#888899", labelsize=6)
        ax.spines[:].set_color("#333344")
        ax.grid(True, color="#2a2d3a", lw=0.5)
        ax.set_xlim(f[0], f[-1])

    for col_idx, (param, (mc, pc)) in enumerate(zip(params, col_pairs)):
        ax_mag = fig.add_subplot(gs[0, col_idx])
        style(ax_mag, f"{param}  Magnitude (dB)", "dB")
        ax_mag.plot(f, db_orig[mask, mc],     color="#ff6666", lw=0.5, alpha=0.7, label="RI-converted")
        ax_mag.plot(f, db_smoothed[mask, mc], color="#66aaff", lw=1.4,            label="Smoothed")
        if col_idx == 0:
            ax_mag.legend(fontsize=6, facecolor="#1a1d27", labelcolor="white", framealpha=0.8)

        ax_phs = fig.add_subplot(gs[1, col_idx])
        style(ax_phs, f"{param}  Phase (°)", "degrees")
        ax_phs.plot(f, db_orig[mask, pc],     color="#ff6666", lw=0.5, alpha=0.7)
        ax_phs.plot(f, db_smoothed[mask, pc], color="#66aaff", lw=1.4)

    plt.savefig(output_png, dpi=150, bbox_inches="tight", facecolor="#0e1117")
    plt.close()
    print(f"  Report saved : {output_png}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Read an RI-format S2P file, convert to dB/angle internally, "
            "smooth using phase-unwrap-aware filtering (same method as smooth_s2p_v2.py), "
            "and write a DB-format output file."
        )
    )
    parser.add_argument("input",  help="Input .s2p file (must be RI format)")
    parser.add_argument("output", nargs="?", default=None,
                        help="Output .s2p path (default: <input>_smoothed.s2p)")

    parser.add_argument("--window", type=int, default=21,
                        help="Savitzky-Golay window length, must be odd (default 21). "
                             "Larger = more smoothing.")
    parser.add_argument("--poly",   type=int, default=3,
                        help="Savitzky-Golay polynomial order (default 3). "
                             "Lower = smoother.")
    parser.add_argument("--method", choices=["savgol", "gaussian"], default="savgol",
                        help="Filter: 'savgol' (default) for dense measured data; "
                             "'gaussian' for sparse simulation sweeps (e.g. 50 MHz steps).")
    parser.add_argument("--sigma",  type=float, default=100.0,
                        help="Gaussian kernel width in MHz (default 100 MHz). "
                             "Auto-converted to points from the file's step size. "
                             "Only used with --method gaussian.")
    parser.add_argument("--freq-range", nargs=2, type=float,
                        metavar=("START_GHz", "STOP_GHz"),
                        help="Restrict smoothing to this band (GHz). "
                             "Points outside are written through unchanged.")
    parser.add_argument("--report", action="store_true",
                        help="Save a before/after PNG (dB mag + phase for each S-param).")
    parser.add_argument("--force",  action="store_true",
                        help="Skip the RI format check and process anyway.")

    args = parser.parse_args()

    input_path  = args.input
    output_path = args.output or str(
        Path(input_path).with_stem(Path(input_path).stem + "_smoothed")
    )

    window = args.window
    if window % 2 == 0:
        window += 1
        print(f"  [info] Window must be odd – adjusted to {window}.")

    print(f"Reading  : {input_path}")
    comments, options, ri_data = parse_s2p(input_path)
    print(f"  Points : {ri_data.shape[0]}")
    print(f"  Format : {options['fmt']}  ({options['freq_unit']})")

    if options["fmt"].upper() != "RI" and not args.force:
        print(
            f"\n  [error] This file is '{options['fmt']}' format, not RI.\n"
            "          Use smooth_s2p_v2.py for MA or DB input files.\n"
            "          Pass --force to skip this check."
        )
        raise SystemExit(1)

    # Convert RI → DB/angle before smoothing
    print("Converting: RI → dB/angle")
    db_data = ri_to_db(ri_data)

    print(f"Method   : {args.method}  |  window={window}, poly={args.poly}, sigma={args.sigma} MHz")
    if args.freq_range:
        print(f"Freq band: {args.freq_range[0]} – {args.freq_range[1]} GHz")

    db_smoothed = smooth_db_data(
        db_data,
        window=window,
        poly=args.poly,
        method=args.method,
        sigma_mhz=args.sigma,
        freq_range=args.freq_range,
        freq_unit=options["freq_unit"],
    )

    print(f"Writing  : {output_path}  (DB format)")
    write_s2p_db(output_path, comments, options, db_smoothed, Path(input_path).name)

    # Summary
    print("\nMax change per S-parameter (phase = circular distance):")
    labels = ["S11 Mag", "S11 Phase", "S21 Mag", "S21 Phase",
              "S12 Mag", "S12 Phase", "S22 Mag", "S22 Phase"]
    for i, label in enumerate(labels, start=1):
        diff     = db_smoothed[:, i] - db_data[:, i]
        is_phase = (i % 2 == 0)
        if is_phase:
            delta = np.max(np.abs(((diff + 180.0) % 360.0) - 180.0))
        else:
            delta = np.max(np.abs(diff))
        print(f"  {label}: {delta:.4f}")

    if args.report:
        report_path = str(Path(output_path).with_suffix(".png"))
        generate_report(db_data, db_smoothed, options, report_path,
                        freq_range=args.freq_range)

    print("\nDone.")


if __name__ == "__main__":
    main()
