#!/usr/bin/env python3
"""
smooth_s2p_ri.py  –  Smooth noisy S2P (Touchstone) files in Real/Imaginary format.

Unlike magnitude/phase files, RI data needs no phase-unwrapping – real and imaginary
parts are continuous signals that can be filtered directly.  The smoothed Re and Im
components are written back out in RI format; the computed magnitude is shown in the
optional report but is NOT written to the file (the RI values are the source of truth).

Usage:
    python smooth_s2p_ri.py input.s2p [output.s2p]
                            [--window 21] [--poly 3]
                            [--method savgol|gaussian]
                            [--sigma 100]
                            [--freq-range 27.5 31.0]
                            [--report]

The script will error if the file's option line declares a format other than RI.
Use smooth_s2p_v2.py for MA or DB format files.

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
        Columns: freq, Re(S11), Im(S11), Re(S21), Im(S21),
                       Re(S12), Im(S12), Re(S22), Im(S22)
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
# Smoothing
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


def smooth_data(
    data: np.ndarray,
    window: int = 21,
    poly: int = 3,
    method: str = "savgol",
    sigma_mhz: float = 100.0,
    freq_range=None,
    freq_unit: str = "GHz",
) -> np.ndarray:
    """
    Smooth RI-format S-parameter data.

    Real and imaginary parts are both continuous, so they are filtered directly
    with no unwrapping step required.

    Parameters
    ----------
    data       : (N, 9) array – freq + [Re, Im] × 4 S-params
    window     : Savitzky-Golay window length (odd, must be > poly)
    poly       : Savitzky-Golay polynomial order
    method     : 'savgol' or 'gaussian'
    sigma_mhz  : Gaussian kernel width in MHz. Converted to points internally
                 using the file's actual step size, so the same value gives
                 consistent physical smoothing regardless of sweep density.
    freq_range : (f_start_GHz, f_stop_GHz) or None for full sweep
    freq_unit  : frequency unit string from the option line ('MHz', 'GHz', etc.)
    """
    n = data.shape[0]
    smoothed = data.copy()

    # Derive step size in MHz directly from the frequency column
    freq_native = data[:, 0]
    step_native = np.median(np.diff(freq_native))   # median is robust to irregular spacing
    freq_to_mhz = {"hz": 1e-6, "khz": 1e-3, "mhz": 1.0, "ghz": 1e3}.get(
        freq_unit.lower(), 1.0
    )
    step_mhz = step_native * freq_to_mhz
    sigma_pts = sigma_mhz / step_mhz
    print(f"  [info] Step size: {step_mhz:.4f} MHz  →  sigma = {sigma_mhz:.1f} MHz = {sigma_pts:.2f} points")

    # Determine which indices to smooth (optional band restriction)
    if freq_range is not None:
        mhz_per_native = freq_to_mhz
        f_lo = freq_range[0] * 1e3 / mhz_per_native   # GHz → native unit
        f_hi = freq_range[1] * 1e3 / mhz_per_native
        mask = (freq_native >= f_lo) & (freq_native <= f_hi)
        idxs = np.where(mask)[0]
        if len(idxs) < 4:
            print(f"  [warn] Only {len(idxs)} points in freq_range – smoothing full sweep instead.")
            idxs = np.arange(n)
    else:
        idxs = np.arange(n)

    w, p = _auto_window(len(idxs), window, poly)
    if w != window:
        print(f"  [info] Window reduced to {w} (poly {p}) to fit {len(idxs)} points.")

    # Smooth every data column identically – Re and Im are both plain signals
    for col in range(1, 9):
        y_full = smoothed[:, col].copy()
        y = y_full[idxs]
        if method == "savgol":
            y_full[idxs] = savgol_filter(y, window_length=w, polyorder=p)
        else:
            y_full[idxs] = gaussian_filter1d(y, sigma=sigma_pts)
        smoothed[:, col] = y_full

    return smoothed


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def write_s2p(filepath: str, comments: list, options: dict,
              data: np.ndarray, source_file: str) -> None:
    """Write smoothed data back as a Touchstone S2P file, preserving the RI format."""
    with open(filepath, "w") as fh:
        for c in comments:
            fh.write(c + "\n")
        fh.write(f"! Smoothed by smooth_s2p_ri.py  –  source: {source_file}\n")
        if options.get("raw_line"):
            fh.write(options["raw_line"] + "\n")
        else:
            fh.write(
                f"# {options['freq_unit']} {options['param']} RI R {options['z0']:.0f}\n"
            )
        for row in data:
            freq = row[0]
            freq_str = f"{freq:.6e}" if freq >= 1e9 else f"{freq:.6f}"
            pairs = " ".join(f"{row[i]:>16.8e} {row[i+1]:>16.8e}" for i in range(1, 9, 2))
            fh.write(f"{freq_str}  {pairs}\n")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate_report(orig: np.ndarray, smoothed: np.ndarray,
                    options: dict, output_png: str, freq_range=None) -> None:
    """
    Save a 3-row × 4-col PNG:
      Row 0 – Real part (Re)
      Row 1 – Imaginary part (Im)
      Row 2 – Magnitude in dB computed from smoothed Re/Im, overlaid on original magnitude
    """
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
    freq_ghz = orig[:, 0] * to_ghz

    if freq_range:
        mask = (freq_ghz >= freq_range[0]) & (freq_ghz <= freq_range[1])
    else:
        mask = np.ones(len(freq_ghz), dtype=bool)

    f = freq_ghz[mask]
    params = ["S11", "S21", "S12", "S22"]
    # Column pairs: (re_col, im_col)
    col_pairs = [(1, 2), (3, 4), (5, 6), (7, 8)]

    fig = plt.figure(figsize=(16, 12), facecolor="#0e1117")
    fig.suptitle("S2P RI Smoothing Report  –  smooth_s2p_ri.py",
                 color="white", fontsize=13, y=0.995)
    gs = GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.33)

    def style(ax, title, ylabel):
        ax.set_facecolor("#1a1d27")
        ax.set_title(title, color="white", fontsize=8.5, pad=5)
        ax.set_xlabel("Freq (GHz)", color="#aaaacc", fontsize=7)
        ax.set_ylabel(ylabel, color="#aaaacc", fontsize=7)
        ax.tick_params(colors="#888899", labelsize=6)
        ax.spines[:].set_color("#333344")
        ax.grid(True, color="#2a2d3a", lw=0.5)
        ax.set_xlim(f[0], f[-1])

    for col_idx, (param, (rc, ic)) in enumerate(zip(params, col_pairs)):
        # Row 0 – Real
        ax_re = fig.add_subplot(gs[0, col_idx])
        style(ax_re, f"{param}  Re", "Real")
        ax_re.plot(f, orig[mask, rc],     color="#ff6666", lw=0.5, alpha=0.7, label="Original")
        ax_re.plot(f, smoothed[mask, rc], color="#66aaff", lw=1.4,            label="Smoothed")
        if col_idx == 0:
            ax_re.legend(fontsize=6, facecolor="#1a1d27", labelcolor="white", framealpha=0.8)

        # Row 1 – Imaginary
        ax_im = fig.add_subplot(gs[1, col_idx])
        style(ax_im, f"{param}  Im", "Imaginary")
        ax_im.plot(f, orig[mask, ic],     color="#ff6666", lw=0.5, alpha=0.7)
        ax_im.plot(f, smoothed[mask, ic], color="#66aaff", lw=1.4)

        # Row 2 – Magnitude in dB (derived from Re/Im)
        ax_mag = fig.add_subplot(gs[2, col_idx])
        style(ax_mag, f"{param}  Magnitude (dB)", "dB")

        # Guard against log(0)
        eps = 1e-30
        mag_orig = 20.0 * np.log10(
            np.sqrt(orig[mask, rc] ** 2 + orig[mask, ic] ** 2) + eps
        )
        mag_smth = 20.0 * np.log10(
            np.sqrt(smoothed[mask, rc] ** 2 + smoothed[mask, ic] ** 2) + eps
        )
        ax_mag.plot(f, mag_orig, color="#ff6666", lw=0.5, alpha=0.7)
        ax_mag.plot(f, mag_smth, color="#66aaff", lw=1.4)

    plt.savefig(output_png, dpi=150, bbox_inches="tight", facecolor="#0e1117")
    plt.close()
    print(f"  Report saved : {output_png}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Smooth a noisy S2P file in Real/Imaginary (RI) format. "
            "Re and Im parts are filtered directly – no phase unwrapping needed. "
            "For MA or DB format files use smooth_s2p_v2.py instead."
        )
    )
    parser.add_argument("input",  help="Input .s2p file (must be RI format)")
    parser.add_argument("output", nargs="?", default=None,
                        help="Output .s2p path (default: <input>_smoothed.s2p)")

    parser.add_argument("--window", type=int, default=21,
                        help="Savitzky-Golay window length, must be odd (default 21). "
                             "Larger values = more smoothing.")
    parser.add_argument("--poly",   type=int, default=3,
                        help="Savitzky-Golay polynomial order (default 3). "
                             "Lower values = smoother curve.")
    parser.add_argument("--method", choices=["savgol", "gaussian"], default="savgol",
                        help="Filter type: 'savgol' (default) works well for dense measured data; "
                             "'gaussian' is better for sparse simulation sweeps (e.g. 50 MHz steps).")
    parser.add_argument("--sigma",  type=float, default=100.0,
                        help="Gaussian kernel width in MHz (default 100 MHz). "
                             "Automatically converted to points from the file's actual step size, "
                             "so the same value gives consistent smoothing regardless of sweep density. "
                             "Only used with --method gaussian.")
    parser.add_argument("--freq-range", nargs=2, type=float,
                        metavar=("START_GHz", "STOP_GHz"),
                        help="Restrict smoothing to this band (GHz). Points outside the band are "
                             "written through unchanged.")
    parser.add_argument("--report", action="store_true",
                        help="Save a before/after PNG showing Re, Im, and computed magnitude.")
    parser.add_argument("--force",  action="store_true",
                        help="Skip the RI format check and process the file anyway.")

    args = parser.parse_args()

    input_path  = args.input
    output_path = args.output or str(
        Path(input_path).with_stem(Path(input_path).stem + "_smoothed")
    )

    # Ensure window is odd
    window = args.window
    if window % 2 == 0:
        window += 1
        print(f"  [info] Window must be odd – adjusted to {window}.")

    print(f"Reading  : {input_path}")
    comments, options, data = parse_s2p(input_path)
    print(f"  Points : {data.shape[0]}")
    print(f"  Format : {options['fmt']}  ({options['freq_unit']})")

    # Guard: refuse non-RI files unless --force is set
    if options["fmt"].upper() != "RI" and not args.force:
        print(
            f"\n  [error] This file uses '{options['fmt']}' format, not RI.\n"
            "          Use smooth_s2p_v2.py for MA or DB files.\n"
            "          Pass --force to override this check and smooth anyway\n"
            "          (results will be incorrect for phase columns in non-RI files)."
        )
        raise SystemExit(1)

    print(f"Method   : {args.method}  |  window={window}, poly={args.poly}, sigma={args.sigma} MHz")
    if args.freq_range:
        print(f"Freq band: {args.freq_range[0]} – {args.freq_range[1]} GHz")

    smoothed = smooth_data(
        data,
        window=window,
        poly=args.poly,
        method=args.method,
        sigma_mhz=args.sigma,
        freq_range=args.freq_range,
        freq_unit=options["freq_unit"],
    )

    print(f"Writing  : {output_path}")
    write_s2p(output_path, comments, options, smoothed, Path(input_path).name)

    # Summary statistics
    print("\nMax absolute change per column:")
    labels = ["S11 Re", "S11 Im", "S21 Re", "S21 Im",
              "S12 Re", "S12 Im", "S22 Re", "S22 Im"]
    for i, label in enumerate(labels, start=1):
        delta = np.max(np.abs(smoothed[:, i] - data[:, i]))
        print(f"  {label}: {delta:.6f}")

    # Also report magnitude change (computed from Re/Im)
    print("\nMax magnitude change (dB, computed from Re/Im):")
    eps = 1e-30
    param_names = ["S11", "S21", "S12", "S22"]
    for k, (rc, ic) in enumerate([(1,2),(3,4),(5,6),(7,8)]):
        mag_o = 20.0 * np.log10(np.sqrt(data[:,rc]**2     + data[:,ic]**2)     + eps)
        mag_s = 20.0 * np.log10(np.sqrt(smoothed[:,rc]**2 + smoothed[:,ic]**2) + eps)
        print(f"  {param_names[k]}: {np.max(np.abs(mag_s - mag_o)):.4f} dB")

    if args.report:
        report_path = str(Path(output_path).with_suffix(".png"))
        generate_report(data, smoothed, options, report_path, freq_range=args.freq_range)

    print("\nDone.")


if __name__ == "__main__":
    main()
