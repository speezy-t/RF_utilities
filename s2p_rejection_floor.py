#!/usr/bin/env python3
"""
s2p_rejection_floor.py

Reads a .s2p S-parameter file and writes a new .s2p file where the magnitudes
of S21 and S12 are floored to a user-specified power level (in dBm / dB).
Any S21 or S12 magnitude in dB that is less than --max_rejection_dB will be
clamped to that value; all other data (S11, S22, phase, frequency) is preserved
exactly as in the source file.

Usage
-----
python s2p_rejection_floor.py input.s2p output.s2p --max_rejection_dB -60
"""

import argparse
import math
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def linear_to_db(linear: float) -> float:
    """Convert a linear (voltage) magnitude to dB.  Clamp near-zero values."""
    if linear <= 0.0:
        return -math.inf
    return 20.0 * math.log10(linear)


def db_to_linear(db: float) -> float:
    """Convert a dB magnitude to a linear (voltage) magnitude."""
    return 10.0 ** (db / 20.0)


# ---------------------------------------------------------------------------
# S2P parser
# ---------------------------------------------------------------------------

class S2PFile:
    """Minimal .s2p reader / writer that preserves comments and formatting."""

    def __init__(self):
        self.comments_before_option: list[str] = []   # '!' lines before '#'
        self.option_line: str = ""                     # the '#' line, raw
        self.freq_unit: str = "GHz"
        self.fmt: str = "MA"                           # MA | DB | RI
        self.r_ohms: float = 50.0
        self.comments_after_option: list[str] = []    # '!' lines after '#'
        # Each row: [freq_raw_str, (s11_a, s11_b), (s21_a, s21_b),
        #            (s12_a, s12_b), (s22_a, s22_b)]
        # a/b meaning depends on fmt: MA=(mag_lin, deg), DB=(mag_db, deg), RI=(re, im)
        self.rows: list = []

    # ------------------------------------------------------------------
    def parse(self, path: Path) -> None:
        option_seen = False
        pending_values: list[float] = []
        pending_freq_str: str = ""

        with path.open("r") as fh:
            for raw_line in fh:
                line = raw_line.rstrip("\n")
                stripped = line.strip()

                # ---- comment line ----------------------------------------
                if stripped.startswith("!"):
                    if option_seen:
                        self.comments_after_option.append(line)
                    else:
                        self.comments_before_option.append(line)
                    continue

                # ---- option line -----------------------------------------
                if stripped.startswith("#"):
                    self.option_line = line
                    self._parse_option(stripped)
                    option_seen = True
                    continue

                # ---- blank line ------------------------------------------
                if not stripped:
                    continue

                # ---- data line -------------------------------------------
                # Touchtone .s2p: one row = 9 numbers (may be split across lines)
                tokens = stripped.split()
                for tok in tokens:
                    if not pending_values:
                        # first token of a new row is the frequency
                        pending_freq_str = tok
                        pending_values.append(float(tok))
                    else:
                        pending_values.append(float(tok))

                    if len(pending_values) == 9:
                        freq_str = pending_freq_str
                        v = pending_values
                        self.rows.append(
                            [
                                freq_str,
                                (v[1], v[2]),   # S11
                                (v[3], v[4]),   # S21
                                (v[5], v[6]),   # S12
                                (v[7], v[8]),   # S22
                            ]
                        )
                        pending_values = []
                        pending_freq_str = ""

        if pending_values:
            raise ValueError(
                f"Incomplete data row at end of file "
                f"(got {len(pending_values)} values, expected 9)."
            )

    # ------------------------------------------------------------------
    def _parse_option(self, line: str) -> None:
        """Extract frequency unit, format, and R value from the option line."""
        parts = line.upper().split()
        # Default Touchstone option-line: # GHz S MA R 50
        freq_map = {"HZ": "Hz", "KHZ": "KHz", "MHZ": "MHz", "GHZ": "GHz"}
        for i, tok in enumerate(parts):
            if tok in freq_map:
                self.freq_unit = freq_map[tok]
            elif tok in ("MA", "DB", "RI"):
                self.fmt = tok
            elif tok == "R" and i + 1 < len(parts):
                try:
                    self.r_ohms = float(parts[i + 1])
                except ValueError:
                    pass

    # ------------------------------------------------------------------
    def get_mag_db(self, pair: tuple[float, float]) -> float:
        """Return the dB magnitude of an (a, b) pair for the current format."""
        a, b = pair
        if self.fmt == "DB":
            return a
        if self.fmt == "MA":
            return linear_to_db(a)
        # RI
        mag_lin = math.hypot(a, b)
        return linear_to_db(mag_lin)

    def set_mag_db(self, pair: tuple[float, float], new_db: float) -> tuple[float, float]:
        """Return a new (a, b) pair with magnitude replaced by new_db, phase preserved."""
        a, b = pair
        if self.fmt == "DB":
            return (new_db, b)          # b is phase angle — unchanged
        if self.fmt == "MA":
            return (db_to_linear(new_db), b)   # b is phase angle — unchanged
        # RI: preserve phase angle, update magnitude
        old_mag = math.hypot(a, b)
        if old_mag == 0.0:
            return (db_to_linear(new_db), 0.0)
        scale = db_to_linear(new_db) / old_mag
        return (a * scale, b * scale)

    # ------------------------------------------------------------------
    def write(self, path: Path) -> None:
        """Write the (possibly modified) data to a new .s2p file."""
        with path.open("w") as fh:
            for c in self.comments_before_option:
                fh.write(c + "\n")

            fh.write(self.option_line + "\n")

            for c in self.comments_after_option:
                fh.write(c + "\n")

            for row in self.rows:
                freq_str, s11, s21, s12, s22 = row
                # Use the original freq string to avoid float-formatting changes
                fh.write(
                    f"{freq_str}"
                    f"  {s11[0]:.10g}  {s11[1]:.10g}"
                    f"  {s21[0]:.10g}  {s21[1]:.10g}"
                    f"  {s12[0]:.10g}  {s12[1]:.10g}"
                    f"  {s22[0]:.10g}  {s22[1]:.10g}"
                    "\n"
                )


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def apply_rejection_floor(
    s2p: S2PFile,
    max_rejection_db: float,
) -> int:
    """
    Clamp S21 and S12 magnitudes so they are no less than *max_rejection_db*.

    Parameters
    ----------
    s2p : S2PFile
        Parsed S2P object (modified in place).
    max_rejection_db : float
        Floor value in dB.  Magnitudes below this value are raised to it.

    Returns
    -------
    int
        Number of data points (per parameter) that were clamped.
    """
    clamped = 0
    for row in s2p.rows:
        freq_str, s11, s21, s12, s22 = row

        mag_s21 = s2p.get_mag_db(s21)
        if mag_s21 < max_rejection_db:
            s21 = s2p.set_mag_db(s21, max_rejection_db)
            clamped += 1

        mag_s12 = s2p.get_mag_db(s12)
        if mag_s12 < max_rejection_db:
            s12 = s2p.set_mag_db(s12, max_rejection_db)
            clamped += 1

        row[2] = s21
        row[3] = s12

    return clamped


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a rejection floor to S21 and S12 in a .s2p file.  "
            "Any S21/S12 magnitude in dB that falls below --max_rejection_dB "
            "is clamped to that value.  All other data is preserved."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the input .s2p file.",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Path for the output .s2p file (will be created or overwritten).",
    )
    parser.add_argument(
        "--max_rejection_dB",
        type=float,
        required=True,
        metavar="dB",
        help=(
            "Rejection floor in dB (e.g. -60).  S21 and S12 magnitudes that "
            "are less than this value will be set to this value in the output."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path: Path = args.input
    output_path: Path = args.output
    floor_db: float = args.max_rejection_dB

    # -- validate input ------------------------------------------------------
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1
    if input_path.suffix.lower() != ".s2p":
        print(
            f"Warning: input file does not have a .s2p extension: {input_path}",
            file=sys.stderr,
        )

    # -- parse ---------------------------------------------------------------
    print(f"Reading  : {input_path}")
    s2p = S2PFile()
    try:
        s2p.parse(input_path)
    except Exception as exc:
        print(f"Error parsing {input_path}: {exc}", file=sys.stderr)
        return 1

    print(
        f"Format   : {s2p.fmt}  |  "
        f"Freq unit: {s2p.freq_unit}  |  "
        f"R: {s2p.r_ohms} Ω  |  "
        f"Data rows: {len(s2p.rows)}"
    )

    # -- apply floor ---------------------------------------------------------
    print(f"Floor    : {floor_db} dB  (applied to S21 and S12)")
    clamped = apply_rejection_floor(s2p, floor_db)
    print(f"Clamped  : {clamped} value(s) across S21 + S12")

    # -- write ---------------------------------------------------------------
    print(f"Writing  : {output_path}")
    try:
        s2p.write(output_path)
    except Exception as exc:
        print(f"Error writing {output_path}: {exc}", file=sys.stderr)
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
