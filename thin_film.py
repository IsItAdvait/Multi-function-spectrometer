"""
Thin-film optics used by the Nd and Nk window.

One film on a substrate, in air, with s-polarized light, using the optical
admittance (characteristic matrix) method. These are the same formulas as
Optical_Nd_V7S.py and Optical_NK_V7S.py, written to work on NumPy arrays.
This file contains no Qt / user-interface code, so it can be tested on its
own against those scripts.

Conventions:
  * complex refractive index N = n - i*k
  * wavelengths and thicknesses in nm, angles in degrees
  * PI = true pi everywhere (see below)
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

# Pi for the phase thickness (fits) and the angle conversion (Si reflectance).
# Decided Nov 2026: true pi everywhere, so everything is consistent. The
# friend's scripts and the lab's Excel sheet use 3.14; pass pi=3.14 to a
# function here to reproduce their numbers exactly.
PI: float = math.pi

# Refractive index of the ambient medium (air).
AMBIENT_INDEX: float = 1.0


def reflectance_s(
    n1: np.ndarray,
    k1: np.ndarray,
    d1: np.ndarray,
    n2: np.ndarray,
    k2: np.ndarray,
    wavelength_nm: np.ndarray,
    angle_deg: float,
    n0: float = AMBIENT_INDEX,
    pi: float = PI,
) -> np.ndarray:
    """Reflectance of air / film (n1, k1, thickness d1) / substrate (n2, k2).

    s-polarized light at ``angle_deg``. Every input except the angle may be a
    NumPy array; they are combined element-wise (NumPy broadcasting).
    Step by step, as in the scripts:
      Snell's law      theta = asin(N0 * sin(theta0) / N)
      phase thickness  delta = 2 * pi * N1 * d1 * cos(theta1) / wavelength
      admittances      eta = N * cos(theta)                    (s-wave)
      film + substrate Y = (eta2 cos d + i eta1 sin d) / (cos d + i (eta2/eta1) sin d)
      reflection       r = (eta0 - Y) / (eta0 + Y),  R = r * conj(r)
    """
    with np.errstate(all="ignore"):
        z0 = complex(n0, 0.0)
        z1 = np.asarray(n1, dtype=np.float64) - 1j * np.asarray(k1, dtype=np.float64)
        z2 = np.asarray(n2, dtype=np.float64) - 1j * np.asarray(k2, dtype=np.float64)
        theta0 = math.radians(angle_deg)
        sin_theta0 = math.sin(theta0)
        theta1 = np.arcsin(z0 / z1 * sin_theta0)
        theta2 = np.arcsin(z0 / z2 * sin_theta0)
        delta = (
            2.0
            * pi
            * z1
            * np.asarray(d1, dtype=np.float64)
            * np.cos(theta1)
            / np.asarray(wavelength_nm, dtype=np.float64)
        )
        eta0 = z0 * math.cos(theta0)
        eta1 = z1 * np.cos(theta1)
        eta2 = z2 * np.cos(theta2)
        admittance = (eta2 * np.cos(delta) + 1j * eta1 * np.sin(delta)) / (
            np.cos(delta) + 1j * eta2 / eta1 * np.sin(delta)
        )
        r = (eta0 - admittance) / (eta0 + admittance)
        return (r * np.conj(r)).real


def bare_substrate_reflectance_s(
    n2: np.ndarray, k2: np.ndarray, angle_deg: float, n0: float = AMBIENT_INDEX
) -> np.ndarray:
    """Reflectance of an uncoated substrate (e.g. the bare Si wafer), s-polarized.

    This is ``reflectance_s`` with no film (d1 = 0): the admittance Y then
    equals the substrate admittance eta2, so r = (eta0 - eta2) / (eta0 + eta2).
    At 0 degrees the s and p polarizations are identical. NaN inputs give NaN.
    """
    with np.errstate(all="ignore"):
        z0 = complex(n0, 0.0)
        z2 = np.asarray(n2, dtype=np.float64) - 1j * np.asarray(k2, dtype=np.float64)
        theta0 = math.radians(angle_deg)
        theta2 = np.arcsin(z0 / z2 * math.sin(theta0))
        eta0 = z0 * math.cos(theta0)
        eta2 = z2 * np.cos(theta2)
        r = (eta0 - eta2) / (eta0 + eta2)
        return (r * np.conj(r)).real


# ---------------------------------------------------------------------------
# Bare Si wafer reflectance, exactly as the lab's Excel sheet calculates it
# (Si_Std_Fit_Full_Santhosh.xlsx, columns K to AL with film thickness 0).
# ---------------------------------------------------------------------------


def _excel_round(values: np.ndarray, digits: int) -> np.ndarray:
    """Excel's ROUND: halves go away from zero (Python and NumPy round them to even).

    Excel works to about 15 significant digits, so a value stored as
    0.83544999999999 is treated as 0.83545 and rounds up; snapping the scaled
    value to 9 decimals first does the same here.
    """
    values = np.asarray(values, dtype=np.float64)
    scaled = np.round(np.abs(values) * 10.0**digits, 9)
    return np.copysign(np.floor(scaled + 0.5), values) / 10.0**digits


def _excel_round_complex(values: np.ndarray, digits: int) -> np.ndarray:
    """COMPLEX(ROUND(IMREAL(z), d), ROUND(IMAGINARY(z), d))."""
    values = np.asarray(values, dtype=np.complex128)
    return _excel_round(values.real, digits) + 1j * _excel_round(values.imag, digits)


def si_wafer_reflectance_excel(
    n2: np.ndarray,
    k2: np.ndarray,
    angle_deg: float,
    n0: float = AMBIENT_INDEX,
    pi: float = PI,
) -> np.ndarray:
    """Reflectance of the bare Si wafer, step by step as the Excel sheet does it.

    Same physics as ``bare_substrate_reflectance_s`` (s-polarized light, no
    film), with every step rounded to 4 decimals with Excel's ROUND, as the
    sheet does. The angle is converted with ``pi`` (true pi by default; the
    sheet itself uses 3.14).
    The steps (sheet column in brackets):
      N2 = ROUND(n2, 4) - i ROUND(k2, 4)                               [AA]
      sin(theta2) = n0 sin(angle / 180 * pi) / N2, rounded             [N, AB]
      cos(theta2) = sqrt(1 - sin(theta2)^2), rounded                   [AC]
      eta2 = N2 cos(theta2), rounded  (= the assembly admittance Y)    [AD, AG]
      eta0 = n0 cos(angle * pi / 180)                                  [O]
      r = (eta0 - eta2) / (eta0 + eta2), rounded                       [AH]
      R = ROUND(r * conj(r), 4)                                        [AI]
    Checked against the sheet: with pi=3.14, all 360 R0 and R60 values match
    exactly. NaN inputs give NaN.
    """
    with np.errstate(all="ignore"):
        sin_theta0 = n0 * math.sin(angle_deg / 180.0 * pi)
        eta0 = n0 * math.cos(angle_deg * pi / 180.0)
        z2 = _excel_round(n2, 4) + 1j * _excel_round(-np.asarray(k2, dtype=np.float64), 4)
        sin_theta2 = _excel_round_complex(sin_theta0 / z2, 4)
        cos_theta2 = _excel_round_complex(np.sqrt(1.0 - sin_theta2 * sin_theta2), 4)
        eta2 = _excel_round_complex(z2 * cos_theta2, 4)
        r = _excel_round_complex((eta0 - eta2) / (eta0 + eta2), 4)
        return _excel_round((r * np.conj(r)).real, 4)


# ---------------------------------------------------------------------------
# Nd and Nk fits (Optical_Nd_V7S.py and Optical_NK_V7S.py)
# ---------------------------------------------------------------------------


def _zoom_search(
    evaluate,
    a_start: float,
    a_range: float,
    b_start: float,
    b_range: float,
    steps: int,
) -> tuple[dict[str, object], dict[str, Optional[str]]]:
    """The scripts' 3-round grid search over two parameters a and b.

    ``evaluate(a_values, b_values)`` returns (error, r0, r60) on the grid
    a_values[:, None] x b_values[None, :].
      Round 1: a = a_start + da, + 2 da, ... (steps - 1 values, da = a_range / steps,
               built by repeated addition as the scripts do), b likewise.
      Rounds 2 and 3: a new (steps - 1) x (steps - 1) grid spanning one previous
               step either side of the best point so far.
    As in the scripts, the error starts at 1, a pair only becomes the best if its
    error is strictly smaller than the best so far, and pairs are tried a first,
    then b, so the first of equal pairs wins.
    Returns the best point and ``edges``: whether a / b ended round 1 on the
    first ("low") or last ("high") value of its range.
    """
    if steps < 3:
        raise ValueError("steps must be at least 3")
    count = steps - 1  # the scripts' range(1, steps)
    best: dict[str, object] = {"error": 1.0, "a": None, "b": None, "r0": None, "r60": None, "round": 0}

    def search(a_values: np.ndarray, b_values: np.ndarray, round_number: int) -> tuple[int, int]:
        error, r0, r60 = evaluate(a_values, b_values)
        usable = np.where(np.isfinite(error), error, np.inf)
        index = int(np.argmin(usable))  # first smallest, in the scripts' order
        i, j = divmod(index, b_values.size)
        if usable[i, j] < best["error"]:
            best.update(
                error=float(error[i, j]), a=float(a_values[i]), b=float(b_values[j]),
                r0=float(r0[i, j]), r60=float(r60[i, j]), round=round_number,
            )
        return i, j

    # Round 1 (a = a + da and b = b + db, repeated, as in the scripts).
    da = a_range / steps
    db = b_range / steps
    a_values = np.add.accumulate(np.r_[float(a_start), np.full(count, da)])[1:]
    b_values = np.add.accumulate(np.r_[float(b_start), np.full(count, db)])[1:]
    i, j = search(a_values, b_values, 1)
    if best["a"] is None:
        raise ValueError("No pair gave an error below 1; check the inputs.")
    edges = {
        "a": "low" if i == 0 else "high" if i == count - 1 else None,
        "b": "low" if j == 0 else "high" if j == count - 1 else None,
    }
    # Rounds 2 and 3: zoom in around the best point.
    steps_index = np.arange(1, steps)
    for round_number in (2, 3):
        a_low, b_low = best["a"] - 1 * da, best["b"] - 1 * db
        da, db = (2 * da) / steps, (2 * db) / steps
        search(a_low + steps_index * da, b_low + steps_index * db, round_number)
    return best, edges


def _two_angle_error(n1, k1, d1, n2, k2, wavelength_nm, r0_measured, r60_measured, pi):
    """(error, R0, R60) with error = (R60 - R60m)^2 + (R0 - R0m)^2, as in the scripts."""
    r60 = reflectance_s(n1, k1, d1, n2, k2, wavelength_nm, 60.0, pi=pi)
    r0 = reflectance_s(n1, k1, d1, n2, k2, wavelength_nm, 0.0, pi=pi)
    error = (r60 - r60_measured) * (r60 - r60_measured) + (r0 - r0_measured) * (
        r0 - r0_measured
    )
    return error, r0, r60


def fit_nd(
    wavelength_nm: float,
    r0_measured: float,
    r60_measured: float,
    n2: float,
    k2: float,
    n1_start: float,
    n1_range: float,
    d1_start: float,
    d1_range: float,
    k1: float = 0.0,
    steps: int = 100,
    pi: float = PI,
) -> dict[str, object]:
    """Nd fit (Optical_Nd_V7S.py): n1 and d1 at one wavelength, k1 fixed.

    n1 is searched from n1_start to n1_start + n1_range and d1 from d1_start to
    d1_start + d1_range with ``_zoom_search``. Returns n1, d1, error, r0_fit,
    r60_fit, the inputs, and ``edges`` ({"n1": ..., "d1": ...}).
    """
    best, edges = _zoom_search(
        lambda n1_values, d1_values: _two_angle_error(
            n1_values[:, None], k1, d1_values[None, :], n2, k2, wavelength_nm,
            r0_measured, r60_measured, pi,
        ),
        n1_start, n1_range, d1_start, d1_range, steps,
    )
    return {
        "n1": best["a"],
        "d1": best["b"],
        "error": best["error"],
        "r0_fit": best["r0"],
        "r60_fit": best["r60"],
        "r0_measured": float(r0_measured),
        "r60_measured": float(r60_measured),
        "wavelength": float(wavelength_nm),
        "n2": float(n2),
        "k2": float(k2),
        "k1": float(k1),
        "n1_start": float(n1_start),
        "n1_range": float(n1_range),
        "d1_start": float(d1_start),
        "d1_range": float(d1_range),
        "round": best["round"],
        "edges": {"n1": edges["a"], "d1": edges["b"]},
    }


def fit_nk(
    wavelength_nm: float,
    r0_measured: float,
    r60_measured: float,
    n2: float,
    k2: float,
    d1: float,
    n1_start: float,
    n1_range: float,
    k1_start: float,
    k1_range: float,
    steps: int = 100,
    pi: float = PI,
) -> dict[str, object]:
    """Nk fit (Optical_NK_V7S.py): n1 and k1 at one wavelength, d1 fixed.

    n1 is searched from n1_start to n1_start + n1_range and k1 from k1_start to
    k1_start + k1_range with ``_zoom_search``. Returns n1, k1, error, r0_fit,
    r60_fit and ``edges`` ({"n1": ..., "k1": ...}).
    """
    best, edges = _zoom_search(
        lambda n1_values, k1_values: _two_angle_error(
            n1_values[:, None], k1_values[None, :], d1, n2, k2, wavelength_nm,
            r0_measured, r60_measured, pi,
        ),
        n1_start, n1_range, k1_start, k1_range, steps,
    )
    return {
        "n1": best["a"],
        "k1": best["b"],
        "error": best["error"],
        "r0_fit": best["r0"],
        "r60_fit": best["r60"],
        "edges": {"n1": edges["a"], "k1": edges["b"]},
    }


def fit_nk_spectrum(
    wavelengths_nm: np.ndarray,
    r0_measured: np.ndarray,
    r60_measured: np.ndarray,
    n2: np.ndarray,
    k2: np.ndarray,
    d1: float,
    n1_start: float,
    n1_range: float,
    k1_start: float,
    k1_range: float,
    steps: int = 100,
    pi: float = PI,
    report=None,
) -> dict[str, np.ndarray]:
    """``fit_nk`` at every wavelength (each one on its own, same inputs).

    Wavelengths where R0, R60, n2 or k2 is not a number give NaN. ``report``,
    if given, is called with the fraction done (0..1) after each wavelength
    (the progress pop-up's report, which also stops the work on Cancel).
    Returns arrays: wavelength, n1, k1, error, r0_fit, r60_fit, and
    n1_edge / k1_edge ("low", "high" or "" per wavelength).
    """
    wavelengths = np.asarray(wavelengths_nm, dtype=np.float64)
    count = wavelengths.size
    out = {key: np.full(count, np.nan) for key in ("n1", "k1", "error", "r0_fit", "r60_fit")}
    n1_edge = np.full(count, "", dtype=object)
    k1_edge = np.full(count, "", dtype=object)
    inputs = [np.asarray(x, dtype=np.float64) for x in (r0_measured, r60_measured, n2, k2)]
    for index in range(count):
        r0m, r60m, n2_i, k2_i = (float(x[index]) for x in inputs)
        if all(np.isfinite(v) for v in (wavelengths[index], r0m, r60m, n2_i, k2_i)):
            try:
                result = fit_nk(
                    wavelengths[index], r0m, r60m, n2_i, k2_i, d1,
                    n1_start, n1_range, k1_start, k1_range, steps=steps, pi=pi,
                )
            except ValueError:
                result = None
            if result is not None:
                for key in out:
                    out[key][index] = result[key]
                n1_edge[index] = result["edges"]["n1"] or ""
                k1_edge[index] = result["edges"]["k1"] or ""
        if report is not None:
            report((index + 1) / count)
    out["wavelength"] = wavelengths.copy()
    out["n1_edge"] = n1_edge
    out["k1_edge"] = k1_edge
    return out
