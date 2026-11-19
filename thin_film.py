"""
Thin-film optics used by the Nd and Nk window.

One film on a substrate, in air, with s-polarized light, using the optical
admittance (characteristic matrix) method. These are the same formulas as
Optical_Nd_V7S.py and Optical_NK_V7S.py, written to work on NumPy arrays.
This file contains no Qt / user-interface code, so it can be tested on its
own against those scripts.

Conventions (the same as the scripts):
  * complex refractive index N = n - i*k
  * wavelengths and thicknesses in nm, angles in degrees
  * PI = 3.14 in the phase thickness (see below)
"""

from __future__ import annotations

import math

import numpy as np

# The fitting scripts use 3.14 for pi in the phase thickness. It is kept here
# so the app reproduces their results; change it to math.pi if that is decided.
PI: float = 3.14

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
