"""
FLAME spectrometer desktop application.

Drives ``hardware.SpectrometerWorker`` on a dedicated ``QThread`` and provides
multi-geometry (R₀ / R₆₀ / T) Baseline / Standard / Sample capture with
CSV and plot-image export.
"""

from __future__ import annotations

import csv
import logging
import math
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter
from PyQt6.QtCore import (
    QAbstractTableModel,
    QLocale,
    QMetaObject,
    QModelIndex,
    Qt,
    QThread,
    QTimer,
    Q_ARG,
)
from PyQt6.QtGui import QAction, QCloseEvent, QDoubleValidator, QFont, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStatusBar,
    QStyle,
    QTableView,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from hardware import MockSpectrometer, SpectrometerWorker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configurable optical-mode labels
# Internal keys are ASCII-safe (CSV / dict); display uses Unicode subscripts.
# ---------------------------------------------------------------------------

# (internal_key, display_label)
OPTICAL_MODES: tuple[tuple[str, str], ...] = (
    ("R0", "R₀"),
    ("R60", "R₆₀"),
    ("T", "T"),
)

OPTICAL_MODE_KEYS: tuple[str, ...] = tuple(key for key, _ in OPTICAL_MODES)
OPTICAL_MODE_DISPLAY: dict[str, str] = {key: display for key, display in OPTICAL_MODES}


class CaptureMode(str, Enum):
    BASELINE = "Baseline"
    STANDARD = "Standard"
    SAMPLE = "Sample"


CAPTURE_LABELS: tuple[str, ...] = tuple(mode.value for mode in CaptureMode)

# Cyan / green / magenta dashed pens for frozen captures; live stays bright yellow.
_CAPTURE_PENS = {
    CaptureMode.BASELINE: pg.mkPen(
        color=(0, 220, 255, 200), width=1.8, style=Qt.PenStyle.DashLine
    ),
    CaptureMode.STANDARD: pg.mkPen(
        color=(80, 220, 100, 200), width=1.8, style=Qt.PenStyle.DashLine
    ),
    CaptureMode.SAMPLE: pg.mkPen(
        color=(255, 80, 220, 200), width=1.8, style=Qt.PenStyle.DashLine
    ),
}

_LIVE_PEN = pg.mkPen(color=(255, 220, 40), width=2)

# Ocean Optics FLAME-class integration-time bounds (microseconds).
# SeaBreeze applies the total via spectrometer.integration_time_micros().
_MIN_INTEGRATION_US: int = 1_000
_MAX_INTEGRATION_MS: int = 10_000


class MeasurementCsvError(ValueError):
    """Raised when an imported measurement CSV cannot be used."""


def _header_key(name: str) -> str:
    return name.strip().lower().replace("µ", "u").replace("μ", "u")


def _is_derived_column(header: str) -> bool:
    key = _header_key(header)
    return "interp" in key or "_avg" in key or "avg_" in key


def _parse_numeric_cell(token: str) -> Optional[float]:
    text = token.strip()
    if not text or text.upper() in {"N/A", "NA", "NAN"}:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if math.isnan(value):
        return None
    return value


def _row_looks_numeric(row: list[str]) -> bool:
    seen = 0
    for cell in row:
        text = cell.strip()
        if not text:
            continue
        if _parse_numeric_cell(text) is None:
            return False
        seen += 1
    return seen > 0


def _classify_measurement_column(header: str) -> Optional[tuple[Optional[str], str]]:
    """Map a CSV header to ``(geometry key or None, slot name)``.

    ``None`` geometry means an unprefixed Baseline/Standard/Sample column,
    which is applied to the optical mode that is active at import time.
    Mode-prefixed headers such as ``R0_Baseline (counts)`` target that mode.
    """
    if _is_derived_column(header):
        return None
    key = _header_key(header)
    if "wavelength" in key or "intensity" == key:
        return None
    for mode_key in OPTICAL_MODE_KEYS:
        mode_token = mode_key.lower()
        for slot in CAPTURE_LABELS:
            slot_token = slot.lower()
            if (
                f"{mode_token}_{slot_token}" in key
                or f"{mode_token} {slot_token}" in key
                or f"{mode_token}-{slot_token}" in key
            ):
                return mode_key, slot
    for slot in CAPTURE_LABELS:
        slot_token = slot.lower()
        if slot_token not in key:
            continue
        if any(
            f"{mode_key.lower()}_" in key
            or f"{mode_key.lower()} " in key
            or f"{mode_key.lower()}-" in key
            for mode_key in OPTICAL_MODE_KEYS
        ):
            continue
        return None, slot
    return None


def parse_unified_measurement_csv(
    path: Path,
) -> tuple[np.ndarray, dict[Optional[str], dict[str, np.ndarray]]]:
    """Load every raw measurement column from one CSV.

    Returns ``(wavelengths, columns)`` where ``columns`` maps an optical-mode
    key (or ``None`` for unprefixed columns) to slot-name intensity arrays.
    Average and interpolation columns are ignored.
    """
    try:
        raw_text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise MeasurementCsvError(f"Could not read file: {exc}") from exc

    rows = [
        row
        for row in csv.reader(raw_text.splitlines())
        if any(cell.strip() for cell in row)
    ]
    if rows and rows[0] and rows[0][0].strip().lower().startswith("sep="):
        rows = rows[1:]
    if not rows:
        raise MeasurementCsvError("The CSV file is empty.")

    if _row_looks_numeric(rows[0]):
        width = max(len(row) for row in rows)
        headers = [f"Column {i + 1}" for i in range(width)]
        data_rows = rows
    else:
        headers = [cell.strip() for cell in rows[0]]
        data_rows = rows[1:]

    if not data_rows:
        raise MeasurementCsvError("The CSV file has a header but no data rows.")

    wavelength_index: Optional[int] = None
    for index, header in enumerate(headers):
        key = _header_key(header)
        if "wavelength" in key and not _is_derived_column(header):
            wavelength_index = index
            break
    if wavelength_index is None:
        for index, header in enumerate(headers):
            if "wavelength" in _header_key(header):
                wavelength_index = index
                break
    if wavelength_index is None:
        wavelength_index = 0

    selected: dict[tuple[Optional[str], str], int] = {}
    for index, header in enumerate(headers):
        if index == wavelength_index:
            continue
        classified = _classify_measurement_column(header)
        if classified is None or classified in selected:
            continue
        selected[classified] = index

    if not selected:
        available = ", ".join(header for header in headers if header) or "(none)"
        raise MeasurementCsvError(
            "No Baseline, Standard, or Sample columns were found.\n"
            f"Columns in file: {available}"
        )

    wavelengths: list[float] = []
    series: dict[tuple[Optional[str], str], list[float]] = {
        key: [] for key in selected
    }
    for row in data_rows:
        if wavelength_index >= len(row):
            continue
        wavelength = _parse_numeric_cell(row[wavelength_index])
        if wavelength is None:
            continue
        parsed: dict[tuple[Optional[str], str], float] = {}
        missing = False
        for key, column in selected.items():
            if column >= len(row):
                missing = True
                break
            value = _parse_numeric_cell(row[column])
            if value is None:
                missing = True
                break
            parsed[key] = value
        if missing:
            continue
        wavelengths.append(wavelength)
        for key, value in parsed.items():
            series[key].append(value)

    if len(wavelengths) < 2:
        raise MeasurementCsvError(
            "The file does not contain at least two complete wavelength rows "
            "for the detected measurement columns."
        )

    grouped: dict[Optional[str], dict[str, np.ndarray]] = {}
    for (geometry, slot), values in series.items():
        grouped.setdefault(geometry, {})[slot] = np.asarray(values, dtype=np.float64)
    return np.asarray(wavelengths, dtype=np.float64), grouped


def _load_table_rows(path: Path) -> list[list[str]]:
    """Read a CSV or Excel sheet as strings, one list per row."""
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        try:
            import pandas as pd
        except ImportError as exc:
            raise MeasurementCsvError(
                "Reading Excel files requires pandas and openpyxl."
            ) from exc
        try:
            frame = pd.read_excel(path, header=None, dtype=object)
        except Exception as exc:
            raise MeasurementCsvError(f"Could not read Excel file: {exc}") from exc
        rows: list[list[str]] = []
        for record in frame.itertuples(index=False, name=None):
            converted: list[str] = []
            for cell in record:
                if cell is None or pd.isna(cell):
                    converted.append("")
                else:
                    converted.append(str(cell).strip())
            rows.append(converted)
        return rows

    try:
        raw_text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise MeasurementCsvError(f"Could not read file: {exc}") from exc
    return list(csv.reader(raw_text.splitlines()))


def _header_tokens(key: str) -> set[str]:
    return set(
        key.replace("_", " ")
        .replace("-", " ")
        .replace("(", " ")
        .replace(")", " ")
        .split()
    )


def _silicon_role_scores(key: str) -> dict[str, int]:
    """Score one header for wavelength, refractive index n, or extinction k."""
    tokens = _header_tokens(key)
    scores = {"wl": 0, "n": 0, "k": 0}
    if "wavelength" in key or "lambda" in key:
        scores["wl"] = 6
    elif "wl" in tokens:
        scores["wl"] = 5
    elif "nm" in tokens or "(nm)" in key:
        scores["wl"] = 4
    if "extinct" in key or "imag" in key:
        scores["k"] = 6
    elif "k" in tokens:
        scores["k"] = 5
    if "n" in tokens or "refractive" in key:
        scores["n"] = 6
    elif "real" in key and "imag" not in key:
        scores["n"] = 5
    elif "index" in key and "extinct" not in key and "imag" not in key:
        scores["n"] = 4
    return scores


def _silicon_column_indexes(headers: list[str]) -> tuple[int, int, int]:
    """Pick wavelength, n, and k columns, or fall back to the first three."""
    if len(headers) < 3:
        raise MeasurementCsvError(
            "The file needs wavelength, n, and k columns."
        )
    keys = [_header_key(header) for header in headers]
    role_scores = [_silicon_role_scores(key) for key in keys]
    chosen: dict[str, int] = {}
    used: set[int] = set()
    for role in ("wl", "n", "k"):
        best_index = None
        best_score = 0
        for index, scores in enumerate(role_scores):
            if index in used:
                continue
            score = scores[role]
            if score > best_score:
                best_score = score
                best_index = index
        if best_index is None:
            return 0, 1, 2
        chosen[role] = best_index
        used.add(best_index)
    return chosen["wl"], chosen["n"], chosen["k"]


def parse_silicon_reference(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load wavelength, refractive index n, and extinction coefficient k."""
    try:
        rows = _load_table_rows(path)
    except MeasurementCsvError:
        raise
    except OSError as exc:
        raise MeasurementCsvError(f"Could not read file: {exc}") from exc

    rows = [row for row in rows if any(str(cell).strip() for cell in row)]
    if rows and str(rows[0][0]).strip().lower().startswith("sep="):
        rows = rows[1:]
    if not rows:
        raise MeasurementCsvError("The file is empty.")

    first = [str(cell).strip() for cell in rows[0]]
    if _row_looks_numeric(first):
        if len(first) < 3:
            raise MeasurementCsvError(
                "The file needs wavelength, n, and k columns."
            )
        wl_idx, n_idx, k_idx = 0, 1, 2
        data_rows = rows
    else:
        headers = first
        wl_idx, n_idx, k_idx = _silicon_column_indexes(headers)
        data_rows = rows[1:]

    if len({wl_idx, n_idx, k_idx}) < 3:
        raise MeasurementCsvError(
            "Could not find separate wavelength, n, and k columns."
        )

    wavelengths: list[float] = []
    refractive_index: list[float] = []
    extinction: list[float] = []
    for row in data_rows:
        cells = [str(cell).strip() for cell in row]
        if max(wl_idx, n_idx, k_idx) >= len(cells):
            continue
        wavelength = _parse_numeric_cell(cells[wl_idx])
        n_value = _parse_numeric_cell(cells[n_idx])
        k_value = _parse_numeric_cell(cells[k_idx])
        if wavelength is None or n_value is None or k_value is None:
            continue
        wavelengths.append(wavelength)
        refractive_index.append(n_value)
        extinction.append(k_value)

    if len(wavelengths) < 2:
        raise MeasurementCsvError(
            "The file does not contain at least two numeric wavelength, n, and k rows."
        )
    return (
        np.asarray(wavelengths, dtype=np.float64),
        np.asarray(refractive_index, dtype=np.float64),
        np.asarray(extinction, dtype=np.float64),
    )


def _wavelength_axes_match(imported: np.ndarray, reference: np.ndarray) -> bool:
    """True when both axes have the same length and agree to ~0.05 nm."""
    if imported.shape != reference.shape:
        return False
    return bool(np.allclose(imported, reference, rtol=1e-5, atol=0.05))


def _empty_geometry_slot() -> dict[str, Optional[np.ndarray]]:
    return {label: None for label in CAPTURE_LABELS}


def _build_spectra_data() -> dict[str, dict[str, Optional[np.ndarray]]]:
    return {key: _empty_geometry_slot() for key in OPTICAL_MODE_KEYS}


def _display_mode(key: str) -> str:
    return OPTICAL_MODE_DISPLAY.get(key, key)


def _geometry_filename_token(geometry_key: str) -> str:
    """Filesystem-safe token for default image names (e.g. ``R0``)."""
    return (
        geometry_key.replace("°", "deg")
        .replace("/", "-")
        .replace(" ", "_")
        .replace("\\", "-")
    )


_RAW_HEADERS: tuple[str, ...] = (
    "Wavelength (nm)",
    "Baseline (counts)",
    "Standard (counts)",
    "Sample (counts)",
)

_SOURCE_TRACES: tuple[str, ...] = ("Baseline", "Standard", "Sample")

try:
    from scipy.interpolate import CubicSpline as _ScipyCubicSpline
except ImportError:  # pragma: no cover - optional dependency
    _ScipyCubicSpline = None  # type: ignore[misc, assignment]


def _moving_average_centered(trace: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling mean of window ``W``, length ``N``, with NaN edges.

    ``offset = (W - 1) // 2``. Valid means from ``np.convolve(..., mode='valid')``
    are placed starting at index ``offset``. Indices ``< offset`` and
    ``>= N - offset`` are ``np.nan`` (shown as ``N/A`` in the table).

    For ``W = 5`` (offset 2): row ``i`` holds ``mean(trace[i-2 : i+3])``.
    """
    x = np.asarray(trace, dtype=np.float64)
    n = x.size
    w = int(window)
    if w < 2 or w > n:
        raise ValueError(f"Window size {w} invalid for trace length {n}")

    offset = (w - 1) // 2
    valid = np.convolve(x, np.ones(w, dtype=np.float64) / w, mode="valid")
    out = np.full(n, np.nan, dtype=np.float64)
    # valid has length N - W + 1; center it in the full-length array.
    out[offset : offset + valid.size] = valid
    return out


def _natural_cubic_interpolate(
    x: np.ndarray, y: np.ndarray, x_new: np.ndarray
) -> np.ndarray:
    """Natural cubic spline via NumPy (used when SciPy is unavailable)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_new = np.asarray(x_new, dtype=np.float64)
    n = x.size
    if n < 3:
        return np.interp(x_new, x, y)

    h = np.diff(x)
    if np.any(h <= 0):
        raise ValueError("Wavelength axis must be strictly increasing")

    a = np.zeros((n, n), dtype=np.float64)
    rhs = np.zeros(n, dtype=np.float64)
    a[0, 0] = 1.0
    a[-1, -1] = 1.0
    for i in range(1, n - 1):
        a[i, i - 1] = h[i - 1]
        a[i, i] = 2.0 * (h[i - 1] + h[i])
        a[i, i + 1] = h[i]
        rhs[i] = 6.0 * (
            (y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]
        )
    m = np.linalg.solve(a, rhs)

    out = np.full(x_new.shape, np.nan, dtype=np.float64)
    for j, xi in enumerate(x_new):
        if xi < x[0] or xi > x[-1]:
            continue
        i = int(np.searchsorted(x, xi, side="right") - 1)
        i = min(max(i, 0), n - 2)
        hi = h[i]
        alpha = (x[i + 1] - xi) / hi
        beta = (xi - x[i]) / hi
        out[j] = (
            alpha * y[i]
            + beta * y[i + 1]
            + ((alpha**3 - alpha) * m[i] + (beta**3 - beta) * m[i + 1])
            * (hi**2)
            / 6.0
        )
    return out


def _interpolate_series(
    wavelengths: np.ndarray,
    intensities: np.ndarray,
    new_wavelengths: np.ndarray,
    method: str,
) -> np.ndarray:
    """Interpolate one intensity series onto ``new_wavelengths``."""
    wl = np.asarray(wavelengths, dtype=np.float64)
    y = np.asarray(intensities, dtype=np.float64)
    mask = np.isfinite(wl) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return np.full(new_wavelengths.shape, np.nan, dtype=np.float64)

    x = wl[mask]
    v = y[mask]
    order = np.argsort(x)
    x = x[order]
    v = v[order]
    # Drop duplicate wavelengths (keep first).
    uniq = np.concatenate(([True], np.diff(x) > 0))
    x = x[uniq]
    v = v[uniq]
    if x.size < 2:
        return np.full(new_wavelengths.shape, np.nan, dtype=np.float64)

    if method == "Linear":
        out = np.interp(new_wavelengths, x, v, left=np.nan, right=np.nan)
        return out.astype(np.float64)

    # Cubic spline
    if _ScipyCubicSpline is not None:
        spline = _ScipyCubicSpline(x, v, extrapolate=False)
        return np.asarray(spline(new_wavelengths), dtype=np.float64)
    return _natural_cubic_interpolate(x, v, new_wavelengths)


def _interpolate_reflectance_series(
    wavelengths: np.ndarray,
    reflectance: np.ndarray,
    new_wavelengths: np.ndarray,
    method: str,
) -> np.ndarray:
    """Resample reflectance onto ``new_wavelengths``.

    Target points outside the finite reflectance span are clamped to that span
    so edge NaNs from averaging do not produce an empty or crashing result.
    """
    wl = np.asarray(wavelengths, dtype=np.float64)
    values = np.asarray(reflectance, dtype=np.float64)
    finite = np.isfinite(wl) & np.isfinite(values)
    if int(finite.sum()) < 2:
        raise ValueError(
            "Reflectance does not have enough finite wavelength points to interpolate."
        )
    low = float(np.min(wl[finite]))
    high = float(np.max(wl[finite]))
    if high <= low:
        raise ValueError(
            "Reflectance does not have enough finite wavelength points to interpolate."
        )
    queries = np.clip(np.asarray(new_wavelengths, dtype=np.float64), low, high)
    return _interpolate_series(wl, values, queries, method)


def _prepare_si_axis(
    wavelengths: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Finite, strictly increasing wavelength/value pairs for silicon interpolation."""
    wl = np.asarray(wavelengths, dtype=np.float64)
    series = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(wl) & np.isfinite(series)
    if int(mask.sum()) < 2:
        raise ValueError("Silicon data does not have enough finite points to interpolate.")
    order = np.argsort(wl[mask])
    x = wl[mask][order]
    y = series[mask][order]
    unique = np.concatenate(([True], np.diff(x) > 0))
    x = x[unique]
    y = y[unique]
    if x.size < 2:
        raise ValueError("Silicon data does not have enough finite points to interpolate.")
    return x, y


def _extrapolate_linear(
    x: np.ndarray, y: np.ndarray, queries: np.ndarray
) -> np.ndarray:
    """Linear interpolation that continues the end-segment slope past the data."""
    out = np.interp(queries, x, y).astype(np.float64, copy=True)
    left_slope = (y[1] - y[0]) / (x[1] - x[0])
    right_slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
    below = queries < x[0]
    above = queries > x[-1]
    out[below] = y[0] + left_slope * (queries[below] - x[0])
    out[above] = y[-1] + right_slope * (queries[above] - x[-1])
    return out


def _extrapolate_nearest(
    x: np.ndarray, y: np.ndarray, queries: np.ndarray
) -> np.ndarray:
    """Nearest-neighbor interpolation, using the end samples outside the data."""
    distances = np.abs(x.reshape(-1, 1) - queries.reshape(1, -1))
    return y[np.argmin(distances, axis=0)].astype(np.float64, copy=False)


def _interpolate_si_series(
    wavelengths: np.ndarray,
    values: np.ndarray,
    new_wavelengths: np.ndarray,
    method: str,
) -> np.ndarray:
    """Resample one silicon series, extrapolating outside the imported span."""
    x, y = _prepare_si_axis(wavelengths, values)
    queries = np.asarray(new_wavelengths, dtype=np.float64)
    kind = {"Linear": "linear", "Cubic": "cubic", "Nearest": "nearest"}.get(method)
    if kind is None:
        raise ValueError(f"Unknown interpolation method: {method}")
    if kind == "cubic" and x.size < 4:
        raise ValueError("Cubic interpolation needs at least four silicon points.")
    try:
        from scipy.interpolate import interp1d
    except ImportError:
        interp1d = None  # type: ignore[assignment]
    if interp1d is not None:
        interpolator = interp1d(
            x, y, kind=kind, bounds_error=False, fill_value="extrapolate"
        )
        return np.asarray(interpolator(queries), dtype=np.float64)
    if kind == "nearest":
        return _extrapolate_nearest(x, y, queries)
    if kind == "cubic":
        interior = _natural_cubic_interpolate(x, y, queries)
        outside = ~np.isfinite(interior)
        if np.any(outside):
            interior = interior.copy()
            interior[outside] = _extrapolate_linear(x, y, queries[outside])
        return interior
    return _extrapolate_linear(x, y, queries)


# Denominators at or below this count difference are treated as zero.
_REFLECTANCE_DENOM_FLOOR = 1e-9


def _reflectance_ratio(
    sample_avg: np.ndarray,
    baseline_avg: np.ndarray,
    standard_avg: np.ndarray,
) -> np.ndarray:
    """Reflectance = (Sample_avg - Baseline_avg) / (Standard_avg - Baseline_avg).

    Points with a zero or tiny denominator, or any non-finite input, become
    ``np.nan`` instead of ``inf``.
    """
    sample = np.asarray(sample_avg, dtype=np.float64)
    baseline = np.asarray(baseline_avg, dtype=np.float64)
    standard = np.asarray(standard_avg, dtype=np.float64)
    numerator = sample - baseline
    denominator = standard - baseline
    with np.errstate(divide="ignore", invalid="ignore"):
        reflectance = numerator / denominator
    undefined = (~np.isfinite(reflectance)) | (
        np.abs(denominator) <= _REFLECTANCE_DENOM_FLOOR
    )
    reflectance = np.array(reflectance, dtype=np.float64, copy=True)
    reflectance[undefined] = np.nan
    return reflectance


def _build_uniform_wavelengths(
    start_wl: float, end_wl: float, step_wl: float, wl_min: float, wl_max: float
) -> np.ndarray:
    """Build a clamped uniform wavelength grid inside the raw spectral range."""
    start = max(float(start_wl), float(wl_min))
    end = min(float(end_wl), float(wl_max))
    step = float(step_wl)
    if step <= 0:
        raise ValueError("Step size must be positive")
    if end < start:
        raise ValueError("End wavelength must be >= start wavelength")
    grid = np.arange(start, end + step / 2.0, step, dtype=np.float64)
    grid = grid[(grid >= wl_min - 1e-12) & (grid <= wl_max + 1e-12)]
    if grid.size == 0:
        raise ValueError("Uniform grid is empty after clamping to raw range")
    return grid


class SpectrumTableModel(QAbstractTableModel):
    """Ordered column model: raw traces, adjacent averages, and appended interp columns."""

    def __init__(
        self,
        wavelengths: np.ndarray,
        baseline: Optional[np.ndarray],
        standard: Optional[np.ndarray],
        sample: Optional[np.ndarray],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._n = int(np.asarray(wavelengths).size)
        # Each entry: (header, values | None, kind)
        # kind is "wavelength", "source:<Trace>", "avg", "interp_wl", or "interp:<Trace>".
        self._columns: list[tuple[str, Optional[np.ndarray], str]] = [
            (
                "Wavelength (nm)",
                np.asarray(wavelengths, dtype=np.float64),
                "wavelength",
            ),
            (
                "Baseline (counts)",
                None if baseline is None else np.asarray(baseline, dtype=np.float64),
                "source:Baseline",
            ),
            (
                "Standard (counts)",
                None if standard is None else np.asarray(standard, dtype=np.float64),
                "source:Standard",
            ),
            (
                "Sample (counts)",
                None if sample is None else np.asarray(sample, dtype=np.float64),
                "source:Sample",
            ),
        ]

    def wavelengths(self) -> np.ndarray:
        series = self._columns[0][1]
        assert series is not None
        return np.asarray(series, dtype=np.float64)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.isValid():
            return 0
        return self._n

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.isValid():
            return 0
        return len(self._columns)

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> object:
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            if 0 <= section < len(self._columns):
                return self._columns[section][0]
            return None
        if orientation == Qt.Orientation.Vertical:
            return section + 1
        return None

    def data(  # noqa: N802
        self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole
    ) -> object:
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        row = index.row()
        col = index.column()
        if row < 0 or row >= self._n or col < 0 or col >= len(self._columns):
            return None
        _header, series, kind = self._columns[col]
        # Interp columns are shorter than raw N — remaining rows stay blank.
        if series is None or row >= series.size:
            return ""
        value = float(series[row])
        if math.isnan(value):
            if kind in {"avg", "reflectance", "reflectance_interp"}:
                return "N/A"
            return ""
        return f"{value:.6f}"

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:  # noqa: N802
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable

    def source_column_index(self, trace_name: str) -> int:
        """Return column index of the raw ``<Trace> (counts)`` measurement."""
        tag = f"source:{trace_name}"
        for i, (_header, _series, kind) in enumerate(self._columns):
            if kind == tag:
                return i
        raise KeyError(f"Source column not found for {trace_name}")

    def raw_trace(self, trace_name: str) -> Optional[np.ndarray]:
        idx = self.source_column_index(trace_name)
        return self._columns[idx][1]

    def set_source_trace(self, trace_name: str, values: np.ndarray) -> None:
        """Replace one raw measurement column without touching averages."""
        idx = self.source_column_index(trace_name)
        header, _series, kind = self._columns[idx]
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (self._n,):
            raise ValueError(
                f"{trace_name} length {array.size} does not match "
                f"the table length {self._n}"
            )
        self._columns[idx] = (header, array, kind)
        top = self.index(0, idx)
        bottom = self.index(self._n - 1, idx)
        self.dataChanged.emit(top, bottom)

    def apply_imported_slots(self, updates: dict[str, np.ndarray]) -> None:
        """Replace raw slot columns and drop stale average/interpolation columns."""
        if not updates:
            return
        for trace_name, values in updates.items():
            array = np.asarray(values, dtype=np.float64)
            axis = self.wavelengths()
            if array.shape != axis.shape:
                raise ValueError(
                    f"{trace_name} length {array.size} does not match "
                    f"the measurement length {axis.size}"
                )
        self.beginResetModel()
        kept: list[tuple[str, Optional[np.ndarray], str]] = []
        for header, series, kind in self._columns:
            if (
                kind == "wavelength"
                or kind.startswith("source:")
                or kind in {
                    "si_wl",
                    "si_n",
                    "si_k",
                    "si_interp_wl",
                    "si_interp_n",
                    "si_interp_k",
                }
            ):
                kept.append((header, series, kind))
        self._columns = kept
        for trace_name, values in updates.items():
            tag = f"source:{trace_name}"
            for index, (header, _series, kind) in enumerate(self._columns):
                if kind == tag:
                    self._columns[index] = (
                        header,
                        np.asarray(values, dtype=np.float64),
                        kind,
                    )
                    break
        self.endResetModel()

    def insert_average_adjacent(
        self, trace_name: str, window: int, values: np.ndarray
    ) -> int:
        """Insert ``<Trace>_Avg_W{W}`` immediately to the right of the source column."""
        source_idx = self.source_column_index(trace_name)
        insert_at = source_idx + 1
        header = f"{trace_name}_Avg_W{int(window)} (counts)"
        self.beginInsertColumns(QModelIndex(), insert_at, insert_at)
        self._columns.insert(
            insert_at,
            (header, np.asarray(values, dtype=np.float64), "avg"),
        )
        self.endInsertColumns()
        return insert_at

    def latest_average(self, trace_name: str) -> Optional[np.ndarray]:
        """Return the newest average inserted beside ``trace_name``, if any."""
        source_idx = self.source_column_index(trace_name)
        for _header, series, kind in self._columns[source_idx + 1 :]:
            if str(kind).startswith("source:"):
                break
            if kind == "avg" and series is not None:
                return np.asarray(series, dtype=np.float64)
        return None

    def set_reflectance(self, values: np.ndarray) -> None:
        """Append or replace the Reflectance column at the far right."""
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (self._n,):
            raise ValueError(
                f"Reflectance length {array.size} does not match "
                f"the table length {self._n}"
            )
        header = "Relative Reflectance"
        idx = self._find_kind_index("reflectance")
        if idx is None:
            new_col = self.columnCount()
            self.beginInsertColumns(QModelIndex(), new_col, new_col)
            self._columns.append((header, array, "reflectance"))
            self.endInsertColumns()
            return
        self._columns[idx] = (header, array, "reflectance")
        top = self.index(0, idx)
        bottom = self.index(self._n - 1, idx)
        self.dataChanged.emit(top, bottom)
        self.headerDataChanged.emit(Qt.Orientation.Horizontal, idx, idx)

    def clear_interpolated_reflectance(self) -> None:
        """Drop resampled reflectance columns after the source ratio changes."""
        kinds = {"reflectance_interp_wl", "reflectance_interp"}
        if not any(column[2] in kinds for column in self._columns):
            return
        self.beginResetModel()
        self._columns = [column for column in self._columns if column[2] not in kinds]
        self.endResetModel()

    def set_interpolated_reflectance(
        self, wavelengths: np.ndarray, values: np.ndarray
    ) -> None:
        """Show the reflectance grid beside Interpolated Relative Reflectance.

        Arrays may be shorter than the raw table; leftover rows stay blank.
        """
        wl_vals = np.asarray(wavelengths, dtype=np.float64)
        reflectance_vals = np.asarray(values, dtype=np.float64)
        if wl_vals.shape != reflectance_vals.shape or wl_vals.size < 1:
            raise ValueError(
                "Interpolated reflectance wavelengths and values must be the same non-empty length."
            )
        columns = (
            ("Relative Reflectance Wavelength (nm)", wl_vals, "reflectance_interp_wl"),
            ("Interpolated Relative Reflectance", reflectance_vals, "reflectance_interp"),
        )
        for header, array, kind in columns:
            idx = self._find_kind_index(kind)
            if idx is None:
                new_col = self.columnCount()
                self.beginInsertColumns(QModelIndex(), new_col, new_col)
                self._columns.append((header, array, kind))
                self.endInsertColumns()
                continue
            self._columns[idx] = (header, array, kind)
            top = self.index(0, idx)
            bottom = self.index(self._n - 1, idx)
            self.dataChanged.emit(top, bottom)
            self.headerDataChanged.emit(Qt.Orientation.Horizontal, idx, idx)

    def set_silicon_reference(
        self,
        wavelengths: np.ndarray,
        refractive_index: np.ndarray,
        extinction: np.ndarray,
    ) -> None:
        """Append or replace Si Wavelength, Si n, and Si k without touching measurements.

        The table grows when the reference is longer than the current row count.
        Measurement cells past their own data stay blank.
        """
        wl_vals = np.asarray(wavelengths, dtype=np.float64)
        n_vals = np.asarray(refractive_index, dtype=np.float64)
        k_vals = np.asarray(extinction, dtype=np.float64)
        if (
            wl_vals.ndim != 1
            or wl_vals.shape != n_vals.shape
            or wl_vals.shape != k_vals.shape
            or wl_vals.size < 2
        ):
            raise ValueError(
                "Silicon wavelength, n, and k must be the same length "
                "and contain at least two points."
            )
        columns = (
            ("Si Wavelength (nm)", wl_vals, "si_wl"),
            ("Si n", n_vals, "si_n"),
            ("Si k", k_vals, "si_k"),
        )
        self.beginResetModel()
        self._n = max(self._n, int(wl_vals.size))
        self._columns = [
            column
            for column in self._columns
            if column[2]
            not in {
                "si_wl",
                "si_n",
                "si_k",
                "si_reflectance",
                "si_interp_wl",
                "si_interp_n",
                "si_interp_k",
            }
        ]
        self._columns.extend(columns)
        self.endResetModel()

    def set_interpolated_silicon(
        self,
        wavelengths: np.ndarray,
        refractive_index: np.ndarray,
        extinction: np.ndarray,
    ) -> None:
        """Append or replace the interpolated silicon wavelength, n, and k columns."""
        wl_vals = np.asarray(wavelengths, dtype=np.float64)
        n_vals = np.asarray(refractive_index, dtype=np.float64)
        k_vals = np.asarray(extinction, dtype=np.float64)
        if (
            wl_vals.ndim != 1
            or wl_vals.shape != n_vals.shape
            or wl_vals.shape != k_vals.shape
            or wl_vals.size < 1
        ):
            raise ValueError(
                "Interpolated silicon wavelength, n, and k must be the same non-empty length."
            )
        columns = (
            ("Interpolated Si Wavelength (nm)", wl_vals, "si_interp_wl"),
            ("Interpolated Si n", n_vals, "si_interp_n"),
            ("Interpolated Si k", k_vals, "si_interp_k"),
        )
        self.beginResetModel()
        self._n = max(self._n, int(wl_vals.size))
        self._columns = [
            column
            for column in self._columns
            if column[2] not in {"si_interp_wl", "si_interp_n", "si_interp_k"}
        ]
        self._columns.extend(columns)
        self.endResetModel()

    def _find_kind_index(self, kind: str) -> Optional[int]:
        for i, (_header, _series, col_kind) in enumerate(self._columns):
            if col_kind == kind:
                return i
        return None

    def set_interpolation_columns(
        self,
        interp_wl: np.ndarray,
        trace_name: str,
        interp_intensity: np.ndarray,
    ) -> None:
        """Append/update Interp_Wavelength + ``<Trace>_Interp`` on the far right.

        Arrays may be shorter than ``len(raw_wavelengths)``; trailing rows render blank.
        Raw measurement and average columns are never cleared or shortened.
        """
        wl_vals = np.asarray(interp_wl, dtype=np.float64)
        inten_vals = np.asarray(interp_intensity, dtype=np.float64)
        wl_header = "Interp_Wavelength (nm)"
        inten_header = f"{trace_name}_Interp (counts)"
        wl_kind = "interp_wl"
        inten_kind = f"interp:{trace_name}"

        wl_idx = self._find_kind_index(wl_kind)
        inten_idx = self._find_kind_index(inten_kind)

        if wl_idx is None:
            new_col = self.columnCount()
            self.beginInsertColumns(QModelIndex(), new_col, new_col)
            self._columns.append((wl_header, wl_vals, wl_kind))
            self.endInsertColumns()
        else:
            self._columns[wl_idx] = (wl_header, wl_vals, wl_kind)
            top = self.index(0, wl_idx)
            bottom = self.index(self._n - 1, wl_idx)
            self.dataChanged.emit(top, bottom)
            self.headerDataChanged.emit(Qt.Orientation.Horizontal, wl_idx, wl_idx)

        if inten_idx is None:
            new_col = self.columnCount()
            self.beginInsertColumns(QModelIndex(), new_col, new_col)
            self._columns.append((inten_header, inten_vals, inten_kind))
            self.endInsertColumns()
        else:
            self._columns[inten_idx] = (inten_header, inten_vals, inten_kind)
            top = self.index(0, inten_idx)
            bottom = self.index(self._n - 1, inten_idx)
            self.dataChanged.emit(top, bottom)
            self.headerDataChanged.emit(
                Qt.Orientation.Horizontal, inten_idx, inten_idx
            )

    def set_interpolation_batch(
        self, interp_wl: np.ndarray, intensities: dict[str, np.ndarray]
    ) -> None:
        """Replace the shared interpolated grid and every supplied trace column.

        Intensity columns follow Baseline, Standard, Sample. Traces omitted from
        ``intensities`` are dropped so the table stays aligned to this grid.
        """
        wl_vals = np.asarray(interp_wl, dtype=np.float64)
        prepared = {
            name: np.asarray(values, dtype=np.float64)
            for name, values in intensities.items()
        }
        self.beginResetModel()
        kept = [
            column
            for column in self._columns
            if column[2] != "interp_wl" and not str(column[2]).startswith("interp:")
        ]
        kept.append(("Interp_Wavelength (nm)", wl_vals, "interp_wl"))
        for trace_name in _SOURCE_TRACES:
            if trace_name not in prepared:
                continue
            kept.append(
                (
                    f"{trace_name}_Interp (counts)",
                    prepared[trace_name],
                    f"interp:{trace_name}",
                )
            )
        self._columns = kept
        self.endResetModel()

    def export_rows(self) -> tuple[list[str], list[list[str]]]:
        """Return (headers, rows) in current left-to-right visual order."""
        headers = [col[0] for col in self._columns]
        rows: list[list[str]] = []
        for r in range(self._n):
            row: list[str] = []
            for c in range(len(self._columns)):
                text = self.data(self.index(r, c), Qt.ItemDataRole.DisplayRole)
                row.append("" if text is None else str(text))
            rows.append(row)
        return headers, rows


class PostProcessDialog(QDialog):
    """Modal data viewer with averaging and uniform-grid interpolation."""

    def __init__(
        self,
        mode_display: str,
        mode_key: str,
        wavelengths: np.ndarray,
        baseline: Optional[np.ndarray],
        standard: Optional[np.ndarray],
        sample: Optional[np.ndarray],
        parent: Optional[QWidget] = None,
        on_imported: Optional[
            Callable[
                [np.ndarray, dict[Optional[str], dict[str, np.ndarray]]],
                Optional[list[str]],
            ]
        ] = None,
    ) -> None:
        super().__init__(parent)
        self._mode_display = mode_display
        self._mode_key = mode_key
        self._on_imported = on_imported
        self._reflectance: Optional[np.ndarray] = None
        self.interpolated_reflectance: Optional[dict[str, np.ndarray]] = None
        self.si_wavelength: Optional[np.ndarray] = None
        self.si_n: Optional[np.ndarray] = None
        self.si_k: Optional[np.ndarray] = None
        self.si_interp_wavelength: Optional[np.ndarray] = None
        self.si_interp_n: Optional[np.ndarray] = None
        self.si_interp_k: Optional[np.ndarray] = None
        self._raw_wavelengths = np.asarray(wavelengths, dtype=np.float64)
        self._wl_min = float(self._raw_wavelengths.min())
        self._wl_max = float(self._raw_wavelengths.max())
        self.setWindowTitle("Post-Processing - Data Viewer")
        self.setModal(True)
        self.resize(1400, 900)
        self.setStyleSheet(
            """
            QDialog {
                background-color: #12141a;
                color: #e8eaed;
            }
            QScrollArea {
                background-color: transparent;
                border: none;
            }
            QScrollArea > QWidget > QWidget {
                background-color: transparent;
            }
            QLabel {
                color: #c5cad4;
            }
            QGroupBox {
                color: #e8eaed;
                border: 1px solid #3d4450;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 8px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QTableView {
                background-color: #1a1d23;
                alternate-background-color: #21252d;
                gridline-color: #2e3440;
                color: #e8eaed;
                selection-background-color: #2a4568;
                border: 1px solid #3d4450;
            }
            QHeaderView::section {
                background-color: #242a35;
                color: #e8eaed;
                padding: 6px;
                border: 1px solid #3d4450;
                font-weight: 600;
            }
            QPushButton {
                background-color: #2a2f3a;
                border: 1px solid #3d4450;
                border-radius: 6px;
                padding: 2px 16px;
                min-height: 22px;
            }
            QPushButton:hover {
                background-color: #353b48;
            }
            QPushButton:disabled, QToolButton:disabled {
                color: #6d7480;
            }
            QToolButton {
                background-color: #2a2f3a;
                border: 1px solid #3d4450;
                border-radius: 6px;
                padding: 2px 12px;
                min-height: 22px;
                color: #e8eaed;
            }
            QToolButton:hover {
                background-color: #353b48;
            }
            QToolButton::menu-indicator {
                image: none;
                width: 0px;
            }
            QPushButton#accentBtn {
                background-color: #123e4f;
                border: 1px solid #00bcd4;
                color: #e8fbff;
                font-weight: 600;
            }
            QPushButton#accentBtn:hover:!disabled {
                background-color: #16566c;
                border: 1px solid #00bcd4;
            }
            QPushButton#accentBtn:pressed:!disabled {
                background-color: #0e3340;
                border: 1px solid #00bcd4;
            }
            QMenu {
                background-color: #1e2229;
                color: #e8eaed;
                border: 1px solid #3d4450;
            }
            QMenu::item:selected {
                background-color: #2a4568;
            }
            QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
                background-color: #1e2229;
                border: 1px solid #3d4450;
                border-radius: 4px;
                padding: 4px 8px;
                min-height: 26px;
                color: #e8eaed;
            }
            QLineEdit:focus, QSpinBox:focus, QComboBox:focus {
                border: 1px solid #00bcd4;
            }
            """
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        self._summary = QLabel(
            f"Optical mode: {mode_display}  ·  "
            f"{int(self._raw_wavelengths.size)} spectral points"
        )
        self._summary.setStyleSheet("font-weight: 600; color: #e8eaed;")
        self._summary.setWordWrap(True)
        self._summary.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )

        self._back_btn = QPushButton("Back")
        self._back_btn.setObjectName("accentBtn")
        self._back_btn.setMinimumWidth(88)
        self._back_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._back_btn.setAutoDefault(False)
        self._back_btn.setDefault(False)
        self._back_btn.clicked.connect(self.accept)

        header = QHBoxLayout()
        header.setSpacing(12)
        header.addWidget(self._summary, stretch=1)
        header.addWidget(
            self._back_btn,
            alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )
        root.addLayout(header)

        self._model = SpectrumTableModel(
            wavelengths, baseline, standard, sample, parent=self
        )
        self._table = self._make_table_view()
        self._table.setModel(self._model)

        body = QHBoxLayout()
        body.setSpacing(12)
        body.addWidget(self._build_sidebar())
        body.addLayout(self._build_table_pane(), stretch=1)
        root.addLayout(body, stretch=1)

        self._sync_average_buttons(
            baseline is not None,
            standard is not None,
            sample is not None,
        )
        self._populate_interp_trace_combo(
            baseline is not None,
            standard is not None,
            sample is not None,
        )

    def _build_sidebar(self) -> QScrollArea:
        content = QWidget()
        content.setMinimumWidth(300)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(self._build_average_group())
        layout.addWidget(self._build_interpolation_group())
        layout.addWidget(self._build_reflectance_group())
        layout.addWidget(self._build_silicon_group())
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        scroll.setFixedWidth(320)
        scroll.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding
        )
        return scroll

    def _build_table_pane(self) -> QVBoxLayout:
        pane = QVBoxLayout()
        pane.setContentsMargins(0, 0, 0, 0)
        pane.setSpacing(8)
        pane.addWidget(self._table, stretch=1)

        self._export_table_btn = QPushButton("Export")
        self._export_table_btn.setObjectName("accentBtn")
        self._export_table_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._export_table_btn.setToolTip(
            "Export all table columns, including relative reflectance, "
            "interpolated relative reflectance, and imported silicon values"
        )
        self._export_table_btn.clicked.connect(self._export_processed_table)

        self._import_csv_btn = QPushButton("Import")
        self._import_csv_btn.setObjectName("accentBtn")
        self._import_csv_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._import_csv_btn.setToolTip(
            "Load Baseline, Standard, and Sample together from a multi-column CSV"
        )
        self._import_csv_btn.clicked.connect(self._import_measurement_csv)

        bar = QHBoxLayout()
        bar.setSpacing(8)
        bar.addStretch(1)
        bar.addWidget(self._import_csv_btn)
        bar.addWidget(self._export_table_btn)
        pane.addLayout(bar)
        return pane

    def _make_grid_edit(
        self, default: float, minimum: float, maximum: float, placeholder: str
    ) -> QLineEdit:
        edit = QLineEdit(f"{default:.3f}")
        edit.setPlaceholderText(placeholder)
        validator = QDoubleValidator(minimum, maximum, 3, edit)
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        validator.setLocale(QLocale.c())
        edit.setValidator(validator)
        return edit

    def _field_float(self, edit: QLineEdit) -> float:
        text = edit.text().strip().replace(",", ".")
        try:
            value = float(text)
        except ValueError as exc:
            label = edit.placeholderText() or "This field"
            raise ValueError(f"{label} must be a number.") from exc
        if not math.isfinite(value):
            raise ValueError("Wavelength values must be finite.")
        return value

    def _build_average_group(self) -> QGroupBox:
        group = QGroupBox("Average")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        window_row = QHBoxLayout()
        window_row.setSpacing(8)
        window_row.addWidget(QLabel("Window size"))
        self._window_spin = QSpinBox()
        self._window_spin.setRange(2, 50)
        self._window_spin.setValue(5)
        self._window_spin.setSingleStep(1)
        self._window_spin.setMinimumWidth(72)
        window_row.addWidget(self._window_spin, stretch=1)
        layout.addLayout(window_row)

        self._avg_menu_btn = QToolButton(self)
        self._avg_menu_btn.setObjectName("avgTraceBtn")
        self._avg_menu_btn.setText("Average trace ▾")
        self._avg_menu_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._avg_menu_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self._avg_menu_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._avg_menu_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._avg_menu_btn.setToolTip(
            "Average a single measurement column (inserted beside its source)"
        )

        avg_menu = QMenu(self._avg_menu_btn)
        self._avg_baseline_action = QAction("Average Baseline", self)
        self._avg_standard_action = QAction("Average Standard", self)
        self._avg_sample_action = QAction("Average Sample", self)
        self._avg_baseline_action.triggered.connect(
            lambda: self._average_trace("Baseline")
        )
        self._avg_standard_action.triggered.connect(
            lambda: self._average_trace("Standard")
        )
        self._avg_sample_action.triggered.connect(
            lambda: self._average_trace("Sample")
        )
        avg_menu.addAction(self._avg_baseline_action)
        avg_menu.addAction(self._avg_standard_action)
        avg_menu.addAction(self._avg_sample_action)
        self._avg_menu_btn.setMenu(avg_menu)

        self._avg_all_btn = QPushButton("Average All")
        self._avg_all_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._avg_all_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._avg_all_btn.setToolTip(
            "Average Baseline, Standard, and Sample at once "
            "(each result inserted beside its source column)"
        )
        self._avg_all_btn.clicked.connect(self._average_all_traces)

        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        action_row.addWidget(self._avg_menu_btn, stretch=1)
        action_row.addWidget(self._avg_all_btn, stretch=1)
        layout.addLayout(action_row)
        return group

    def _build_interpolation_group(self) -> QGroupBox:
        group = QGroupBox("Interpolation")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        start_default = float(math.ceil(self._wl_min))
        end_default = float(math.floor(self._wl_max))
        if start_default >= end_default:
            start_default = self._wl_min
            end_default = self._wl_max

        self._start_wl_edit = self._make_grid_edit(
            start_default, self._wl_min, self._wl_max, "Start wavelength (nm)"
        )
        self._end_wl_edit = self._make_grid_edit(
            end_default, self._wl_min, self._wl_max, "End wavelength (nm)"
        )
        self._step_wl_edit = self._make_grid_edit(1.0, 0.1, 50.0, "Step size (nm)")

        # Kept so the existing interpolate path can read the selected trace.
        self._interp_trace_combo = QComboBox(self)
        self._interp_trace_combo.hide()

        form = QFormLayout()
        form.setSpacing(8)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        form.addRow("Start", self._start_wl_edit)
        form.addRow("End", self._end_wl_edit)
        form.addRow("Step size", self._step_wl_edit)
        self._interp_method_combo = QComboBox()
        self._interp_method_combo.addItems(["Linear", "Cubic Spline"])
        form.addRow("Method", self._interp_method_combo)
        layout.addLayout(form)

        self._interpolate_btn = QToolButton(self)
        self._interpolate_btn.setText("Interpolate ▾")
        self._interpolate_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._interpolate_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self._interpolate_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._interpolate_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._interpolate_btn.setToolTip(
            "Interpolate Baseline, Standard, or Sample onto this wavelength grid"
        )
        interp_menu = QMenu(self._interpolate_btn)
        self._interp_baseline_action = QAction("Baseline", self)
        self._interp_standard_action = QAction("Standard", self)
        self._interp_sample_action = QAction("Sample", self)
        self._interp_baseline_action.triggered.connect(
            lambda: self._interpolate_selected("Baseline")
        )
        self._interp_standard_action.triggered.connect(
            lambda: self._interpolate_selected("Standard")
        )
        self._interp_sample_action.triggered.connect(
            lambda: self._interpolate_selected("Sample")
        )
        interp_menu.addAction(self._interp_baseline_action)
        interp_menu.addAction(self._interp_standard_action)
        interp_menu.addAction(self._interp_sample_action)
        self._interpolate_btn.setMenu(interp_menu)
        layout.addWidget(self._interpolate_btn)

        self._interpolate_all_btn = QPushButton("Interpolate all")
        self._interpolate_all_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._interpolate_all_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._interpolate_all_btn.setToolTip(
            "Interpolate Baseline, Standard, and Sample onto this wavelength grid"
        )
        self._interpolate_all_btn.clicked.connect(self._interpolate_all)
        layout.addWidget(self._interpolate_all_btn)
        return group

    def _build_reflectance_group(self) -> QGroupBox:
        """Relative-reflectance controls, with a grid independent of the traces."""
        group = QGroupBox("Reflectance")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        self._ref_start_wl_edit = self._make_grid_edit(
            350.0, 0.0, 5000.0, "Reflectance start wavelength (nm)"
        )
        self._ref_end_wl_edit = self._make_grid_edit(
            1000.0, 0.0, 5000.0, "Reflectance end wavelength (nm)"
        )
        self._ref_step_wl_edit = self._make_grid_edit(
            1.0, 0.1, 50.0, "Reflectance step size (nm)"
        )
        self._ref_method_combo = QComboBox()
        self._ref_method_combo.addItems(["Linear", "Cubic Spline"])
        self._ref_method_combo.setCurrentText("Linear")

        form = QFormLayout()
        form.setSpacing(8)
        form.setLabelAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        form.addRow("Start", self._ref_start_wl_edit)
        form.addRow("End", self._ref_end_wl_edit)
        form.addRow("Step size", self._ref_step_wl_edit)
        form.addRow("Method", self._ref_method_combo)
        layout.addLayout(form)

        self._reflectance_btn = QPushButton("Relative Reflectance")
        self._reflectance_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._reflectance_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._reflectance_btn.setToolTip(
            "Relative Reflectance = (Sample_avg − Baseline_avg) / "
            "(Standard_avg − Baseline_avg) at each wavelength"
        )
        self._reflectance_btn.clicked.connect(self._calculate_reflectance)
        layout.addWidget(self._reflectance_btn)

        self._interp_reflectance_btn = QPushButton("Interpolate Relative Reflectance")
        self._interp_reflectance_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._interp_reflectance_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._interp_reflectance_btn.setToolTip(
            "Resample relative reflectance onto the Reflectance group wavelength grid"
        )
        self._interp_reflectance_btn.clicked.connect(self._interpolate_reflectance)
        layout.addWidget(self._interp_reflectance_btn)
        return group

    def _build_silicon_group(self) -> QGroupBox:
        """Silicon n and k import, with a grid used only for those curves."""
        group = QGroupBox("Silicon Calibration")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)
        self._import_si_btn = QPushButton("Import Si Values")
        self._import_si_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._import_si_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._import_si_btn.setToolTip(
            "Load silicon wavelength, refractive index n, and extinction coefficient k"
        )
        self._import_si_btn.clicked.connect(self._import_si_values)
        layout.addWidget(self._import_si_btn)

        self._si_start_wl_edit = self._make_grid_edit(
            350.0, 0.0, 5000.0, "Silicon start wavelength (nm)"
        )
        self._si_end_wl_edit = self._make_grid_edit(
            1000.0, 0.0, 5000.0, "Silicon end wavelength (nm)"
        )
        self._si_step_wl_edit = self._make_grid_edit(
            1.0, 0.1, 50.0, "Silicon step size (nm)"
        )
        self._si_method_combo = QComboBox()
        self._si_method_combo.addItems(["Linear", "Cubic", "Nearest"])
        self._si_method_combo.setCurrentText("Linear")

        form = QFormLayout()
        form.setSpacing(8)
        form.setLabelAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        form.addRow("Start", self._si_start_wl_edit)
        form.addRow("End", self._si_end_wl_edit)
        form.addRow("Step size", self._si_step_wl_edit)
        form.addRow("Method", self._si_method_combo)
        layout.addLayout(form)

        self._interp_si_btn = QPushButton("Interpolate Si (n and k)")
        self._interp_si_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._interp_si_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._interp_si_btn.setToolTip(
            "Resample imported silicon n and k onto this wavelength grid"
        )
        self._interp_si_btn.clicked.connect(self._interpolate_silicon)
        layout.addWidget(self._interp_si_btn)
        return group

    def _interpolate_selected(self, trace_name: str) -> None:
        """Choose one trace from the Interpolate menu, then run the grid routine."""
        index = self._interp_trace_combo.findText(trace_name)
        if index < 0:
            QMessageBox.warning(
                self,
                "No Trace Selected",
                f"No captured {trace_name} data available to interpolate.",
            )
            return
        self._interp_trace_combo.setCurrentIndex(index)
        self._interpolate_grid()

    def _populate_interp_trace_combo(
        self, has_baseline: bool, has_standard: bool, has_sample: bool
    ) -> None:
        self._interp_trace_combo.clear()
        for trace, available in (
            ("Baseline", has_baseline),
            ("Standard", has_standard),
            ("Sample", has_sample),
        ):
            if available:
                self._interp_trace_combo.addItem(trace)
        has_any = self._interp_trace_combo.count() > 0
        self._interpolate_btn.setEnabled(has_any)
        self._interpolate_all_btn.setEnabled(has_any)
        self._interp_baseline_action.setEnabled(has_baseline)
        self._interp_standard_action.setEnabled(has_standard)
        self._interp_sample_action.setEnabled(has_sample)

    def _make_table_view(self) -> QTableView:
        table = QTableView(self)
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QTableView.SelectionMode.ExtendedSelection)
        table.setSortingEnabled(False)
        table.verticalHeader().setVisible(True)
        table.verticalHeader().setDefaultSectionSize(22)
        table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        table.horizontalHeader().setStretchLastSection(True)
        table.setWordWrap(False)
        table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        return table

    def _active_model(self) -> SpectrumTableModel:
        return self._model

    def _active_table(self) -> QTableView:
        return self._table

    def _sync_average_buttons(
        self, has_baseline: bool, has_standard: bool, has_sample: bool
    ) -> None:
        self._avg_baseline_action.setEnabled(has_baseline)
        self._avg_standard_action.setEnabled(has_standard)
        self._avg_sample_action.setEnabled(has_sample)
        self._avg_menu_btn.setEnabled(
            has_baseline or has_standard or has_sample
        )
        self._avg_all_btn.setEnabled(
            has_baseline or has_standard or has_sample
        )

    def _average_trace(self, trace_name: str, *, quiet: bool = False) -> bool:
        """Average one trace on the active grid and insert adjacent."""
        model = self._active_model()
        table = self._active_table()
        window = int(self._window_spin.value())
        raw = model.raw_trace(trace_name)
        if raw is None:
            if not quiet:
                QMessageBox.warning(
                    self,
                    "No Data",
                    f"No captured {trace_name} data for this optical mode.",
                )
            return False
        if window > raw.size:
            if not quiet:
                QMessageBox.warning(
                    self,
                    "Window Too Large",
                    f"Window size {window} exceeds {trace_name} length "
                    f"({raw.size}).",
                )
            return False
        try:
            averaged = _moving_average_centered(raw, window)
        except ValueError as exc:
            if not quiet:
                QMessageBox.warning(self, "Moving Average Failed", str(exc))
            return False

        insert_at = model.insert_average_adjacent(trace_name, window, averaged)
        table.resizeColumnsToContents()
        table.scrollTo(
            model.index(0, insert_at),
            QTableView.ScrollHint.PositionAtCenter,
        )
        return True

    def _average_all_traces(self) -> None:
        """Average Baseline, Standard, and Sample on the active grid."""
        model = self._active_model()
        applied: list[str] = []
        skipped: list[str] = []
        for trace_name in _SOURCE_TRACES:
            if model.raw_trace(trace_name) is None:
                skipped.append(trace_name)
                continue
            if self._average_trace(trace_name, quiet=True):
                applied.append(trace_name)
            else:
                skipped.append(trace_name)

        if not applied:
            QMessageBox.warning(
                self,
                "No Data",
                "No captured traces available to average.",
            )
            return
        if skipped:
            QMessageBox.information(
                self,
                "Partial Apply",
                f"Averaged: {', '.join(applied)}.\n"
                f"Skipped (no data): {', '.join(skipped)}",
            )
        else:
            self._active_table().resizeColumnsToContents()

    def _interpolate_grid(self) -> None:
        """Append Interp_Wavelength + selected-trace intensity; keep raw rows intact."""
        trace_name = self._interp_trace_combo.currentText()
        if not trace_name:
            QMessageBox.warning(
                self,
                "No Trace Selected",
                "Select Baseline, Standard, or Sample to interpolate.",
            )
            return

        intensities = self._model.raw_trace(trace_name)
        if intensities is None:
            QMessageBox.warning(
                self,
                "No Data",
                f"No captured {trace_name} data available to interpolate.",
            )
            return

        try:
            interp_wl = self._current_interp_grid()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Grid", str(exc))
            return

        method = self._interp_method_combo.currentText()
        method_key = "Linear" if method == "Linear" else "Cubic Spline"
        try:
            interp_intensity = _interpolate_series(
                self._model.wavelengths(),
                intensities,
                interp_wl,
                method_key,
            )
        except Exception as exc:
            logger.exception("Interpolation failed")
            QMessageBox.critical(self, "Interpolation Failed", str(exc))
            return

        # Pad is implicit: store shorter arrays; table rows stay len(raw).
        self._model.set_interpolation_columns(interp_wl, trace_name, interp_intensity)
        self._table.resizeColumnsToContents()
        self._summary.setText(
            f"Optical mode: {self._mode_display}  ·  "
            f"{self._model.rowCount()} raw rows  ·  "
            f"{interp_wl.size} interpolated {trace_name} points "
            f"({interp_wl[0]:.3f}–{interp_wl[-1]:.3f} nm, "
            f"step {self._field_float(self._step_wl_edit):.3f} nm, {method_key})"
        )

    def _current_interp_method(self) -> str:
        method = self._interp_method_combo.currentText()
        return "Linear" if method == "Linear" else "Cubic Spline"

    def _current_interp_grid(self) -> np.ndarray:
        """Wavelength grid from the Interpolation group (Baseline, Standard, Sample)."""
        return _build_uniform_wavelengths(
            self._field_float(self._start_wl_edit),
            self._field_float(self._end_wl_edit),
            self._field_float(self._step_wl_edit),
            self._wl_min,
            self._wl_max,
        )

    def _current_reflectance_method(self) -> str:
        method = self._ref_method_combo.currentText()
        return "Linear" if method == "Linear" else "Cubic Spline"

    def _current_reflectance_grid(self) -> np.ndarray:
        """Wavelength grid from the Reflectance group only."""
        return _build_uniform_wavelengths(
            self._field_float(self._ref_start_wl_edit),
            self._field_float(self._ref_end_wl_edit),
            self._field_float(self._ref_step_wl_edit),
            self._wl_min,
            self._wl_max,
        )

    def _current_si_method(self) -> str:
        method = self._si_method_combo.currentText()
        if method in {"Linear", "Cubic", "Nearest"}:
            return method
        return "Linear"

    def _current_si_grid(self) -> np.ndarray:
        """Target wavelengths from the Silicon Calibration fields only."""
        start = self._field_float(self._si_start_wl_edit)
        end = self._field_float(self._si_end_wl_edit)
        step = self._field_float(self._si_step_wl_edit)
        if step <= 0:
            raise ValueError("Step size must be positive")
        if end < start:
            raise ValueError("End wavelength must be >= start wavelength")
        grid = np.arange(start, end + step / 2.0, step, dtype=np.float64)
        if grid.size < 1:
            raise ValueError("Target wavelength grid is empty")
        return grid

    def _interpolate_all(self) -> None:
        """Interpolate every captured trace onto the current wavelength grid."""
        present: list[str] = []
        skipped: list[str] = []
        for trace_name in _SOURCE_TRACES:
            if self._model.raw_trace(trace_name) is None:
                skipped.append(trace_name)
            else:
                present.append(trace_name)

        if not present:
            QMessageBox.warning(
                self,
                "No Data",
                "No captured traces available to interpolate.",
            )
            return

        try:
            interp_wl = self._current_interp_grid()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Grid", str(exc))
            return

        method_key = self._current_interp_method()
        results: dict[str, np.ndarray] = {}
        for trace_name in present:
            intensities = self._model.raw_trace(trace_name)
            if intensities is None:
                skipped.append(trace_name)
                continue
            try:
                results[trace_name] = _interpolate_series(
                    self._model.wavelengths(),
                    intensities,
                    interp_wl,
                    method_key,
                )
            except Exception as exc:
                logger.exception("Interpolation failed for %s", trace_name)
                skipped.append(f"{trace_name} ({exc})")

        if not results:
            QMessageBox.warning(
                self,
                "Interpolation Failed",
                "None of the captured traces could be interpolated.\n"
                + "\n".join(skipped),
            )
            return

        self._model.set_interpolation_batch(interp_wl, results)
        self._table.resizeColumnsToContents()
        names = ", ".join(results)
        self._summary.setText(
            f"Optical mode: {self._mode_display}  ·  "
            f"{self._model.rowCount()} raw rows  ·  "
            f"{interp_wl.size} interpolated points for {names} "
            f"({interp_wl[0]:.3f}–{interp_wl[-1]:.3f} nm, "
            f"step {self._field_float(self._step_wl_edit):.3f} nm, {method_key})"
        )
        if skipped:
            missing = [name for name in skipped if name in _SOURCE_TRACES]
            failed = [name for name in skipped if name not in _SOURCE_TRACES]
            lines = [f"Interpolated: {names}."]
            if missing:
                lines.append(f"Skipped (no data): {', '.join(missing)}.")
            if failed:
                lines.append("Skipped (failed): " + "; ".join(failed) + ".")
            QMessageBox.information(
                self,
                "Partial Interpolation",
                "\n".join(lines),
            )

    def _averaged_trace(self, trace_name: str) -> Optional[np.ndarray]:
        """Newest average for one trace, or ``None`` when it is unusable."""
        raw = self._model.raw_trace(trace_name)
        averaged = self._model.latest_average(trace_name)
        if raw is None or averaged is None:
            return None
        if averaged.shape != (self._model.rowCount(),):
            return None
        if not np.any(np.isfinite(averaged)):
            return None
        return averaged

    def _calculate_reflectance(self) -> bool:
        """Reflectance from averaged Baseline, Standard, and Sample."""
        averages: dict[str, np.ndarray] = {}
        missing: list[str] = []
        for trace_name in _SOURCE_TRACES:
            averaged = self._averaged_trace(trace_name)
            if averaged is None:
                missing.append(trace_name)
            else:
                averages[trace_name] = averaged
        if missing:
            QMessageBox.warning(
                self,
                "Measurements Required",
                "Relative Reflectance needs averaged Baseline, Standard, and Sample "
                "data at each wavelength.\n\n"
                f"Missing or incomplete: {', '.join(missing)}.\n"
                "Capture or import all three measurements, average them, "
                "and calculate again.",
            )
            return False

        reflectance = _reflectance_ratio(
            averages["Sample"],
            averages["Baseline"],
            averages["Standard"],
        )
        try:
            self._model.set_reflectance(reflectance)
        except ValueError as exc:
            QMessageBox.warning(self, "Relative Reflectance Failed", str(exc))
            return False

        self._model.clear_interpolated_reflectance()
        self.interpolated_reflectance = None
        self._reflectance = reflectance.copy()
        parent = self.parent()
        store = getattr(parent, "_store_reflectance", None)
        if callable(store):
            store(self._mode_key, self._reflectance)
        clear_interp = getattr(parent, "_store_interpolated_reflectance", None)
        if callable(clear_interp):
            clear_interp(self._mode_key, None, None)
        finite = int(np.isfinite(reflectance).sum())
        self._summary.setText(
            f"Optical mode: {self._mode_display}  ·  "
            f"{self._model.rowCount()} raw rows  ·  "
            f"Relative Reflectance computed ({finite} finite points)"
        )
        return True

    def _interpolate_reflectance(self) -> None:
        """Resample relative reflectance using only the Reflectance group grid."""
        if self._reflectance is None or not np.any(np.isfinite(self._reflectance)):
            if not self._calculate_reflectance():
                return
        assert self._reflectance is not None

        try:
            grid = self._current_reflectance_grid()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Grid", str(exc))
            return

        method_key = self._current_reflectance_method()
        try:
            resampled = _interpolate_reflectance_series(
                self._model.wavelengths(),
                self._reflectance,
                grid,
                method_key,
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Reflectance Required", str(exc))
            return
        except Exception as exc:
            logger.exception("Reflectance interpolation failed")
            QMessageBox.critical(self, "Interpolation Failed", str(exc))
            return

        try:
            self._model.set_interpolated_reflectance(grid, resampled)
        except ValueError as exc:
            QMessageBox.warning(self, "Interpolation Failed", str(exc))
            return

        self.interpolated_reflectance = {
            "wavelengths": np.asarray(grid, dtype=np.float64).copy(),
            "values": np.asarray(resampled, dtype=np.float64).copy(),
        }
        parent = self.parent()
        store = getattr(parent, "_store_interpolated_reflectance", None)
        if callable(store):
            store(self._mode_key, grid, resampled)
        self._table.resizeColumnsToContents()
        self._summary.setText(
            f"Optical mode: {self._mode_display}  ·  "
            f"{self._model.rowCount()} raw rows  ·  "
            f"{grid.size} interpolated relative reflectance points "
            f"({grid[0]:.3f}–{grid[-1]:.3f} nm, "
            f"step {self._field_float(self._ref_step_wl_edit):.3f} nm, {method_key})"
        )

    def _export_processed_table(self) -> None:
        headers, rows = self._model.export_rows()
        # Every visible column is written, including Relative Reflectance,
        # Interpolated Relative Reflectance, imported silicon values, and
        # interpolated silicon n and k.
        col_count = len(headers)
        normalized_rows: list[list[str]] = []
        for row in rows:
            padded = [str(row[c]) if c < len(row) else "" for c in range(col_count)]
            normalized_rows.append(padded)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_token = _geometry_filename_token(self._mode_key)
        default_path = (
            Path.home() / f"spectrum_{mode_token}_processed_{stamp}.csv"
        )
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Export Processed Table",
            str(default_path),
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path_str:
            return

        path = Path(path_str)
        if path.suffix.lower() != ".csv":
            path = path.with_suffix(".csv")

        try:
            # utf-8-sig helps Excel detect Unicode; sep=, forces comma columns
            # even when the OS locale expects semicolon separators.
            with path.open("w", newline="", encoding="utf-8-sig") as csvfile:
                csvfile.write("sep=,\n")
                writer = csv.writer(
                    csvfile,
                    delimiter=",",
                    quotechar='"',
                    quoting=csv.QUOTE_MINIMAL,
                )
                writer.writerow(headers)
                writer.writerows(normalized_rows)
        except OSError as exc:
            QMessageBox.critical(self, "Export Failed", str(exc))
            return

        QMessageBox.information(
            self,
            "Export Complete",
            f"Saved processed table to:\n{path}",
        )

    def _refresh_slot_controls(self) -> None:
        self._sync_average_buttons(
            self._model.raw_trace("Baseline") is not None,
            self._model.raw_trace("Standard") is not None,
            self._model.raw_trace("Sample") is not None,
        )
        self._populate_interp_trace_combo(
            self._model.raw_trace("Baseline") is not None,
            self._model.raw_trace("Standard") is not None,
            self._model.raw_trace("Sample") is not None,
        )

    def _import_measurement_csv(self) -> None:
        """Import every measurement column and refresh this table."""
        path_str, _selected = QFileDialog.getOpenFileName(
            self,
            "Import CSV Data",
            str(Path.home()),
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path_str:
            return

        try:
            imported_wavelengths, columns = parse_unified_measurement_csv(Path(path_str))
        except MeasurementCsvError as exc:
            QMessageBox.warning(self, "Import Failed", str(exc))
            return

        reference = self._model.wavelengths()
        if not _wavelength_axes_match(imported_wavelengths, reference):
            QMessageBox.warning(
                self,
                "Wavelength Axis Mismatch",
                "The imported wavelength axis does not match the current "
                "measurement grid.\n"
                f"Imported points: {imported_wavelengths.size}\n"
                f"Current points: {reference.size}\n"
                "Import a file acquired on this wavelength axis "
                "(same length and wavelength values).",
            )
            return

        loaded: Optional[list[str]] = None
        if self._on_imported is not None:
            loaded = self._on_imported(imported_wavelengths, columns)
            if not loaded:
                return

        updates: dict[str, np.ndarray] = dict(columns.get(self._mode_key, {}))
        for slot_name, values in columns.get(None, {}).items():
            updates.setdefault(slot_name, values)
        try:
            self._model.apply_imported_slots(updates)
        except ValueError as exc:
            QMessageBox.warning(self, "Import Failed", str(exc))
            return

        self._reflectance = None
        self.interpolated_reflectance = None
        self._refresh_slot_controls()
        self._table.resizeColumnsToContents()
        summary = ", ".join(loaded) if loaded else ", ".join(updates)
        QMessageBox.information(
            self,
            "Import Complete",
            f"Loaded {imported_wavelengths.size} points into: {summary}.",
        )

    def _import_si_values(self) -> None:
        """Load silicon wavelength, n, and k into the table."""
        path_str, _selected = QFileDialog.getOpenFileName(
            self,
            "Import Si Values",
            str(Path.home()),
            "Excel/CSV Files (*.xlsx *.xls *.csv);;All Files (*)",
        )
        if not path_str:
            return
        try:
            wavelengths, refractive_index, extinction = parse_silicon_reference(
                Path(path_str)
            )
        except MeasurementCsvError as exc:
            QMessageBox.warning(self, "Import Failed", str(exc))
            return
        try:
            self._model.set_silicon_reference(
                wavelengths, refractive_index, extinction
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Import Failed", str(exc))
            return

        self.si_wavelength = np.asarray(wavelengths, dtype=np.float64).copy()
        self.si_n = np.asarray(refractive_index, dtype=np.float64).copy()
        self.si_k = np.asarray(extinction, dtype=np.float64).copy()
        self.si_interp_wavelength = None
        self.si_interp_n = None
        self.si_interp_k = None
        parent = self.parent()
        if parent is not None and hasattr(parent, "si_wavelength"):
            parent.si_wavelength = self.si_wavelength.copy()
            parent.si_n = self.si_n.copy()
            parent.si_k = self.si_k.copy()
            if hasattr(parent, "si_interp_wavelength"):
                parent.si_interp_wavelength = None
                parent.si_interp_n = None
                parent.si_interp_k = None
        self._table.resizeColumnsToContents()
        self._summary.setText(
            f"Optical mode: {self._mode_display}  ·  "
            f"{self._model.rowCount()} raw rows  ·  "
            f"{wavelengths.size} silicon reference points"
        )
        QMessageBox.information(
            self,
            "Import Complete",
            f"Loaded {wavelengths.size} silicon reference points into "
            "Si Wavelength (nm), Si n, and Si k.",
        )

    def _interpolate_silicon(self) -> None:
        """Resample imported silicon n and k onto the Silicon Calibration grid."""
        if (
            self.si_wavelength is None
            or self.si_n is None
            or self.si_k is None
            or self.si_wavelength.size < 2
        ):
            QMessageBox.warning(
                self,
                "Silicon Data Required",
                "Import Si values before interpolating n and k.",
            )
            return
        try:
            grid = self._current_si_grid()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Grid", str(exc))
            return
        method = self._current_si_method()
        try:
            interp_n = _interpolate_si_series(
                self.si_wavelength, self.si_n, grid, method
            )
            interp_k = _interpolate_si_series(
                self.si_wavelength, self.si_k, grid, method
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Interpolation Failed", str(exc))
            return
        except Exception as exc:
            logger.exception("Silicon interpolation failed")
            QMessageBox.critical(self, "Interpolation Failed", str(exc))
            return
        try:
            self._model.set_interpolated_silicon(grid, interp_n, interp_k)
        except ValueError as exc:
            QMessageBox.warning(self, "Interpolation Failed", str(exc))
            return

        self.si_interp_wavelength = np.asarray(grid, dtype=np.float64).copy()
        self.si_interp_n = np.asarray(interp_n, dtype=np.float64).copy()
        self.si_interp_k = np.asarray(interp_k, dtype=np.float64).copy()
        parent = self.parent()
        if parent is not None and hasattr(parent, "si_interp_wavelength"):
            parent.si_interp_wavelength = self.si_interp_wavelength.copy()
            parent.si_interp_n = self.si_interp_n.copy()
            parent.si_interp_k = self.si_interp_k.copy()
        self._table.resizeColumnsToContents()
        self._summary.setText(
            f"Optical mode: {self._mode_display}  ·  "
            f"{self._model.rowCount()} raw rows  ·  "
            f"{grid.size} interpolated silicon points "
            f"({grid[0]:.3f}–{grid[-1]:.3f} nm, "
            f"step {self._field_float(self._si_step_wl_edit):.3f} nm, {method})"
        )


# ===========================================================================
# ADDED FEATURE: progress pop-up for post-processing calculations
# ---------------------------------------------------------------------------
# Everything in this block is new. PostProcessDialog above is not modified.
#
# There are two ways to run work inside the pop-up:
#
# 1. Existing buttons  -> PostProcessDialogWithProgress._run_with_progress
#    The original handler methods are reused unchanged. Because they update
#    widgets and show message boxes, they must run on the GUI thread and they
#    cannot report how far along they are. So the bar fills on a short timer,
#    then shows "Working…" while the handler actually runs, and "Completed"
#    reports the real calculation time. (Measured worst cases at the most
#    extreme settings the UI allows: under 0.5 s with SciPy installed.)
#
# 2. New long calculations  -> run_with_real_progress(...) /
#                              PostProcessDialogWithProgress.run_long_task(...)
#    The work runs on a background thread and reports real progress, so the
#    bar shows the true percentage, elapsed and remaining time, Cancel works,
#    and the window (including the live plot) never freezes. Use this for
#    anything that can take more than about a second, e.g. the planned
#    d1/n1 and n1(λ)/k1(λ) solvers.
# ===========================================================================

import threading  # noqa: E402  (imports used only by this block)
import time  # noqa: E402
from functools import partial  # noqa: E402

from PyQt6.QtCore import QElapsedTimer, QObject, pyqtSignal, pyqtSlot  # noqa: E402
from PyQt6.QtWidgets import QProgressBar  # noqa: E402

# Timed fill used for the existing (fast, GUI-thread) buttons.
_PROGRESS_FILL_MS: int = 800
# How long "Completed" / "Cancelled" stays visible before the pop-up closes.
_PROGRESS_DONE_HOLD_MS: int = 600
_PROGRESS_TICK_MS: int = 15
# Background tasks: minimum time between progress updates sent to the GUI,
# so a task that reports every loop iteration cannot flood the event queue.
_PROGRESS_REPORT_MIN_INTERVAL_S: float = 0.03


class TaskCancelled(Exception):
    """Raised inside a background task's ``report()`` after Cancel is pressed."""


class _BackgroundTask(QObject):
    """Runs ``work(report)`` on a worker thread and relays progress to the GUI."""

    progress = pyqtSignal(float)
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(object)
    cancelled = pyqtSignal()

    def __init__(self, work: Callable[[Callable[[float], None]], object]) -> None:
        super().__init__()
        self._work = work
        self._cancel_requested = threading.Event()
        self._last_emit = 0.0
        self._last_value = -1.0

    def request_cancel(self) -> None:
        self._cancel_requested.set()

    def report(self, fraction: float) -> None:
        """Called by the work function (worker thread) with 0.0 … 1.0."""
        if self._cancel_requested.is_set():
            raise TaskCancelled()
        value = min(1.0, max(0.0, float(fraction)))
        now = time.monotonic()
        if value >= 1.0 or (
            value != self._last_value
            and now - self._last_emit >= _PROGRESS_REPORT_MIN_INTERVAL_S
        ):
            self._last_emit = now
            self._last_value = value
            self.progress.emit(value)

    @pyqtSlot()
    def run(self) -> None:
        try:
            result = self._work(self.report)
        except TaskCancelled:
            self.cancelled.emit()
        except Exception as exc:  # reported to the user by the pop-up
            logger.exception("Background calculation failed")
            self.failed.emit(exc)
        else:
            self.succeeded.emit(result)


class ProgressPopup(QDialog):
    """Small modal pop-up with a 0 → 100 % bar.

    ``run_handler()`` is used for the existing GUI-thread buttons (timed fill,
    then the handler runs). ``run_task()`` runs a background calculation with
    real progress, elapsed / remaining time and a Cancel button.
    After closing, ``outcome`` is one of "completed", "unchanged",
    "cancelled" or "failed".
    """

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(
            parent, Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint
        )
        self._title = title
        self._mode: Optional[str] = None
        self._finished = False
        self._action: Optional[Callable[[], object]] = None
        self._did_change: Optional[Callable[[], bool]] = None
        self._on_done: Optional[Callable[[object], object]] = None
        self._task: Optional[_BackgroundTask] = None
        self._thread: Optional[QThread] = None
        self._fraction = 0.0
        self._cancel_requested = False
        self.completed = False
        self.outcome: Optional[str] = None
        self.result: object = None
        self.error: Optional[BaseException] = None

        self.setObjectName("progressPopup")
        self.setModal(True)
        self.setFixedWidth(380)
        self.setStyleSheet(
            """
            QDialog#progressPopup {
                background-color: #1a1d23;
            }
            QFrame#progressCard {
                background-color: #1a1d23;
                border: 1px solid #00bcd4;
            }
            QLabel#progressTitle {
                color: #e8eaed;
                font-weight: 600;
                background: transparent;
                border: none;
            }
            QLabel#progressStatus {
                color: #9aa3b2;
                background: transparent;
                border: none;
            }
            QProgressBar {
                background-color: #12141a;
                border: 1px solid #3d4450;
                border-radius: 6px;
                color: #e8eaed;
                text-align: center;
                min-height: 20px;
            }
            QProgressBar::chunk {
                background-color: #00bcd4;
                border-radius: 5px;
            }
            QProgressBar[done="true"]::chunk {
                background-color: #4caf50;
            }
            QPushButton#progressCancel {
                background-color: #2a2f3a;
                border: 1px solid #3d4450;
                border-radius: 6px;
                padding: 2px 14px;
                min-height: 22px;
                color: #e8eaed;
            }
            QPushButton#progressCancel:hover:!disabled {
                background-color: #353b48;
            }
            QPushButton#progressCancel:disabled {
                color: #6d7480;
            }
            """
        )

        card = QFrame(self)
        card.setObjectName("progressCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(16, 14, 16, 14)
        card_layout.setSpacing(8)

        self._title_label = QLabel(title, card)
        self._title_label.setObjectName("progressTitle")
        self._bar = QProgressBar(card)
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setProperty("done", "false")
        self._status_label = QLabel("Processing…", card)
        self._status_label.setObjectName("progressStatus")

        self._cancel_btn = QPushButton("Cancel", card)
        self._cancel_btn.setObjectName("progressCancel")
        self._cancel_btn.setAutoDefault(False)
        self._cancel_btn.setDefault(False)
        self._cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._cancel_btn.clicked.connect(self._request_cancel)
        self._cancel_btn.hide()

        status_row = QHBoxLayout()
        status_row.setSpacing(8)
        status_row.addWidget(self._status_label, stretch=1)
        status_row.addWidget(self._cancel_btn)

        card_layout.addWidget(self._title_label)
        card_layout.addWidget(self._bar)
        card_layout.addLayout(status_row)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(card)

        self._clock = QElapsedTimer()
        self._fill_timer = QTimer(self)
        self._fill_timer.setInterval(_PROGRESS_TICK_MS)
        self._fill_timer.timeout.connect(self._fill_tick)
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(200)
        self._status_timer.timeout.connect(self._refresh_task_status)
        self._close_timer = QTimer(self)
        self._close_timer.setSingleShot(True)
        self._close_timer.timeout.connect(self.accept)

    # ------------------------------------------------------------------
    # Mode 1: existing GUI-thread handlers
    # ------------------------------------------------------------------

    def run_handler(
        self, action: Callable[[], object], did_change: Callable[[], bool]
    ) -> Optional[str]:
        """Timed fill, then run ``action`` on the GUI thread. Blocks until closed."""
        self._mode = "handler"
        self._action = action
        self._did_change = did_change
        self._clock.start()
        self._fill_timer.start()
        self.exec()
        return self.outcome

    def _fill_tick(self) -> None:
        percent = min(100, int(self._clock.elapsed() * 100 / _PROGRESS_FILL_MS))
        self._bar.setValue(percent)
        if percent >= 100:
            self._fill_timer.stop()
            self._run_handler_now()

    def _run_handler_now(self) -> None:
        assert self._action is not None and self._did_change is not None
        # Paint "Working…" immediately so it stays visible however long the
        # handler takes (the handler blocks the GUI thread while it runs).
        self._status_label.setText("Working…")
        self.repaint()
        started = time.perf_counter()
        try:
            self._action()
        except BaseException:
            # Close the pop-up, then let the error propagate exactly as it
            # would have from the original button.
            self._finished = True
            self.accept()
            raise
        seconds = time.perf_counter() - started
        self._finished = True

        if not self._did_change():
            # The handler showed its own warning or did nothing.
            self.outcome = "unchanged"
            self.accept()
            return
        self._show_completed(f"Completed ✓  (calculation took {seconds:.2f} s)")

    # ------------------------------------------------------------------
    # Mode 2: background calculation with real progress
    # ------------------------------------------------------------------

    def run_task(
        self,
        work: Callable[[Callable[[float], None]], object],
        on_done: Optional[Callable[[object], object]] = None,
    ) -> Optional[str]:
        """Run ``work(report)`` on a background thread. Blocks until closed."""
        self._mode = "task"
        self._on_done = on_done
        self._cancel_btn.show()
        self._status_label.setText("Starting…")

        self._task = _BackgroundTask(work)
        self._thread = QThread(self)
        self._task.moveToThread(self._thread)
        self._thread.started.connect(self._task.run)
        self._thread.finished.connect(self._task.deleteLater)
        self._task.progress.connect(self._on_task_progress)
        self._task.succeeded.connect(self._on_task_succeeded)
        self._task.failed.connect(self._on_task_failed)
        self._task.cancelled.connect(self._on_task_cancelled)

        self._clock.start()
        self._status_timer.start()
        self._thread.start()
        self.exec()
        self._status_timer.stop()
        self._thread.quit()
        self._thread.wait()
        self._task = None
        return self.outcome

    def _on_task_progress(self, fraction: float) -> None:
        if self._finished:
            return
        self._fraction = fraction
        self._bar.setValue(int(fraction * 100))
        self._refresh_task_status()

    def _refresh_task_status(self) -> None:
        if self._finished or self._mode != "task":
            return
        if self._cancel_requested:
            self._status_label.setText("Cancelling…")
            return
        elapsed = self._clock.elapsed() / 1000.0
        text = f"{elapsed:.1f} s elapsed"
        if 0.02 <= self._fraction < 1.0:
            remaining = elapsed * (1.0 - self._fraction) / self._fraction
            if remaining < 1.0:
                text += "  ·  under 1 s left"
            else:
                text += f"  ·  about {remaining:.0f} s left"
        self._status_label.setText(text)

    def _on_task_succeeded(self, result: object) -> None:
        self._finished = True
        self._cancel_btn.hide()
        seconds = self._clock.elapsed() / 1000.0
        self.result = result
        if self._on_done is not None:
            try:
                self._on_done(result)
            except Exception as exc:
                logger.exception("Displaying calculation results failed")
                self.outcome = "failed"
                self.error = exc
                self.accept()
                return
        self._show_completed(f"Completed ✓  ({seconds:.2f} s)")

    def _on_task_failed(self, exc: object) -> None:
        self._finished = True
        self.outcome = "failed"
        self.error = exc if isinstance(exc, BaseException) else RuntimeError(str(exc))
        self.accept()

    def _on_task_cancelled(self) -> None:
        self._finished = True
        self.outcome = "cancelled"
        self._cancel_btn.hide()
        self._status_label.setText("Cancelled")
        self._close_timer.start(_PROGRESS_DONE_HOLD_MS)

    def _request_cancel(self) -> None:
        if self._mode != "task" or self._finished or self._task is None:
            return
        self._cancel_requested = True
        self._cancel_btn.setEnabled(False)
        self._task.request_cancel()
        self._refresh_task_status()

    # ------------------------------------------------------------------
    # Shared
    # ------------------------------------------------------------------

    def _show_completed(self, text: str) -> None:
        self.completed = True
        self.outcome = "completed"
        self._bar.setValue(100)
        self._status_label.setText(text)
        self._bar.setProperty("done", "true")
        self._bar.style().unpolish(self._bar)
        self._bar.style().polish(self._bar)
        self._close_timer.start(_PROGRESS_DONE_HOLD_MS)

    def reject(self) -> None:  # Esc / Alt+F4
        if self._finished:
            super().reject()
        elif self._mode == "task":
            self._request_cancel()  # Esc acts like Cancel for long tasks
        # Existing handlers: ignore until the handler has run.


def run_with_real_progress(
    parent: QWidget,
    title: str,
    work: Callable[[Callable[[float], None]], object],
    on_done: Optional[Callable[[object], object]] = None,
) -> Optional[str]:
    """Run a slow calculation in a progress pop-up with real progress.

    ``work(report)`` runs on a background thread. It must only do
    calculations (NumPy, SciPy, plain Python) and must NOT touch any widget,
    message box or table model. Call ``report(fraction)`` regularly with a
    value from 0.0 to 1.0; that drives the bar and lets Cancel stop the work.
    Whatever ``work`` returns is passed to ``on_done(result)``, which runs on
    the GUI thread afterwards: update the table / labels there.

    Returns "completed", "cancelled" or "failed". A failure has already been
    shown to the user in an error box when this returns.

    Example (e.g. a future d1 / n1 solver)::

        def work(report):
            best = None
            for i, d1 in enumerate(d1_values):
                ...                      # heavy NumPy maths only
                report((i + 1) / len(d1_values))
            return best

        def show(result):
            ...                          # update the table here

        run_with_real_progress(self, "Solving d1 and n1", work, show)
    """
    popup = ProgressPopup(title, parent=parent)
    try:
        outcome = popup.run_task(work, on_done)
        if outcome == "failed":
            QMessageBox.critical(parent, f"{title} failed", str(popup.error))
        return outcome
    finally:
        popup.deleteLater()


class PostProcessDialogWithProgress(PostProcessDialog):
    """PostProcessDialog whose calculation buttons run inside ProgressPopup.

    The original handler methods are called unchanged. Only the signal
    connections of these buttons/menu items are swapped for wrappers:
    Average trace (Baseline/Standard/Sample), Average All,
    Interpolate (Baseline/Standard/Sample), Interpolate all,
    Relative Reflectance, Interpolate Relative Reflectance,
    Interpolate Si (n and k).
    Import, Export, Import Si Values and Back keep their original wiring.

    New long-running features should call ``self.run_long_task(...)``.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._attach_progress_popups()

    def run_long_task(
        self,
        title: str,
        work: Callable[[Callable[[float], None]], object],
        on_done: Optional[Callable[[object], object]] = None,
    ) -> Optional[str]:
        """Real-progress pop-up for slow calculations (see run_with_real_progress)."""
        return run_with_real_progress(self, title, work, on_done)

    def _attach_progress_popups(self) -> None:
        wiring = (
            (
                self._avg_baseline_action.triggered,
                "Averaging Baseline",
                lambda: self._average_trace("Baseline"),
            ),
            (
                self._avg_standard_action.triggered,
                "Averaging Standard",
                lambda: self._average_trace("Standard"),
            ),
            (
                self._avg_sample_action.triggered,
                "Averaging Sample",
                lambda: self._average_trace("Sample"),
            ),
            (
                self._avg_all_btn.clicked,
                "Averaging all traces",
                self._average_all_traces,
            ),
            (
                self._interp_baseline_action.triggered,
                "Interpolating Baseline",
                lambda: self._interpolate_selected("Baseline"),
            ),
            (
                self._interp_standard_action.triggered,
                "Interpolating Standard",
                lambda: self._interpolate_selected("Standard"),
            ),
            (
                self._interp_sample_action.triggered,
                "Interpolating Sample",
                lambda: self._interpolate_selected("Sample"),
            ),
            (
                self._interpolate_all_btn.clicked,
                "Interpolating all traces",
                self._interpolate_all,
            ),
            (
                self._reflectance_btn.clicked,
                "Calculating Relative Reflectance",
                self._calculate_reflectance,
            ),
            (
                self._interp_reflectance_btn.clicked,
                "Interpolating Relative Reflectance",
                self._interpolate_reflectance,
            ),
            (
                self._interp_si_btn.clicked,
                "Interpolating Si (n and k)",
                self._interpolate_silicon,
            ),
        )
        for signal, title, action in wiring:
            signal.disconnect()  # remove the original direct connection
            signal.connect(partial(self._run_with_progress, title, action))

    def _table_snapshot(self) -> tuple[int, list[tuple[str, object, str]]]:
        """Row count plus every column (header, array object, kind), read-only."""
        return (self._model.rowCount(), list(self._model._columns))

    @staticmethod
    def _snapshots_differ(
        before: tuple[int, list[tuple[str, object, str]]],
        after: tuple[int, list[tuple[str, object, str]]],
    ) -> bool:
        if before[0] != after[0] or len(before[1]) != len(after[1]):
            return True
        return any(
            h1 != h2 or k1 != k2 or s1 is not s2
            for (h1, s1, k1), (h2, s2, k2) in zip(before[1], after[1])
        )

    def _run_with_progress(
        self, title: str, action: Callable[[], object], *_signal_args: object
    ) -> None:
        before = self._table_snapshot()
        popup = ProgressPopup(title, parent=self)
        try:
            popup.run_handler(
                action,
                lambda: self._snapshots_differ(before, self._table_snapshot()),
            )
        finally:
            popup.deleteLater()


# ===========================================================================
# END OF ADDED FEATURE
# ===========================================================================


class SpectrometerApp(QMainWindow):
    """Main window: live spectrum, multi-geometry capture slots, and export."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Multi-Function Optical Analyzer")
        self.resize(1080, 720)

        self._active_geometry: str = OPTICAL_MODE_KEYS[0]  # R0
        self._active_mode: CaptureMode = CaptureMode.BASELINE
        self._device_name: str = "Connecting…"
        self._acquisition_started: bool = False
        self._integration_syncing: bool = False
        self._latest_wavelengths: Optional[np.ndarray] = None
        self._latest_intensities: Optional[np.ndarray] = None

        # Hierarchical storage: optical mode key → capture slot → intensities
        self.spectra_data: dict[str, dict[str, Optional[np.ndarray]]] = (
            _build_spectra_data()
        )
        self.reflectance_data: dict[str, Optional[np.ndarray]] = {
            key: None for key in OPTICAL_MODE_KEYS
        }
        self.interpolated_reflectance: dict[str, Optional[dict[str, np.ndarray]]] = {
            key: None for key in OPTICAL_MODE_KEYS
        }
        self.si_wavelength: Optional[np.ndarray] = None
        self.si_n: Optional[np.ndarray] = None
        self.si_k: Optional[np.ndarray] = None
        self.si_interp_wavelength: Optional[np.ndarray] = None
        self.si_interp_n: Optional[np.ndarray] = None
        self.si_interp_k: Optional[np.ndarray] = None
        # Wavelength axis shared across modes (detector axis is fixed).
        self._wavelengths: Optional[np.ndarray] = None

        self._build_ui()
        self._start_worker()
        self._refresh_capture_indicators()
        self._refresh_status()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 8)
        root.setSpacing(10)

        root.addLayout(self._build_geometry_row())
        root.addLayout(self._build_capture_row())
        root.addWidget(self._build_plot(), stretch=1)
        root.addLayout(self._build_controls_row())

        status = QStatusBar(self)
        self.setStatusBar(status)
        self._apply_dark_style()

    def _build_geometry_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)

        label = QLabel("Measurment Mode:")
        label.setStyleSheet("font-weight: 600;")
        row.addWidget(label)

        self._geometry_group = QButtonGroup(self)
        self._geometry_group.setExclusive(True)
        self._geometry_buttons: dict[str, QPushButton] = {}

        for key, display in OPTICAL_MODES:
            btn = QPushButton(display)
            btn.setObjectName("geometryModeBtn")
            btn.setCheckable(True)
            btn.setMinimumHeight(36)
            btn.setMinimumWidth(72)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setToolTip(f"Switch to optical mode {display} ({key})")
            self._geometry_group.addButton(btn)
            self._geometry_buttons[key] = btn
            row.addWidget(btn)
            btn.clicked.connect(
                lambda _checked=False, k=key: self._on_geometry_changed(k)
            )

        self._geometry_buttons[self._active_geometry].setChecked(True)

        self._geometry_hint = QLabel("")
        self._geometry_hint.setStyleSheet("color: #9aa3b2;")
        row.addWidget(self._geometry_hint, stretch=1)

        return row

    def _build_capture_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)

        label = QLabel("Optical Data:")
        label.setStyleSheet("font-weight: 600;")
        row.addWidget(label)

        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self._mode_buttons: dict[CaptureMode, QPushButton] = {}
        for mode in CaptureMode:
            btn = QPushButton(mode.value)
            btn.setObjectName("captureSlotBtn")
            btn.setCheckable(True)
            btn.setMinimumHeight(32)
            btn.setMinimumWidth(110)
            btn.setProperty("captured", "false")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self._mode_group.addButton(btn)
            self._mode_buttons[mode] = btn
            row.addWidget(btn)
            btn.clicked.connect(lambda _checked=False, m=mode: self._set_active_mode(m))

        self._mode_buttons[CaptureMode.BASELINE].setChecked(True)

        row.addStretch(1)

        self._capture_btn = QPushButton("Capture Baseline")
        self._capture_btn.setMinimumHeight(36)
        self._capture_btn.setMinimumWidth(200)
        self._capture_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._capture_btn.clicked.connect(self._capture_active)
        row.addWidget(self._capture_btn)

        return row

    def _build_plot(self) -> pg.PlotWidget:
        pg.setConfigOptions(antialias=True, foreground="w", background="#1a1d23")

        self._plot = pg.PlotWidget()
        # Disable SI auto-prefixing so ticks stay in raw nm / counts
        # (e.g. 65000, not "65 kcounts"; 650 nm, not "knm").
        self._plot.getAxis("bottom").enableAutoSIPrefix(False)
        self._plot.getAxis("left").enableAutoSIPrefix(False)
        self._plot.setLabel("bottom", "Wavelength (nm)")
        self._plot.setLabel("left", "Intensity (counts)")
        self._plot.showGrid(x=True, y=True, alpha=0.25)
        self._plot.setYRange(0, 65535)
        self._legend = self._plot.addLegend(offset=(10, 10))
        self._update_plot_title()

        self._capture_curves: dict[CaptureMode, pg.PlotDataItem] = {}
        for mode in CaptureMode:
            curve = self._plot.plot(
                pen=_CAPTURE_PENS[mode],
                name=self._legend_name(mode),
            )
            curve.hide()
            self._capture_curves[mode] = curve

        self._live_curve = self._plot.plot(
            pen=_LIVE_PEN,
            name=self._legend_name_live(),
        )
        return self._plot

    def _build_controls_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(12)

        integ_label = QLabel("Integration Time:")
        row.addWidget(integ_label)

        # Milliseconds and microseconds compose one SeaBreeze integration time:
        # total_micros = (ms * 1000) + us.
        self._integration_ms_spin = QSpinBox()
        self._integration_ms_spin.setRange(_MIN_INTEGRATION_US // 1000, _MAX_INTEGRATION_MS)
        self._integration_ms_spin.setValue(100)
        self._integration_ms_spin.setSingleStep(1)
        self._integration_ms_spin.setMinimumWidth(80)
        self._integration_ms_spin.setToolTip(
            "Integration time, milliseconds. Combined with microseconds "
            f"and applied as total µs (minimum {_MIN_INTEGRATION_US} µs)."
        )
        self._integration_ms_spin.valueChanged.connect(self._on_integration_changed)
        row.addWidget(self._integration_ms_spin)

        row.addWidget(QLabel("ms"))

        self._integration_us_spin = QSpinBox()
        self._integration_us_spin.setRange(0, 999)
        self._integration_us_spin.setValue(0)
        self._integration_us_spin.setSingleStep(1)
        self._integration_us_spin.setMinimumWidth(80)
        self._integration_us_spin.setToolTip(
            "Integration time, microseconds (0–999), added to the millisecond field."
        )
        self._integration_us_spin.valueChanged.connect(self._on_integration_changed)
        row.addWidget(self._integration_us_spin)

        row.addWidget(QLabel("µs"))

        scans_label = QLabel("Scans to Average:")
        row.addWidget(scans_label)

        self._scans_spin = QSpinBox()
        self._scans_spin.setRange(1, 100)
        self._scans_spin.setValue(1)
        self._scans_spin.setSingleStep(1)
        self._scans_spin.setMinimumWidth(90)
        self._scans_spin.setToolTip(
            "Pixel-wise mean over the last N scans (1 = raw single scan)"
        )
        # Connected to the worker in _start_worker (cross-thread QueuedConnection).
        row.addWidget(self._scans_spin)

        row.addStretch(1)

        self._reset_current_btn = QPushButton("Reset Current Mode")
        self._reset_current_btn.setMinimumHeight(36)
        self._reset_current_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._reset_current_btn.setToolTip(
            "Clear Baseline / Standard / Sample for the active optical mode only"
        )
        self._reset_current_btn.clicked.connect(self._reset_current_geometry)
        row.addWidget(self._reset_current_btn)

        self._reset_all_btn = QPushButton("Reset All")
        self._reset_all_btn.setMinimumHeight(36)
        self._reset_all_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._reset_all_btn.setToolTip("Clear captured traces for every optical mode")
        self._reset_all_btn.clicked.connect(self._reset_all_geometries)
        row.addWidget(self._reset_all_btn)

        self._save_image_btn = QPushButton("Save Graph Image")
        self._save_image_btn.setObjectName("actionBtn")
        self._save_image_btn.setMinimumHeight(40)
        self._save_image_btn.setMinimumWidth(190)
        self._save_image_btn.setEnabled(False)
        self._save_image_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._save_image_btn.setToolTip(
            "Save the current optical mode's plot (live + overlays) as an image"
        )
        camera_icon = QIcon.fromTheme("camera-photo")
        if camera_icon.isNull():
            camera_icon = self.style().standardIcon(
                QStyle.StandardPixmap.SP_DesktopIcon
            )
        self._save_image_btn.setIcon(camera_icon)
        self._save_image_btn.clicked.connect(self._save_plot_image)
        row.addWidget(self._save_image_btn)

        self._import_csv_btn = QPushButton("Import CSV Data")
        self._import_csv_btn.setObjectName("actionBtn")
        self._import_csv_btn.setMinimumHeight(40)
        self._import_csv_btn.setMinimumWidth(170)
        self._import_csv_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._import_csv_btn.setToolTip(
            "Load Baseline, Standard, and Sample for the active mode from a CSV file"
        )
        self._import_csv_btn.clicked.connect(self._import_csv_data)
        row.addWidget(self._import_csv_btn)

        self._export_btn = QPushButton("Export Data (CSV)")
        self._export_btn.setObjectName("actionBtn")
        self._export_btn.setMinimumHeight(40)
        self._export_btn.setMinimumWidth(180)
        self._export_btn.setEnabled(False)
        self._export_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._export_btn.setToolTip(
            "Export Baseline / Standard / Sample for the active optical mode only"
        )
        self._export_btn.clicked.connect(self._export_csv)
        row.addWidget(self._export_btn)

        self._post_process_btn = QPushButton("Post Process")
        self._post_process_btn.setObjectName("actionBtn")
        self._post_process_btn.setMinimumHeight(40)
        self._post_process_btn.setMinimumWidth(150)
        self._post_process_btn.setEnabled(False)
        self._post_process_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._post_process_btn.setToolTip(
            "Open a table viewer of captured spectra for the active optical mode"
        )
        self._post_process_btn.clicked.connect(self._open_post_process)
        row.addWidget(self._post_process_btn)

        return row

    def _apply_dark_style(self) -> None:
        # Unified teal accent used for checked / captured / action hover states.
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background-color: #12141a;
                color: #e8eaed;
                font-size: 13px;
            }
            QPushButton {
                background-color: #2a2f3a;
                border: 1px solid #3d4450;
                border-radius: 6px;
                padding: 6px 14px;
                color: #e8eaed;
            }
            QPushButton:hover {
                background-color: #123e4f;
                border: 1px solid #00bcd4;
                color: #e8fbff;
            }
            QPushButton:pressed {
                background-color: #0e3140;
                border: 1px solid #00bcd4;
            }
            QPushButton:checked {
                background-color: #123e4f;
                border: 2px solid #00bcd4;
                color: #e8fbff;
            }
            QPushButton:disabled {
                color: #777;
                background-color: #1e2229;
                border: 1px solid #2a2f3a;
            }

            /* Measurement mode buttons (R₀ / R₆₀ / T) */
            QPushButton#geometryModeBtn {
                background-color: #1a1e26;
                border: 1px solid #3a4150;
                color: #8b93a3;
                font-size: 16px;
                font-weight: 600;
                min-width: 72px;
                padding: 8px 18px;
            }
            QPushButton#geometryModeBtn:hover {
                background-color: #123e4f;
                color: #e8fbff;
                border: 1px solid #00bcd4;
            }
            QPushButton#geometryModeBtn:checked {
                background-color: #123e4f;
                color: #e8fbff;
                border: 2px solid #00bcd4;
                padding: 7px 17px;
            }

            /* Optical data slot buttons (Baseline / Standard / Sample) */
            QPushButton#captureSlotBtn {
                background-color: #1a1e26;
                border: 1px solid #3a4150;
                color: #8b93a3;
                font-weight: 600;
                min-width: 110px;
                padding: 6px 14px;
            }
            QPushButton#captureSlotBtn:hover {
                background-color: #123e4f;
                color: #e8fbff;
                border: 1px solid #00bcd4;
            }
            QPushButton#captureSlotBtn:checked {
                background-color: #123e4f;
                color: #e8fbff;
                border: 2px solid #00bcd4;
            }
            QPushButton#captureSlotBtn[captured="true"] {
                border: 2px solid #00bcd4;
                color: #e8fbff;
            }
            QPushButton#captureSlotBtn[captured="true"]:checked {
                background-color: #123e4f;
                border: 2px solid #00bcd4;
                color: #e8fbff;
            }

            /* Bottom action buttons — same accent on hover/pressed when enabled */
            QPushButton#actionBtn {
                background-color: #2a2f3a;
                border: 1px solid #3d4450;
                color: #e8eaed;
                font-weight: 600;
                padding: 8px 18px;
            }
            QPushButton#actionBtn:hover:!disabled {
                background-color: #123e4f;
                border: 1px solid #00bcd4;
                color: #e8fbff;
            }
            QPushButton#actionBtn:pressed:!disabled {
                background-color: #0e3140;
                border: 1px solid #00bcd4;
            }
            QPushButton#actionBtn:disabled {
                color: #777;
                background-color: #1e2229;
                border: 1px solid #2a2f3a;
            }

            QSpinBox {
                background-color: #1e2229;
                border: 1px solid #3d4450;
                border-radius: 4px;
                padding: 4px 8px;
                min-height: 24px;
            }
            QStatusBar {
                background-color: #0e1014;
                color: #aab0ba;
            }
            """
        )
        self._update_export_enabled()
        self._update_image_save_enabled()
        self._update_post_process_enabled()

    # ------------------------------------------------------------------
    # Worker lifecycle
    # ------------------------------------------------------------------

    def _start_worker(self) -> None:
        self._thread = QThread(self)
        self._worker = SpectrometerWorker()
        self._worker.moveToThread(self._thread)

        self._worker.data_acquired.connect(self._on_data_acquired)
        self._worker.status_message.connect(self._on_status_message)
        self._worker.device_error.connect(self._on_device_error)

        # Cross-thread QueuedConnection: spinbox → worker averaging slot.
        self._scans_spin.valueChanged.connect(self._worker.set_scans_to_average)

        self._thread.started.connect(self._worker.initialize)
        self._thread.start()

    def closeEvent(self, event: QCloseEvent) -> None:
        self._shutdown_worker()
        event.accept()

    def _shutdown_worker(self) -> None:
        if not hasattr(self, "_worker") or not hasattr(self, "_thread"):
            return

        if self._thread.isRunning():
            QMetaObject.invokeMethod(
                self._worker,
                "stop_acquisition",
                Qt.ConnectionType.BlockingQueuedConnection,
            )
            self._thread.quit()
            if not self._thread.wait(5000):
                logger.warning("Worker thread did not finish in time; terminating")
                self._thread.terminate()
                self._thread.wait(1000)

    # ------------------------------------------------------------------
    # Geometry / capture context
    # ------------------------------------------------------------------

    def _current_slot(self) -> dict[str, Optional[np.ndarray]]:
        return self.spectra_data[self._active_geometry]

    def _active_display(self) -> str:
        return _display_mode(self._active_geometry)

    def _legend_name(self, mode: CaptureMode) -> str:
        return f"{self._active_display()} {mode.value}"

    def _legend_name_live(self) -> str:
        return f"{self._active_display()} Live"

    def _update_plot_title(self) -> None:
        self._plot.setTitle(
            f"Spectrum — {_display_mode(self._active_geometry)}",
            color="#c5cad4",
            size="12pt",
        )

    def _on_geometry_changed(self, geometry_key: str) -> None:
        if geometry_key not in self.spectra_data:
            return
        if geometry_key == self._active_geometry:
            self._geometry_buttons[geometry_key].setChecked(True)
            return

        self._active_geometry = geometry_key
        self._geometry_buttons[geometry_key].setChecked(True)
        self._update_plot_title()
        self._refresh_plot_for_geometry()
        self._refresh_capture_indicators()
        self._update_export_enabled()
        self._update_image_save_enabled()
        self._update_post_process_enabled()
        self._refresh_status()
        self.statusBar().showMessage(
            f"Switched to {_display_mode(geometry_key)} "
            f"(other modes preserved)",
            3000,
        )

    def _set_active_mode(self, mode: CaptureMode) -> None:
        self._active_mode = mode
        self._mode_buttons[mode].setChecked(True)
        self._refresh_capture_indicators()
        self._refresh_status()

    def _refresh_plot_for_geometry(self) -> None:
        """Show live + captures belonging only to the active optical mode."""
        slot = self._current_slot()
        wavelengths = self._wavelengths

        for mode in CaptureMode:
            curve = self._capture_curves[mode]
            if self._legend is not None:
                self._legend.removeItem(curve)
            curve.opts["name"] = self._legend_name(mode)

            data = slot[mode.value]
            if data is not None and wavelengths is not None:
                curve.setData(wavelengths, data)
                curve.show()
                if self._legend is not None:
                    self._legend.addItem(curve, curve.name())
            else:
                curve.clear()
                curve.hide()

        if self._legend is not None:
            self._legend.removeItem(self._live_curve)
        self._live_curve.opts["name"] = self._legend_name_live()
        if self._legend is not None:
            self._legend.addItem(self._live_curve, self._live_curve.name())

        if (
            self._latest_wavelengths is not None
            and self._latest_intensities is not None
        ):
            self._live_curve.setData(
                self._latest_wavelengths, self._latest_intensities
            )

    def _refresh_capture_indicators(self) -> None:
        """Update mode buttons / capture CTA for the active optical mode."""
        slot = self._current_slot()
        display = self._active_display()
        for mode in CaptureMode:
            recorded = slot[mode.value] is not None
            mark = " ✓" if recorded else ""
            btn = self._mode_buttons[mode]
            btn.setText(f"{mode.value}{mark}")
            # Drive accent via QSS dynamic property (no per-button color overrides).
            btn.setProperty("captured", "true" if recorded else "false")
            style = btn.style()
            style.unpolish(btn)
            style.polish(btn)
            btn.update()

        mode = self._active_mode
        already = slot[mode.value] is not None
        verb = "Recapture" if already else "Capture"
        self._capture_btn.setText(f"{verb} {mode.value} ({display})")

        filled = sum(1 for label in CAPTURE_LABELS if slot[label] is not None)
        self._geometry_hint.setText(
            f"Active context: {display} — "
            f"{filled}/{len(CAPTURE_LABELS)} traces captured"
        )
    def _capture_active(self) -> None:
        if self._latest_wavelengths is None or self._latest_intensities is None:
            QMessageBox.warning(self, "No Data", "No live spectrum available yet.")
            return

        self._wavelengths = self._latest_wavelengths.copy()
        mode = self._active_mode
        slot = self._current_slot()
        slot[mode.value] = self._latest_intensities.copy()

        curve = self._capture_curves[mode]
        curve.setData(self._wavelengths, slot[mode.value])
        curve.show()
        if self._legend is not None:
            self._legend.removeItem(curve)
            curve.opts["name"] = self._legend_name(mode)
            self._legend.addItem(curve, curve.name())

        self._refresh_capture_indicators()
        self._update_export_enabled()
        self._update_image_save_enabled()
        self._update_post_process_enabled()
        self._refresh_status()
        self.statusBar().showMessage(
            f"Captured {self._active_display()} {mode.value} "
            f"({slot[mode.value].size} points)",
            3000,
        )

    def _import_csv_data(self) -> None:
        """Import every measurement column into the active acquisition session."""
        path_str, _selected = QFileDialog.getOpenFileName(
            self,
            "Import CSV Data",
            str(Path.home()),
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path_str:
            return
        try:
            wavelengths, columns = parse_unified_measurement_csv(Path(path_str))
        except MeasurementCsvError as exc:
            QMessageBox.warning(self, "Import Failed", str(exc))
            return
        loaded = self._store_imported_dataset(wavelengths, columns)
        if not loaded:
            return
        QMessageBox.information(
            self,
            "Import Complete",
            "Loaded "
            + ", ".join(loaded)
            + f" ({wavelengths.size} points).\n"
            "Post Process is available for the imported measurements.",
        )

    def _occupied_import_targets(
        self, columns: dict[Optional[str], dict[str, np.ndarray]]
    ) -> list[str]:
        occupied: list[str] = []
        for geometry_key, slots in columns.items():
            geometry = self._active_geometry if geometry_key is None else geometry_key
            if geometry not in self.spectra_data:
                continue
            for slot_name in slots:
                if self.spectra_data[geometry][slot_name] is not None:
                    occupied.append(f"{_display_mode(geometry)} {slot_name}")
        return occupied

    def _store_imported_dataset(
        self,
        wavelengths: np.ndarray,
        columns: dict[Optional[str], dict[str, np.ndarray]],
    ) -> Optional[list[str]]:
        """Store every imported series in the shared capture buffers.

        Unprefixed columns are written to the active optical mode. Returns the
        loaded labels, or ``None`` when the user cancels a replacement.
        """
        imported = np.asarray(wavelengths, dtype=np.float64)
        if self._wavelengths is not None and not _wavelength_axes_match(
            imported, self._wavelengths
        ):
            QMessageBox.warning(
                self,
                "Wavelength Axis Mismatch",
                "The imported wavelength axis does not match the current "
                "measurement grid.\n"
                f"Imported points: {imported.size}\n"
                f"Current points: {self._wavelengths.size}",
            )
            return None

        occupied = self._occupied_import_targets(columns)
        if occupied:
            reply = QMessageBox.question(
                self,
                "Replace Measurements",
                "Replace existing captures with the imported file?\n"
                + ", ".join(occupied),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return None

        if self._wavelengths is None:
            self._wavelengths = imported.copy()

        loaded: list[str] = []
        active_prefixed = set(columns.get(self._active_geometry, {}))
        for geometry_key, slots in columns.items():
            geometry = self._active_geometry if geometry_key is None else geometry_key
            if geometry not in self.spectra_data:
                continue
            for slot_name, values in slots.items():
                if slot_name not in CAPTURE_LABELS:
                    continue
                if geometry_key is None and slot_name in active_prefixed:
                    continue
                array = np.asarray(values, dtype=np.float64)
                if array.shape != self._wavelengths.shape:
                    QMessageBox.warning(
                        self,
                        "Import Failed",
                        f"{slot_name} does not match the session wavelength length.",
                    )
                    return None
                self.spectra_data[geometry][slot_name] = array.copy()
                self.reflectance_data[geometry] = None
                self.interpolated_reflectance[geometry] = None
                loaded.append(f"{_display_mode(geometry)} {slot_name}")

        if not loaded:
            QMessageBox.warning(
                self,
                "Import Failed",
                "None of the CSV columns matched a measurement slot.",
            )
            return None

        self._refresh_plot_for_geometry()
        self._refresh_capture_indicators()
        self._update_export_enabled()
        self._update_image_save_enabled()
        self._update_post_process_enabled()
        self._refresh_status()
        self.statusBar().showMessage(
            "Imported " + ", ".join(loaded),
            5000,
        )
        return loaded

    def _store_reflectance(self, mode_key: str, values: np.ndarray) -> None:
        """Keep the latest reflectance spectrum for one optical mode."""
        if mode_key not in self.reflectance_data:
            return
        self.reflectance_data[mode_key] = np.asarray(values, dtype=np.float64).copy()

    def _store_interpolated_reflectance(
        self,
        mode_key: str,
        wavelengths: Optional[np.ndarray],
        values: Optional[np.ndarray],
    ) -> None:
        """Keep the resampled reflectance grid for one optical mode."""
        if mode_key not in self.interpolated_reflectance:
            return
        if wavelengths is None or values is None:
            self.interpolated_reflectance[mode_key] = None
            return
        self.interpolated_reflectance[mode_key] = {
            "wavelengths": np.asarray(wavelengths, dtype=np.float64).copy(),
            "values": np.asarray(values, dtype=np.float64).copy(),
        }

    def _reset_current_geometry(self) -> None:
        geometry = self._active_geometry
        self.spectra_data[geometry] = _empty_geometry_slot()
        self.reflectance_data[geometry] = None
        self.interpolated_reflectance[geometry] = None
        self._refresh_plot_for_geometry()
        self._refresh_capture_indicators()
        self._update_export_enabled()
        self._update_image_save_enabled()
        self._update_post_process_enabled()
        self._refresh_status()
        self.statusBar().showMessage(
            f"Reset captures for {_display_mode(geometry)}",
            2500,
        )

    def _reset_all_geometries(self) -> None:
        reply = QMessageBox.question(
            self,
            "Reset All Optical Modes",
            "Clear Baseline / Standard / Sample for every optical mode "
            "(R₀, R₆₀, T)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.spectra_data = _build_spectra_data()
        self.reflectance_data = {key: None for key in OPTICAL_MODE_KEYS}
        self.interpolated_reflectance = {key: None for key in OPTICAL_MODE_KEYS}
        self._refresh_plot_for_geometry()
        self._refresh_capture_indicators()
        self._update_export_enabled()
        self._update_image_save_enabled()
        self._update_post_process_enabled()
        self._refresh_status()
        self.statusBar().showMessage("All optical-mode captures cleared", 2500)

    def _active_mode_has_captures(self) -> bool:
        """True when at least one capture exists for the active optical mode."""
        slot = self._current_slot()
        return any(slot[label] is not None for label in CAPTURE_LABELS)

    def _active_mode_complete(self) -> bool:
        """True when Baseline, Standard, and Sample exist for the active mode."""
        slot = self._current_slot()
        return all(slot[label] is not None for label in CAPTURE_LABELS)

    def _has_plotted_trace(self) -> bool:
        if self._latest_wavelengths is not None:
            return True
        return self._active_mode_has_captures()

    def _update_export_enabled(self) -> None:
        self._export_btn.setEnabled(self._active_mode_complete())

    def _update_image_save_enabled(self) -> None:
        self._save_image_btn.setEnabled(self._has_plotted_trace())

    def _update_post_process_enabled(self) -> None:
        self._post_process_btn.setEnabled(self._active_mode_has_captures())
    # ------------------------------------------------------------------
    # Controls / export
    # ------------------------------------------------------------------

    def _open_post_process(self) -> None:
        """Open the modal spectral data viewer for the active optical mode."""
        if not self._active_mode_has_captures():
            QMessageBox.warning(
                self,
                "No Captured Data",
                "Capture at least one trace for the active optical mode first.",
            )
            return

        wavelengths = self._wavelengths
        if wavelengths is None:
            wavelengths = self._latest_wavelengths
        if wavelengths is None:
            QMessageBox.warning(
                self,
                "No Wavelength Axis",
                "No wavelength data is available yet.",
            )
            return

        slot = self._current_slot()
        dialog = PostProcessDialogWithProgress(
            mode_display=self._active_display(),
            mode_key=self._active_geometry,
            wavelengths=wavelengths,
            baseline=slot[CaptureMode.BASELINE.value],
            standard=slot[CaptureMode.STANDARD.value],
            sample=slot[CaptureMode.SAMPLE.value],
            parent=self,
            on_imported=self._store_imported_dataset,
        )
        # Modal exec keeps live acquisition running on the worker thread;
        # stored capture arrays are only read, never mutated.
        dialog.exec()
        self._refresh_status()

    def _total_integration_us(self) -> int:
        """Compose SeaBreeze integration time: ``(ms * 1000) + us``."""
        return (
            int(self._integration_ms_spin.value()) * 1000
            + int(self._integration_us_spin.value())
        )

    def _apply_integration_fields(self, total_us: int) -> None:
        """Write a validated total back into the ms/µs fields without re-entry."""
        total_us = max(_MIN_INTEGRATION_US, min(total_us, _MAX_INTEGRATION_MS * 1000 + 999))
        ms, us = divmod(int(total_us), 1000)
        self._integration_syncing = True
        try:
            self._integration_ms_spin.setValue(ms)
            self._integration_us_spin.setValue(us)
        finally:
            self._integration_syncing = False

    def _on_integration_changed(self, _value: int = 0) -> None:
        """Push the combined integration time to the worker as total microseconds."""
        if self._integration_syncing:
            return

        total_us = self._total_integration_us()
        max_us = _MAX_INTEGRATION_MS * 1000 + 999
        if total_us < _MIN_INTEGRATION_US or total_us > max_us:
            self._apply_integration_fields(total_us)
            total_us = self._total_integration_us()
            self.statusBar().showMessage(
                f"Integration time adjusted to hardware limits ({total_us} µs)",
                4000,
            )

        QMetaObject.invokeMethod(
            self._worker,
            "set_integration_time",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(int, int(total_us)),
        )

    def _export_csv(self) -> None:
        mode_key = self._active_geometry
        mode_display = self._active_display()
        slot = self._current_slot()

        missing = [label for label in CAPTURE_LABELS if slot[label] is None]
        if missing:
            QMessageBox.warning(
                self,
                "Incomplete Data",
                f"Capture all three traces for {mode_display} before exporting.\n"
                f"Missing: {', '.join(missing)}",
            )
            return

        wavelengths = self._wavelengths
        if wavelengths is None:
            wavelengths = self._latest_wavelengths
        if wavelengths is None:
            QMessageBox.warning(
                self,
                "No Wavelength Axis",
                "No wavelength data is available yet.",
            )
            return

        baseline = slot[CaptureMode.BASELINE.value]
        standard = slot[CaptureMode.STANDARD.value]
        sample = slot[CaptureMode.SAMPLE.value]
        assert baseline is not None and standard is not None and sample is not None

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_token = _geometry_filename_token(mode_key)
        default_path = Path.home() / f"spectrum_{mode_token}_{stamp}.csv"
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            f"Export Spectrum Data ({mode_display})",
            str(default_path),
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path_str:
            return

        path = Path(path_str)
        if path.suffix.lower() != ".csv":
            path = path.with_suffix(".csv")

        headers = [
            "Wavelength (nm)",
            f"{mode_key}_Baseline (counts)",
            f"{mode_key}_Standard (counts)",
            f"{mode_key}_Sample (counts)",
        ]
        columns = [baseline, standard, sample]

        n = len(wavelengths)
        try:
            with path.open("w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(headers)
                for i in range(n):
                    row: list[str] = [f"{wavelengths[i]:.6f}"]
                    for col in columns:
                        value = float(col[i])
                        if math.isnan(value):
                            row.append("")
                        else:
                            row.append(f"{value:.6f}")
                    writer.writerow(row)
        except OSError as exc:
            QMessageBox.critical(self, "Export Failed", str(exc))
            return

        self.statusBar().showMessage(
            f"Exported {mode_display} spectrum to {path}",
            5000,
        )

    def _save_plot_image(self) -> None:
        if not self._has_plotted_trace():
            QMessageBox.warning(
                self,
                "Nothing to Save",
                "No live or captured traces are currently plotted.",
            )
            return

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_token = _geometry_filename_token(self._active_geometry)
        default_path = Path.home() / f"spectrum_{mode_token}_{stamp}.png"
        path_str, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Graph Image",
            str(default_path),
            "PNG Image (*.png);;JPEG Image (*.jpg);;All Files (*)",
        )
        if not path_str:
            return

        path = Path(path_str)
        suffix = path.suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg"}:
            if "JPEG" in selected_filter or "*.jpg" in selected_filter:
                path = path.with_suffix(".jpg")
            else:
                path = path.with_suffix(".png")

        QTimer.singleShot(0, lambda p=path: self._write_plot_image(p))

    def _write_plot_image(self, path: Path) -> None:
        try:
            exporter = ImageExporter(self._plot.plotItem)
            exporter.export(str(path))
        except Exception as exc:
            logger.exception("Failed to save graph image to %s", path)
            QMessageBox.critical(
                self,
                "Image Save Failed",
                f"Could not save graph image:\n{exc}",
            )
            self.statusBar().showMessage(f"Image save failed: {exc}", 6000)
            return

        self.statusBar().showMessage(
            f"Graph image saved successfully — {path}",
            5000,
        )

    # ------------------------------------------------------------------
    # Worker callbacks
    # ------------------------------------------------------------------

    def _on_data_acquired(
        self, wavelengths: np.ndarray, intensities: np.ndarray
    ) -> None:
        first_frame = self._latest_wavelengths is None
        self._latest_wavelengths = wavelengths
        self._latest_intensities = intensities
        if self._wavelengths is None:
            self._wavelengths = wavelengths.copy()
        self._live_curve.setData(wavelengths, intensities)
        if first_frame:
            self._update_image_save_enabled()

    def _on_status_message(self, message: str) -> None:
        logger.info("Worker: %s", message)

        if message.startswith("Spectrometer ready"):
            if "(mock)" in message.lower():
                self._device_name = MockSpectrometer.__name__
            elif "(hardware)" in message.lower():
                self._device_name = "Ocean Optics FLAME"
            else:
                self._device_name = message.split(":")[0].strip()

            if not self._acquisition_started:
                self._acquisition_started = True
                QMetaObject.invokeMethod(
                    self._worker,
                    "start_acquisition",
                    Qt.ConnectionType.QueuedConnection,
                )
                # Push current acquisition control values to the worker.
                self._on_integration_changed()
                QMetaObject.invokeMethod(
                    self._worker,
                    "set_scans_to_average",
                    Qt.ConnectionType.QueuedConnection,
                    Q_ARG(int, int(self._scans_spin.value())),
                )

        self._refresh_status()
        self.statusBar().showMessage(message, 4000)

    def _on_device_error(self, message: str) -> None:
        logger.error("Device error: %s", message)
        self.statusBar().showMessage(f"Error: {message}", 8000)
        QMessageBox.warning(self, "Spectrometer Error", message)

    def _refresh_status(self) -> None:
        slot = self._current_slot()
        captured_parts = [
            f"{label}:{'✓' if slot[label] is not None else '—'}"
            for label in CAPTURE_LABELS
        ]
        geo_summary = []
        for key in OPTICAL_MODE_KEYS:
            n = sum(
                1
                for label in CAPTURE_LABELS
                if self.spectra_data[key][label] is not None
            )
            geo_summary.append(f"{_display_mode(key)}:{n}/{len(CAPTURE_LABELS)}")

        text = (
            f"Device: {self._device_name}  |  "
            f"Mode: {self._active_display()}  |  "
            f"Slot: {self._active_mode.value}  |  "
            f"Current: {', '.join(captured_parts)}  |  "
            f"All: {', '.join(geo_summary)}"
        )
        self.statusBar().showMessage(text)


# ===========================================================================
# ADDED FEATURE: Nd and Nk window (shown as the "Post Process" window)
# ---------------------------------------------------------------------------
# A separate window, opened by the main window's Post Process button (which
# used to open the original Post Process dialog; that dialog's code is kept).
# Step 1 chooses what to calculate:
#   * Absolute reflectance: R0, R60 or both (any mode with Baseline, Standard
#     and Sample), steps 1-8;
#   * n1 and k1: needs R0 and R60, steps 1-11;
#   * T (transmission): placeholder button, disabled, nothing behind it yet.
# Workflow (one numbered step per button, in this order):
#   1 What to calculate + measurements (main-window captures, or Import
#     Measurement File here)                             2 Average
#   3 Relative reflectance   4 Interpolate reflectance   5 Import Si values
#   6 Interpolate Si values  7 Absolute reflectance
#   8 Plot ("Absolute Reflectance vs Wavelength" tab)
#   9 Nd fit: opens the Nd Fit window (n1 and d1 at one wavelength, the search
#     of Optical_Nd_V7S.py with pi; see thin_film.fit_nd)
#  10 Nk fit: opens the Nk Fit window (n1 and k1 at every interpolated
#     wavelength with d1 fixed, the search of Optical_NK_V7S.py with pi; see
#     thin_film.fit_nk_spectrum). Results go into the table.
#  11 Plot n1 and k1 ("n1 and k1 vs Wavelength" tab: both on the same axis).
# Excel export: one chart sheet per graph tab (each when its data exists).
# Step 7's bare-Si reflectance follows the lab's Excel sheet exactly
# (thin_film.si_wafer_reflectance_excel).
# The table shows every step's output. Redoing a step clears the steps after
# it. Each button has a footnote saying what it still needs, and a pop-up if
# it is clicked before then.
#
# Everything in this block is new. The original code is unchanged apart from
# one word in main():  SpectrometerApp()  ->  SpectrometerAppWithNdNk()
# It uses ProgressPopup from the progress pop-up block above SpectrometerApp,
# the original helper functions (averaging, relative reflectance,
# interpolation, Si file parsing), and the physics in thin_film.py.
#
# Remembering: one window lives on the main window, so closing and reopening
# it keeps everything. All calculated results are cleared as soon as the R0 or
# R60 measurements change (capture, import or reset). Imported Si values and
# the settings in the fields are kept.
#
# To remove this feature: delete this block, change main() back to
# SpectrometerApp(), and delete thin_film.py.
# ===========================================================================

import thin_film  # noqa: E402
from PyQt6.QtGui import QColor  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QCheckBox,
    QGridLayout,
    QListWidget,
    QListWidgetItem,
    QRadioButton,
    QTabWidget,
)
import zipfile as _zipfile  # noqa: E402
from xml.sax.saxutils import escape as _xml_escape  # noqa: E402

# Optical modes used by the Nd and Nk fits, with the angle each one measures.
_ND_NK_MODES: tuple[tuple[str, float], ...] = (("R0", 0.0), ("R60", 60.0))
_ND_NK_SLOTS: tuple[str, ...] = CAPTURE_LABELS

_ND_NK_STYLE = """
QDialog {
    background-color: #12141a;
    color: #e8eaed;
}
QScrollArea {
    background-color: transparent;
    border: none;
}
QScrollArea > QWidget > QWidget {
    background-color: transparent;
}
QLabel {
    color: #c5cad4;
}
QLabel#ndNkNote {
    color: #9aa3b2;
}
QGroupBox {
    color: #e8eaed;
    border: 1px solid #3d4450;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 8px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}
QTableView {
    background-color: #1a1d23;
    alternate-background-color: #21252d;
    gridline-color: #2e3440;
    color: #e8eaed;
    selection-background-color: #2a4568;
    border: 1px solid #3d4450;
}
QHeaderView::section {
    background-color: #242a35;
    color: #e8eaed;
    padding: 6px;
    border: 1px solid #3d4450;
    font-weight: 600;
}
QPushButton {
    background-color: #2a2f3a;
    border: 1px solid #3d4450;
    border-radius: 6px;
    padding: 2px 16px;
    min-height: 22px;
    color: #e8eaed;
}
QPushButton:hover {
    background-color: #353b48;
}
QPushButton:disabled {
    color: #6d7480;
}
QPushButton#calcChoiceBtn {
    /* Step 1 choice: amber, so it stands apart from the teal used elsewhere. */
    background-color: #2b2214;
    color: #ffc266;
    border: 1px solid #b7791f;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 14px;
    font-weight: 600;
    text-align: left;
}
QPushButton#calcChoiceBtn:hover:!disabled {
    background-color: #3a2c15;
    border: 1px solid #ffb347;
}
QPushButton#calcChoiceBtn:checked {
    background-color: #f0a030;
    color: #14161c;
    border: 2px solid #ffc266;
    font-weight: 700;
}
QPushButton#calcChoiceBtn:disabled {
    background-color: #1a1e26;
    color: #5c6370;
    border: 1px dashed #3a4150;
}
QPushButton#accentBtn {
    background-color: #123e4f;
    border: 1px solid #00bcd4;
    color: #e8fbff;
    font-weight: 600;
}
QPushButton#accentBtn:hover:!disabled {
    background-color: #16566c;
}
QPushButton#accentBtn:disabled {
    background-color: #1e2229;
    color: #5c6370;
    border: 1px solid #2a2f3a;
}
QSpinBox, QComboBox, QLineEdit {
    background-color: #1e2229;
    border: 1px solid #3d4450;
    border-radius: 4px;
    padding: 4px 8px;
    min-height: 26px;
    color: #e8eaed;
}
QLineEdit:focus, QSpinBox:focus, QComboBox:focus {
    border: 1px solid #00bcd4;
}
QLineEdit:disabled {
    color: #6d7480;
}
QCheckBox {
    color: #c5cad4;
}
QTabWidget#ndNkTabs::pane {
    border: 1px solid #3d4450;
    border-radius: 6px;
    top: -1px;
}
QTabWidget#ndNkTabs > QTabBar::tab {
    background-color: #1a1e26;
    color: #8b93a3;
    border: 1px solid #3a4150;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 6px 18px;
    margin-right: 2px;
    font-weight: 600;
}
QTabWidget#ndNkTabs > QTabBar::tab:selected {
    background-color: #123e4f;
    color: #e8fbff;
    border: 1px solid #00bcd4;
    border-bottom: none;
}
"""


class _NdNkTableModel(QAbstractTableModel):
    """Read-only table of named columns. Shorter columns show blank cells,
    NaN values show "N/A"; numbers use 6 decimals like the Post Process table."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._columns: list[tuple[str, np.ndarray]] = []

    def set_columns(self, columns: list[tuple[str, np.ndarray]]) -> None:
        self.beginResetModel()
        self._columns = [
            (header, np.asarray(values, dtype=np.float64)) for header, values in columns
        ]
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: B008
        if parent.isValid():
            return 0
        return max((values.size for _header, values in self._columns), default=0)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: B008
        if parent.isValid():
            return 0
        return len(self._columns)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            if 0 <= section < len(self._columns):
                return self._columns[section][0]
            return None
        return str(section + 1)

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        _header, values = self._columns[index.column()]
        row = index.row()
        if row >= values.size:
            return ""
        value = float(values[row])
        if not math.isfinite(value):
            return "N/A"
        return f"{value:.6f}"

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable

    def export_rows(self) -> tuple[list[str], list[list[str]]]:
        headers = [header for header, _values in self._columns]
        rows = [
            [
                self.data(self.index(row, column)) or ""
                for column in range(len(self._columns))
            ]
            for row in range(self.rowCount())
        ]
        return headers, rows


# ---------------------------------------------------------------------------
# Excel export: one .xlsx with the table, the graph (a real Excel chart linked
# to the table's columns) and the settings / fit results. Written with the
# standard library only (zip + XML), so it needs no extra package in the .exe.
# ---------------------------------------------------------------------------

_XLSX_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_XLSX_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_XLSX_PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_XLSX_A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
_XLSX_C_NS = "http://schemas.openxmlformats.org/drawingml/2006/chart"
_XLSX_XDR_NS = "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing"


def _xlsx_column_name(index: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA, ..."""
    name = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _xlsx_text(value: object) -> str:
    text = "".join(ch for ch in str(value) if ch in "\t\n" or ord(ch) >= 32)
    return _xml_escape(text)


def _xlsx_cell(reference: str, value: object, bold: bool = False) -> str:
    """One cell: a number (NaN / missing -> no cell, i.e. blank) or text."""
    style = ' s="1"' if bold else ""
    if value is None or isinstance(value, bool):
        return ""
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return ""
        return f'<c r="{reference}"{style}><v>{number!r}</v></c>'
    return (
        f'<c r="{reference}"{style} t="inlineStr"><is><t xml:space="preserve">'
        f"{_xlsx_text(value)}</t></is></c>"
    )


def _xlsx_sheet(
    rows: list[str], widths: list[float], freeze_header: bool, drawing: bool = False
) -> str:
    view = (
        '<sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" '
        'activePane="bottomLeft" state="frozen"/><selection pane="bottomLeft"/>'
        "</sheetView>"
        if freeze_header
        else '<sheetView workbookViewId="0"/>'
    )
    cols = "".join(
        f'<col min="{i + 1}" max="{i + 1}" width="{w:.1f}" customWidth="1"/>'
        for i, w in enumerate(widths)
    )
    return (
        f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<worksheet xmlns="{_XLSX_NS}" xmlns:r="{_XLSX_REL_NS}">'
        f"<sheetViews>{view}</sheetViews>"
        '<sheetFormatPr defaultRowHeight="15"/>'
        + (f"<cols>{cols}</cols>" if cols else "")
        + "<sheetData>" + "".join(rows) + "</sheetData>"
        + ('<drawing r:id="rId1"/>' if drawing else "")
        + "</worksheet>"
    )


def _xlsx_rich_title(text: str, size: int = 1200) -> str:
    return (
        f'<c:title><c:tx><c:rich><a:bodyPr/><a:lstStyle/><a:p><a:pPr><a:defRPr sz="{size}" b="1"/>'
        f'</a:pPr><a:r><a:rPr lang="en-US" sz="{size}" b="1"/><a:t>{_xlsx_text(text)}</a:t></a:r>'
        '</a:p></c:rich></c:tx><c:overlay val="0"/></c:title>'
    )


def _xlsx_num_ref(formula: str, values: np.ndarray) -> str:
    points = "".join(
        f'<c:pt idx="{i}"><c:v>{float(v)!r}</c:v></c:pt>'
        for i, v in enumerate(values)
        if math.isfinite(float(v))
    )
    return (
        f"<c:numRef><c:f>{_xlsx_text(formula)}</c:f><c:numCache><c:formatCode>General"
        f'</c:formatCode><c:ptCount val="{len(values)}"/>{points}</c:numCache></c:numRef>'
    )


def _xlsx_chart(chart: dict[str, object]) -> str:
    """A scatter chart with straight lines: one series per curve. Series with
    "secondary": True go on a second y axis on the right (title y2_title)."""
    series_xml = []
    secondary_xml = []
    # Excel lists primary-axis series first in the legend: number them that way.
    ordered = [s for s in chart["series"] if not s.get("secondary")] + [
        s for s in chart["series"] if s.get("secondary")
    ]
    hidden_entries = "".join(
        f'<c:legendEntry><c:idx val="{i}"/><c:delete val="1"/></c:legendEntry>'
        for i, s in enumerate(ordered)
        if not s.get("legend", True)
    )
    for index, series in enumerate(ordered):
        dash = '<a:prstDash val="dash"/>' if series["dashed"] else ""
        (secondary_xml if series.get("secondary") else series_xml).append(
            f'<c:ser><c:idx val="{index}"/><c:order val="{index}"/>'
            f"<c:tx><c:v>{_xlsx_text(series['name'])}</c:v></c:tx>"
            f'<c:spPr><a:ln w="19050" cap="rnd"><a:solidFill><a:srgbClr val="{series["color"]}"/>'
            f"</a:solidFill>{dash}<a:round/></a:ln></c:spPr>"
            '<c:marker><c:symbol val="none"/></c:marker>'
            f"<c:xVal>{_xlsx_num_ref(series['x_ref'], series['x'])}</c:xVal>"
            f"<c:yVal>{_xlsx_num_ref(series['y_ref'], series['y'])}</c:yVal>"
            '<c:smooth val="0"/></c:ser>'
        )

    def axis(ax_id: int, cross_id: int, position: str, title: str, limits: str,
             deleted: bool = False, gridlines: bool = True, crosses: str = "autoZero") -> str:
        return (
            f'<c:valAx><c:axId val="{ax_id}"/><c:scaling><c:orientation val="minMax"/>{limits}'
            f'</c:scaling><c:delete val="{1 if deleted else 0}"/><c:axPos val="{position}"/>'
            + (
                '<c:majorGridlines><c:spPr><a:ln w="6350"><a:solidFill><a:srgbClr val="D9D9D9"/>'
                "</a:solidFill></a:ln></c:spPr></c:majorGridlines>"
                if gridlines else ""
            )
            + (_xlsx_rich_title(title, 1000) if title else "")
            + '<c:numFmt formatCode="General" sourceLinked="0"/><c:majorTickMark val="out"/>'
            '<c:minorTickMark val="none"/><c:tickLblPos val="nextTo"/>'
            f'<c:crossAx val="{cross_id}"/><c:crosses val="{crosses}"/>'
            '<c:crossBetween val="midCat"/></c:valAx>'
        )

    x_limits = f'<c:max val="{chart["x_max"]!r}"/><c:min val="{chart["x_min"]!r}"/>'
    two_axes = bool(secondary_xml)
    # With a right-hand axis the plot is narrower so its labels don't hit the legend.
    plot_w, legend_x, legend_w = ("0.5", "0.74", "0.25") if two_axes else ("0.56", "0.69", "0.3")
    second = (
        '<c:scatterChart><c:scatterStyle val="lineMarker"/><c:varyColors val="0"/>'
        + "".join(secondary_xml)
        + '<c:axId val="500000003"/><c:axId val="500000004"/></c:scatterChart>'
        if two_axes else ""
    )
    second_axes = (
        axis(500000003, 500000004, "b", "", x_limits, deleted=True, gridlines=False)
        + axis(500000004, 500000003, "r", chart.get("y2_title", ""), "", gridlines=False,
               crosses="max")
        if two_axes else ""
    )
    return (
        f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<c:chartSpace xmlns:c="{_XLSX_C_NS}" xmlns:a="{_XLSX_A_NS}" xmlns:r="{_XLSX_REL_NS}">'
        '<c:roundedCorners val="0"/><c:chart>'
        f'{_xlsx_rich_title(chart["title"], 1100)}<c:autoTitleDeleted val="0"/><c:plotArea>'
        '<c:layout><c:manualLayout><c:layoutTarget val="inner"/><c:xMode val="edge"/>'
        f'<c:yMode val="edge"/><c:x val="0.1"/><c:y val="0.12"/><c:w val="{plot_w}"/>'
        '<c:h val="0.7"/></c:manualLayout></c:layout>'
        '<c:scatterChart><c:scatterStyle val="lineMarker"/><c:varyColors val="0"/>'
        + "".join(series_xml)
        + '<c:axId val="500000001"/><c:axId val="500000002"/></c:scatterChart>'
        + second
        + axis(500000001, 500000002, "b", chart["x_title"], x_limits)
        + axis(500000002, 500000001, "l", chart["y_title"], "")
        + second_axes
        + '</c:plotArea><c:legend><c:legendPos val="r"/>' + hidden_entries + '<c:layout><c:manualLayout>'
        f'<c:xMode val="edge"/><c:yMode val="edge"/><c:x val="{legend_x}"/><c:y val="0.12"/>'
        f'<c:w val="{legend_w}"/><c:h val="0.3"/></c:manualLayout></c:layout><c:overlay val="0"/>'
        '<c:txPr><a:bodyPr/><a:lstStyle/><a:p><a:pPr><a:defRPr sz="900"/></a:pPr>'
        '<a:endParaRPr lang="en-US"/></a:p></c:txPr></c:legend>'
        '<c:plotVisOnly val="1"/><c:dispBlanksAs val="gap"/></c:chart></c:chartSpace>'
    )


def _write_results_workbook(
    path: Path,
    data_columns: list[tuple[str, np.ndarray]],
    charts: object,
    info_rows: list[tuple[object, object, bool]],
) -> None:
    """Write the .xlsx: sheet "Data" (the table), one sheet per chart and
    "Settings and fits" (label / value rows; the bool marks bold rows).

    ``charts``: a list of (sheet name, chart), or one chart / None (sheet
    "Graph"). A chart has title, x_title, y_title (y2_title for a right-hand
    axis), x_min, x_max, an optional "note" and series (name, x_column,
    y_column = indexes into data_columns, length = number of points, color,
    dashed, optional secondary). The series point at the Data sheet, so the
    charts follow edits.
    """
    if charts is None:
        charts = []
    elif isinstance(charts, dict):
        charts = [("Graph", charts)]
    # Data sheet
    headers = [header for header, _values in data_columns]
    arrays = [np.asarray(values, dtype=np.float64).ravel() for _header, values in data_columns]
    row_count = max((a.size for a in arrays), default=0)
    letters = [_xlsx_column_name(i) for i in range(len(arrays))]
    rows = [
        '<row r="1">'
        + "".join(_xlsx_cell(f"{letters[i]}1", h, bold=True) for i, h in enumerate(headers))
        + "</row>"
    ]
    for r in range(row_count):
        cells = "".join(
            _xlsx_cell(f"{letters[i]}{r + 2}", a[r]) for i, a in enumerate(arrays) if r < a.size
        )
        rows.append(f'<row r="{r + 2}">{cells}</row>')
    widths = [min(42.0, max(12.0, len(h) * 1.05 + 2)) for h in headers]
    sheets = [("Data", _xlsx_sheet(rows, widths, freeze_header=True))]

    # Chart sheets
    chart_parts: list[tuple[str, str]] = []  # (sheet name, chart xml)
    for sheet_name, chart in charts:
        series = []
        for s in chart["series"]:
            x_col, y_col, n = s["x_column"], s["y_column"], s["length"]
            first = int(s.get("first", 0))  # first data row of the series (0 = row 2)
            top, bottom = first + 2, first + n + 1
            series.append(
                {
                    "name": s["name"], "color": s["color"], "dashed": s["dashed"],
                    "secondary": s.get("secondary", False),
                    "legend": s.get("legend", True),
                    "x_ref": f"'Data'!${letters[x_col]}${top}:${letters[x_col]}${bottom}",
                    "y_ref": f"'Data'!${letters[y_col]}${top}:${letters[y_col]}${bottom}",
                    "x": arrays[x_col][first:first + n], "y": arrays[y_col][first:first + n],
                }
            )
        chart_parts.append((sheet_name, _xlsx_chart({**chart, "series": series})))
        note = _xlsx_cell(
            "A1",
            chart.get("note", "Drawn from the Data sheet: the same curves as the app's Graph tab."),
        )
        sheets.append(
            (sheet_name, _xlsx_sheet([f'<row r="1">{note}</row>'], [], False, drawing=True))
        )

    # Settings and fits sheet
    info = []
    for r, (label, value, bold) in enumerate(info_rows, start=1):
        info.append(
            f'<row r="{r}">{_xlsx_cell(f"A{r}", label, bold)}{_xlsx_cell(f"B{r}", value, bold)}</row>'
        )
    sheets.append(("Settings and fits", _xlsx_sheet(info, [44.0, 60.0], freeze_header=False)))

    # Package parts
    workbook_sheets = "".join(
        f'<sheet name="{_xlsx_text(name)}" sheetId="{i + 1}" r:id="rId{i + 1}"/>'
        for i, (name, _xml) in enumerate(sheets)
    )
    workbook_rels = "".join(
        f'<Relationship Id="rId{i + 1}" Type="{_XLSX_REL_NS}/worksheet" '
        f'Target="worksheets/sheet{i + 1}.xml"/>'
        for i in range(len(sheets))
    ) + (
        f'<Relationship Id="rId{len(sheets) + 1}" Type="{_XLSX_REL_NS}/styles" '
        'Target="styles.xml"/>'
    )
    overrides = "".join(
        f'<Override PartName="/xl/worksheets/sheet{i + 1}.xml" ContentType="application/'
        'vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for i in range(len(sheets))
    )
    for k in range(1, len(chart_parts) + 1):
        overrides += (
            f'<Override PartName="/xl/drawings/drawing{k}.xml" ContentType="application/'
            'vnd.openxmlformats-officedocument.drawing+xml"/>'
            f'<Override PartName="/xl/charts/chart{k}.xml" ContentType="application/'
            'vnd.openxmlformats-officedocument.drawingml.chart+xml"/>'
        )
    header = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    parts = {
        "[Content_Types].xml": (
            f'{header}<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.'
            'relationships+xml"/><Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-'
            'officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/styles.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
            '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-'
            'package.core-properties+xml"/>'
            f"{overrides}</Types>"
        ),
        "_rels/.rels": (
            f'{header}<Relationships xmlns="{_XLSX_PKG_REL_NS}">'
            f'<Relationship Id="rId1" Type="{_XLSX_REL_NS}/officeDocument" Target="xl/workbook.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/'
            'relationships/metadata/core-properties" Target="docProps/core.xml"/></Relationships>'
        ),
        "docProps/core.xml": (
            f'{header}<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/'
            'metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" '
            'xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/'
            'XMLSchema-instance"><dc:title>Post Process results</dc:title><dc:creator>'
            'Multi-Function Optical Analyzer</dc:creator><dcterms:created xsi:type="dcterms:W3CDTF">'
            f'{datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")}</dcterms:created></cp:coreProperties>'
        ),
        "xl/workbook.xml": (
            f'{header}<workbook xmlns="{_XLSX_NS}" xmlns:r="{_XLSX_REL_NS}">'
            f"<bookViews><workbookView/></bookViews><sheets>{workbook_sheets}</sheets></workbook>"
        ),
        "xl/_rels/workbook.xml.rels": (
            f'{header}<Relationships xmlns="{_XLSX_PKG_REL_NS}">{workbook_rels}</Relationships>'
        ),
        "xl/styles.xml": (
            f'{header}<styleSheet xmlns="{_XLSX_NS}"><fonts count="2"><font><sz val="11"/>'
            '<name val="Calibri"/></font><font><b/><sz val="11"/><name val="Calibri"/></font>'
            '</fonts><fills count="2"><fill><patternFill patternType="none"/></fill><fill>'
            '<patternFill patternType="gray125"/></fill></fills><borders count="1"><border>'
            '<left/><right/><top/><bottom/><diagonal/></border></borders><cellStyleXfs count="1">'
            '<xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
            '<cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
            '<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/>'
            '</cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/>'
            "</cellStyles></styleSheet>"
        ),
    }
    for i, (_name, xml) in enumerate(sheets):
        parts[f"xl/worksheets/sheet{i + 1}.xml"] = xml
    sheet_names = [name for name, _xml in sheets]
    for k, (sheet_name, chart_xml) in enumerate(chart_parts, start=1):
        graph_index = sheet_names.index(sheet_name) + 1
        parts[f"xl/worksheets/_rels/sheet{graph_index}.xml.rels"] = (
            f'{header}<Relationships xmlns="{_XLSX_PKG_REL_NS}"><Relationship Id="rId1" '
            f'Type="{_XLSX_REL_NS}/drawing" Target="../drawings/drawing{k}.xml"/></Relationships>'
        )
        parts[f"xl/drawings/drawing{k}.xml"] = (
            f'{header}<xdr:wsDr xmlns:xdr="{_XLSX_XDR_NS}" xmlns:a="{_XLSX_A_NS}">'
            # 6.3 x 4.3 inches from cell A3: fits a printed A4 or Letter page.
            '<xdr:oneCellAnchor><xdr:from><xdr:col>0</xdr:col><xdr:colOff>0</xdr:colOff>'
            '<xdr:row>2</xdr:row><xdr:rowOff>0</xdr:rowOff></xdr:from>'
            '<xdr:ext cx="5760720" cy="3931920"/><xdr:graphicFrame macro="">'
            f'<xdr:nvGraphicFramePr><xdr:cNvPr id="{k + 1}" name="Chart {k}"/>'
            '<xdr:cNvGraphicFramePr/></xdr:nvGraphicFramePr><xdr:xfrm><a:off x="0" y="0"/>'
            '<a:ext cx="0" cy="0"/></xdr:xfrm><a:graphic><a:graphicData uri="http://schemas.'
            f'openxmlformats.org/drawingml/2006/chart"><c:chart xmlns:c="{_XLSX_C_NS}" '
            f'xmlns:r="{_XLSX_REL_NS}" r:id="rId1"/></a:graphicData></a:graphic>'
            "</xdr:graphicFrame><xdr:clientData/></xdr:oneCellAnchor></xdr:wsDr>"
        )
        parts[f"xl/drawings/_rels/drawing{k}.xml.rels"] = (
            f'{header}<Relationships xmlns="{_XLSX_PKG_REL_NS}"><Relationship Id="rId1" '
            f'Type="{_XLSX_REL_NS}/chart" Target="../charts/chart{k}.xml"/></Relationships>'
        )
        parts[f"xl/charts/chart{k}.xml"] = chart_xml
    temp = path.with_name(path.name + ".part")
    with _zipfile.ZipFile(temp, "w", _zipfile.ZIP_DEFLATED) as archive:
        for name, xml in parts.items():
            archive.writestr(name, xml.encode("utf-8"))
    temp.replace(path)  # only replace an existing file once the new one is complete


# Graph tabs of the Post Process window and their sheets in the Excel export
# (sheet names: at most 31 characters).
_ABS_GRAPH_TAB = "Absolute Reflectance vs Wavelength"
_NK_GRAPH_TAB = "n₁ and k₁ vs Wavelength"
_ABS_GRAPH_SHEET = "Abs Reflectance vs Wavelength"
_NK_GRAPH_SHEET = "n1 and k1 vs Wavelength"
_NK_PLOT_COLORS = {"n1": (100, 181, 246), "k1": (255, 138, 101)}  # blue, deep orange


def _show_maximized(window: QWidget) -> None:
    """Open a window maximized (fills the screen it is on, title bar kept).

    A window that is already open on screen keeps the size the user gave it
    and is just brought to the front.
    """
    if not window.isVisible() or window.isMinimized():
        window.showMaximized()
    window.raise_()
    window.activateWindow()


class NdNkWindow(QDialog):
    """Nd and Nk window (non-modal). See the block comment above."""

    _NEEDS_COLOR = "#f0c674"
    _READY_COLOR = "#5dcaa5"
    _PLOT_COLORS = {"R0": (0, 188, 212), "R60": (255, 170, 60)}

    def __init__(self, main_window: "SpectrometerApp") -> None:
        super().__init__(
            main_window,
            Qt.WindowType.Window
            | Qt.WindowType.WindowMinMaxButtonsHint
            | Qt.WindowType.WindowCloseButtonHint,
        )
        self._main = main_window
        self.setWindowTitle("Post Process")
        self.setModal(False)
        self.resize(1320, 860)
        self.setStyleSheet(_ND_NK_STYLE)

        # Step 1 choice: "absolute" (absolute reflectance only: R0, R60 or both,
        # steps 1-8) or "nk" (n1 and k1: R0 and R60, steps 1-10). None until chosen.
        self.calculation: Optional[str] = None
        # Inputs (kept until replaced).
        self.si_wavelength: Optional[np.ndarray] = None
        self.si_n: Optional[np.ndarray] = None
        self.si_k: Optional[np.ndarray] = None
        self._si_source = ""
        # Results of each step. Redoing a step clears the steps after it, and
        # new R0 / R60 measurements clear all of them.
        # {"window", "modes": (R0 and/or R60), (mode, slot): array}
        self.averages: Optional[dict[object, object]] = None
        self.relative: Optional[dict[str, np.ndarray]] = None  # {mode: array}, detector axis
        self.relative_interp: Optional[dict[str, object]] = None  # {"grid", "method", mode: array}
        self.si_interp: Optional[dict[str, object]] = None  # {"grid", "n", "k", "method", "same_grid"}
        self.absolute: Optional[dict[str, object]] = None  # {"grid", "si_r_<mode>", mode: array}
        self.plotted = False
        # Step 9: the last Nd fit (thin_film.fit_nd result) and its window.
        self.nd_fit: Optional[dict[str, object]] = None
        self._nd_fit_dialog: Optional["NdFitDialog"] = None
        # Step 10: finished Nk runs not yet stored (each: one wavelength range and
        # its inputs), and the stored result (runs merged at Finish).
        self.nk_segments: list[dict[str, object]] = []
        self.nk_run_counter = 0
        self.nk_fit: Optional[dict[str, object]] = None
        self._nk_fit_dialog: Optional["NkFitDialog"] = None
        # Step 11: the stored n1, k1 plotted (cleared whenever the stored result changes).
        self.nk_plotted = False
        self._nk_plot_curves: list[object] = []
        # The measurement arrays the results were calculated from.
        self._results_inputs: Optional[tuple[Optional[np.ndarray], ...]] = None
        # The measurement arrays currently shown in the table.
        self._shown_inputs: Optional[tuple[Optional[np.ndarray], ...]] = None
        # Increases with every successful step (used by the progress pop-up).
        self._revision = 0

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        header = QHBoxLayout()
        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        header.addWidget(self._summary, stretch=1)
        close_btn = QPushButton("Close")
        close_btn.setObjectName("accentBtn")
        close_btn.setAutoDefault(False)
        close_btn.setDefault(False)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.clicked.connect(self.close)
        header.addWidget(close_btn)
        root.addLayout(header)

        body = QHBoxLayout()
        body.setSpacing(12)
        body.addWidget(self._build_sidebar())
        body.addWidget(self._build_results_tabs(), stretch=1)
        root.addLayout(body, stretch=1)

        self._apply_calculation_visibility()  # only step 1 until a choice is made
        self.sync_measurements()
        self._refresh_table()

    # ------------------------------------------------------------------
    # Layout: sidebar with the eight workflow steps
    # ------------------------------------------------------------------

    def _build_sidebar(self) -> QScrollArea:
        content = QWidget()
        content.setMinimumWidth(300)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 4, 0)
        layout.setSpacing(10)
        self._step_groups: list[QGroupBox] = []  # steps 1..11, in order
        for build in (
            self._build_measurements_group,
            self._build_average_group,
            self._build_relative_group,
            self._build_interp_reflectance_group,
            self._build_import_si_group,
            self._build_interp_si_group,
            self._build_absolute_group,
            self._build_plot_group,
            self._build_nd_fit_group,
            self._build_nk_fit_group,
            self._build_nk_plot_group,
        ):
            group = build()
            # Never shorter than its text needs (the sidebar scrolls instead of
            # squeezing a box), and it grows when a footnote gets longer.
            group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
            self._step_groups.append(group)
            layout.addWidget(group)
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        scroll.setFixedWidth(340)
        return scroll

    @staticmethod
    def _note(text: str = "") -> QLabel:
        label = QLabel(text)
        label.setObjectName("ndNkNote")
        label.setWordWrap(True)
        return label

    @staticmethod
    def _footnote() -> QLabel:
        """Live note under a button: what is still needed, or that it is ready."""
        label = QLabel("")
        label.setWordWrap(True)
        label.setTextFormat(Qt.TextFormat.RichText)
        return label

    def _set_footnote(self, label: QLabel, needed: list[str], ready_text: str) -> None:
        if needed:
            items = "".join(f"<br>&nbsp;• {item}" for item in needed)
            label.setText(
                f"<span style='color:{self._NEEDS_COLOR}'>Still needed:{items}</span>"
            )
        else:
            label.setText(f"<span style='color:{self._READY_COLOR}'>{ready_text}</span>")

    @staticmethod
    def _grid_edit(default: float, placeholder: str) -> QLineEdit:
        edit = QLineEdit(f"{default:.3f}")
        edit.setPlaceholderText(placeholder)
        validator = QDoubleValidator(0.0, 5000.0, 3, edit)
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        validator.setLocale(QLocale.c())
        edit.setValidator(validator)
        return edit

    def _step_button(self, text: str, step: str, title: str) -> QPushButton:
        button = QPushButton(text)
        button.setObjectName("accentBtn")
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(lambda _checked=False: self._on_step_clicked(step, title))
        return button

    def _grid_form(self, prefix: str) -> tuple[QFormLayout, QLineEdit, QLineEdit, QLineEdit]:
        form = QFormLayout()
        form.setSpacing(8)
        start = self._grid_edit(350.0, f"{prefix} start wavelength")
        end = self._grid_edit(1000.0, f"{prefix} end wavelength")
        step = self._grid_edit(1.0, f"{prefix} step size")
        form.addRow("Start (nm)", start)
        form.addRow("End (nm)", end)
        form.addRow("Step (nm)", step)
        for edit in (start, end, step):
            edit.textChanged.connect(self._update_footnotes)
        return form, start, end, step

    _CALCULATIONS = (
        # key, button text, note under it
        ("absolute", "Absolute reflectance", "R₀, R₆₀ or both · steps 1–8"),
        ("nk", "n₁ and k₁", "needs R₀ and R₆₀ · steps 1–11"),
        ("t", "T (transmission)", "placeholder, not set up yet"),
    )

    def _build_measurements_group(self) -> QGroupBox:
        group = QGroupBox("1 · What to calculate")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)
        self._calc_group = QButtonGroup(self)
        self._calc_group.setExclusive(True)
        self._calc_buttons: dict[str, QPushButton] = {}
        for key, text, note in self._CALCULATIONS:
            button = QPushButton(text)
            button.setObjectName("calcChoiceBtn")
            button.setCheckable(True)
            button.setMinimumHeight(38)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            if key == "t":
                button.setEnabled(False)  # placeholder: nothing behind it yet
                button.setToolTip("Placeholder for T (transmission): not set up yet.")
            else:
                button.clicked.connect(lambda _checked=False, k=key: self._set_calculation(k))
            self._calc_group.addButton(button)
            self._calc_buttons[key] = button
            layout.addWidget(button)
            hint = self._note(note)
            hint.setContentsMargins(4, 0, 0, 4)
            layout.addWidget(hint)
        heading = QLabel("Measurements")
        heading.setStyleSheet("font-weight: 600; margin-top: 6px;")
        layout.addWidget(heading)
        self._measurement_labels: dict[str, QLabel] = {}
        for mode, _angle in _ND_NK_MODES:
            label = QLabel("")
            label.setTextFormat(Qt.TextFormat.RichText)
            self._measurement_labels[mode] = label
            layout.addWidget(label)
        self._measurement_footnote = self._footnote()
        layout.addWidget(self._measurement_footnote)
        self._import_measurements_btn = QPushButton("Import Measurement File")
        self._import_measurements_btn.setObjectName("accentBtn")
        self._import_measurements_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._import_measurements_btn.clicked.connect(self._import_measurements)
        layout.addWidget(self._import_measurements_btn)
        layout.addWidget(
            self._note(
                "Captured in the main window, or imported here from a CSV or Excel "
                "file (same format as Import CSV Data). Capturing or importing new "
                "R₀ or R₆₀ data clears every calculation below."
            )
        )
        return group

    def _build_average_group(self) -> QGroupBox:
        group = QGroupBox("2 · Average")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)
        form = QFormLayout()
        self._window_spin = QSpinBox()
        self._window_spin.setRange(2, 50)
        self._window_spin.setValue(5)
        self._window_spin.setSuffix(" points")
        self._window_spin.valueChanged.connect(self._update_footnotes)
        form.addRow("Window size", self._window_spin)
        layout.addLayout(form)
        self._average_btn = self._step_button(
            "Average All Traces", "average", "Averaging all traces"
        )
        layout.addWidget(self._average_btn)
        self._average_footnote = self._footnote()
        layout.addWidget(self._average_footnote)
        return group

    def _build_relative_group(self) -> QGroupBox:
        group = QGroupBox("3 · Relative reflectance")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)
        self._relative_btn = self._step_button(
            "Calculate Relative Reflectance", "relative", "Calculating relative reflectance"
        )
        layout.addWidget(self._relative_btn)
        self._relative_footnote = self._footnote()
        layout.addWidget(self._relative_footnote)
        layout.addWidget(
            self._note("(Sample − Baseline) / (Standard − Baseline) for R₀ and R₆₀.")
        )
        return group

    def _build_interp_reflectance_group(self) -> QGroupBox:
        group = QGroupBox("4 · Interpolate reflectance")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)
        form, self._refl_start, self._refl_end, self._refl_step = self._grid_form(
            "Reflectance"
        )
        self._refl_method_combo = QComboBox()
        self._refl_method_combo.addItems(["Linear", "Cubic Spline"])
        form.addRow("Method", self._refl_method_combo)
        layout.addLayout(form)
        self._interp_refl_btn = self._step_button(
            "Interpolate Reflectance", "interp_refl", "Interpolating relative reflectance"
        )
        layout.addWidget(self._interp_refl_btn)
        self._interp_refl_footnote = self._footnote()
        layout.addWidget(self._interp_refl_footnote)
        return group

    def _build_import_si_group(self) -> QGroupBox:
        group = QGroupBox("5 · Silicon reference")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)
        self._import_si_btn = QPushButton("Import Si Values")
        self._import_si_btn.setObjectName("accentBtn")
        self._import_si_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._import_si_btn.clicked.connect(self._import_si_values)
        layout.addWidget(self._import_si_btn)
        self._si_footnote = self._footnote()
        layout.addWidget(self._si_footnote)
        return group

    def _build_interp_si_group(self) -> QGroupBox:
        group = QGroupBox("6 · Interpolate Si values")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)
        self._si_same_grid = QCheckBox("Use the reflectance grid from step 4")
        self._si_same_grid.setChecked(True)
        layout.addWidget(self._si_same_grid)
        form, self._si_start, self._si_end, self._si_step = self._grid_form("Si")
        self._si_method_combo = QComboBox()
        self._si_method_combo.addItems(["Linear", "Cubic", "Nearest"])
        form.addRow("Method", self._si_method_combo)
        layout.addLayout(form)
        for edit in (self._si_start, self._si_end, self._si_step):
            edit.setEnabled(False)  # ticked by default: step 4's grid is used
        self._si_same_grid.toggled.connect(self._on_si_same_grid_toggled)
        self._interp_si_btn = self._step_button(
            "Interpolate Si (n and k)", "interp_si", "Interpolating Si values"
        )
        layout.addWidget(self._interp_si_btn)
        self._interp_si_footnote = self._footnote()
        layout.addWidget(self._interp_si_footnote)
        return group

    def _build_absolute_group(self) -> QGroupBox:
        group = QGroupBox("7 · Absolute reflectance")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)
        self._absolute_btn = self._step_button(
            "Calculate Absolute Reflectance", "absolute", "Calculating absolute reflectance"
        )
        layout.addWidget(self._absolute_btn)
        self._absolute_footnote = self._footnote()
        layout.addWidget(self._absolute_footnote)
        layout.addWidget(
            self._note(
                "Interpolated relative reflectance × bare-Si reflectance at the same "
                "angle (0°, and 60° s-polarized), calculated from the interpolated "
                "Si n and k the same way as the lab's Excel sheet (each step rounded "
                "to 4 decimals), using π rather than the sheet's 3.14."
            )
        )
        return group

    def _build_plot_group(self) -> QGroupBox:
        group = QGroupBox("8 · Plot")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)
        self._plot_btn = QPushButton("Plot Absolute Reflectance")
        self._plot_btn.setObjectName("accentBtn")
        self._plot_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._plot_btn.clicked.connect(self._on_plot_clicked)
        layout.addWidget(self._plot_btn)
        self._plot_footnote = self._footnote()
        layout.addWidget(self._plot_footnote)
        layout.addWidget(
            self._note(
                "Plots R₀ and R₆₀ absolute reflectance (solid) and the bare-Si R₀ and "
                "R₆₀ from step 7 (dashed). Click a legend entry to hide or show that "
                "curve."
            )
        )
        return group

    def _build_nd_fit_group(self) -> QGroupBox:
        group = QGroupBox("9 · Nd fit")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)
        self._nd_fit_btn = QPushButton("Open Nd Fit")
        self._nd_fit_btn.setObjectName("accentBtn")
        self._nd_fit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._nd_fit_btn.clicked.connect(self._open_nd_fit)
        layout.addWidget(self._nd_fit_btn)
        self._nd_fit_footnote = self._footnote()
        layout.addWidget(self._nd_fit_footnote)
        layout.addWidget(
            self._note(
                "Fits the film's n₁ and d₁ at one wavelength from the absolute R₀ "
                "and R₆₀ of step 7 (the search of Optical_Nd_V7S.py, with π). Opens "
                "in its own window."
            )
        )
        return group

    def _build_nk_fit_group(self) -> QGroupBox:
        group = QGroupBox("10 · Nk fit")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)
        self._nk_fit_btn = QPushButton("Open Nk Fit")
        self._nk_fit_btn.setObjectName("accentBtn")
        self._nk_fit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._nk_fit_btn.clicked.connect(self._open_nk_fit)
        layout.addWidget(self._nk_fit_btn)
        self._nk_fit_footnote = self._footnote()
        layout.addWidget(self._nk_fit_footnote)
        layout.addWidget(
            self._note(
                "Fits the film's n₁ and k₁ at every interpolated wavelength with d₁ "
                "fixed (from the Nd fit), using the search of Optical_NK_V7S.py with "
                "π. Results are added to the table."
            )
        )
        return group

    def _build_nk_plot_group(self) -> QGroupBox:
        group = QGroupBox("11 · Plot n₁ and k₁")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)
        self._plot_nk_btn = QPushButton("Plot n₁ and k₁")
        self._plot_nk_btn.setObjectName("accentBtn")
        self._plot_nk_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._plot_nk_btn.clicked.connect(self._on_plot_nk_clicked)
        layout.addWidget(self._plot_nk_btn)
        self._plot_nk_footnote = self._footnote()
        layout.addWidget(self._plot_nk_footnote)
        layout.addWidget(
            self._note(
                "Plots the stored n₁ and k₁ from step 10 against wavelength in one "
                "graph, on the same axis. Click a legend entry to hide or show that "
                "curve."
            )
        )
        return group

    def _on_si_same_grid_toggled(self, checked: bool) -> None:
        for edit in (self._si_start, self._si_end, self._si_step):
            edit.setEnabled(not checked)
        self._update_footnotes()

    # ------------------------------------------------------------------
    # Layout: results (table and graph tabs)
    # ------------------------------------------------------------------

    def _build_results_tabs(self) -> QTabWidget:
        self._results_tabs = QTabWidget()
        self._results_tabs.setObjectName("ndNkTabs")

        table_page = QWidget()
        layout = QVBoxLayout(table_page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        self._model = _NdNkTableModel(self)
        self._table = QTableView()
        self._table.setModel(self._model)
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self._table.setWordWrap(False)
        self._table.verticalHeader().setDefaultSectionSize(22)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        layout.addWidget(self._table, stretch=1)
        bar = QHBoxLayout()
        self._export_footnote = self._footnote()
        bar.addWidget(self._export_footnote, stretch=1)
        self._export_btn = QPushButton("Export (Excel or CSV)")
        self._export_btn.setObjectName("accentBtn")
        self._export_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._export_btn.setToolTip(
            "Excel: one file with the table, the graph and the settings / fit "
            "results. CSV: the table only (a CSV file cannot hold a graph)."
        )
        self._export_btn.clicked.connect(self._export_csv)
        bar.addWidget(self._export_btn)
        layout.addLayout(bar)
        self._results_tabs.addTab(table_page, "Table")

        graph_page = QWidget()
        graph_layout = QVBoxLayout(graph_page)
        graph_layout.setContentsMargins(8, 8, 8, 8)
        graph_layout.setSpacing(8)
        self._plot_widget = pg.PlotWidget()
        for axis in ("left", "bottom"):
            self._plot_widget.getAxis(axis).enableAutoSIPrefix(False)
        self._plot_widget.setLabel("bottom", "Wavelength (nm)")
        self._plot_widget.setLabel("left", "Reflectance")
        self._plot_widget.showGrid(x=True, y=True, alpha=0.25)
        # Legend in the top-right corner, on a solid box so curves behind it
        # don't make it hard to read. Clicking an entry hides / shows its curve.
        self._plot_legend = self._plot_widget.addLegend(
            offset=(-10, 10),
            brush=pg.mkBrush(26, 29, 35, 230),
            pen=pg.mkPen(80, 86, 96),
        )
        self._plot_legend.setZValue(10)
        self._plot_widget.setTitle("Absolute reflectance and bare-Si reflectance vs wavelength")
        graph_layout.addWidget(self._plot_widget, stretch=1)
        graph_bar = QHBoxLayout()
        self._save_graph_footnote = self._footnote()
        graph_bar.addWidget(self._save_graph_footnote, stretch=1)
        self._save_graph_btn = QPushButton("Save Graph Image")
        self._save_graph_btn.setObjectName("accentBtn")
        self._save_graph_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._save_graph_btn.clicked.connect(self._save_graph_image)
        graph_bar.addWidget(self._save_graph_btn)
        self._export_graph_tab_btn = QPushButton("Export (Excel or CSV)")
        self._export_graph_tab_btn.setObjectName("accentBtn")
        self._export_graph_tab_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._export_graph_tab_btn.setToolTip(self._export_btn.toolTip())
        self._export_graph_tab_btn.clicked.connect(self._export_csv)
        graph_bar.addWidget(self._export_graph_tab_btn)
        graph_layout.addLayout(graph_bar)
        self._results_tabs.addTab(graph_page, _ABS_GRAPH_TAB)
        self._results_tabs.addTab(self._build_nk_graph_page(), _NK_GRAPH_TAB)
        return self._results_tabs

    def _build_nk_graph_page(self) -> QWidget:
        """Step 11's tab: n1 and k1 vs wavelength on the same axis."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        self._nk_plot_widget = pg.PlotWidget()
        item = self._nk_plot_widget.getPlotItem()
        for axis in ("left", "bottom"):
            item.getAxis(axis).enableAutoSIPrefix(False)
        item.setLabel("bottom", "Wavelength (nm)")
        item.setLabel("left", "n₁ and k₁")
        item.showGrid(x=True, y=True, alpha=0.25)
        item.setTitle("n₁ and k₁ vs wavelength")
        self._nk_plot_legend = item.addLegend(
            offset=(-10, 10),
            brush=pg.mkBrush(26, 29, 35, 230),
            pen=pg.mkPen(80, 86, 96),
        )
        self._nk_plot_legend.setZValue(20)
        layout.addWidget(self._nk_plot_widget, stretch=1)
        bar = QHBoxLayout()
        self._save_nk_graph_footnote = self._footnote()
        bar.addWidget(self._save_nk_graph_footnote, stretch=1)
        self._save_nk_graph_btn = QPushButton("Save Graph Image")
        self._save_nk_graph_btn.setObjectName("accentBtn")
        self._save_nk_graph_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._save_nk_graph_btn.clicked.connect(self._save_nk_graph_image)
        bar.addWidget(self._save_nk_graph_btn)
        self._export_nk_graph_tab_btn = QPushButton("Export (Excel or CSV)")
        self._export_nk_graph_tab_btn.setObjectName("accentBtn")
        self._export_nk_graph_tab_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._export_nk_graph_tab_btn.setToolTip(self._export_btn.toolTip())
        self._export_nk_graph_tab_btn.clicked.connect(self._export_csv)
        bar.addWidget(self._export_nk_graph_tab_btn)
        layout.addLayout(bar)
        return page



    # ------------------------------------------------------------------
    # Requirements (footnotes and pop-ups)
    # ------------------------------------------------------------------

    def _set_calculation(self, key: str) -> None:
        """Step 1 choice: show steps 2-8 (absolute) or 2-11 (n1 and k1).

        Results are kept when switching; steps 9-11 are only hidden.
        """
        self.calculation = key
        self._calc_buttons[key].setChecked(True)
        self._apply_calculation_visibility()
        self._update_footnotes()
        if not self._has_results():
            self._refresh_table()

    def _apply_calculation_visibility(self) -> None:
        last_step = {None: 1, "absolute": 8, "nk": 11}[self.calculation]
        for number, group in enumerate(self._step_groups, start=1):
            group.setVisible(number <= last_step)
        # The n1 / k1 graph tab belongs to the n1 and k1 calculation only.
        nk_tab = self._results_tabs.indexOf(self._nk_plot_widget.parentWidget())
        self._results_tabs.setTabVisible(nk_tab, self.calculation == "nk")

    def _missing_measurements(self, modes: Optional[tuple[str, ...]] = None) -> list[str]:
        modes = modes or tuple(mode for mode, _angle in _ND_NK_MODES)
        return [
            f"{_display_mode(mode)} {slot}"
            for mode in modes
            for slot in _ND_NK_SLOTS
            if self._measurement(mode, slot) is None
        ]

    def _complete_modes(self) -> tuple[str, ...]:
        """R0 / R60 with all of Baseline, Standard and Sample available."""
        return tuple(
            mode
            for mode, _angle in _ND_NK_MODES
            if all(self._measurement(mode, slot) is not None for slot in _ND_NK_SLOTS)
        )

    def _planned_modes(self) -> tuple[str, ...]:
        """The modes step 2 would average now: both for n1 and k1; for absolute
        reflectance, every mode that has all three traces (R0, R60 or both)."""
        if self.calculation == "nk":
            return tuple(mode for mode, _angle in _ND_NK_MODES)
        return self._complete_modes()

    def _result_modes(self) -> tuple[str, ...]:
        """The modes the current results were calculated for."""
        if self.averages is not None:
            return tuple(self.averages["modes"])
        return ()

    @staticmethod
    def _modes_text(modes: tuple[str, ...]) -> str:
        return " and ".join(_display_mode(mode) for mode in modes)

    @staticmethod
    def _read_grid_fields(start: QLineEdit, end: QLineEdit, step: QLineEdit) -> tuple[float, float, float]:
        values = []
        for edit, name in ((start, "Start"), (end, "End"), (step, "Step")):
            text = edit.text().strip().replace(",", ".")
            try:
                value = float(text)
            except ValueError as exc:
                raise ValueError(f"{name} must be a number.") from exc
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number.")
            values.append(value)
        return values[0], values[1], values[2]

    def _reflectance_grid(self) -> np.ndarray:
        """Step 4 grid, limited to the measured range (as in the Original tab)."""
        start, end, step = self._read_grid_fields(self._refl_start, self._refl_end, self._refl_step)
        axis = self._axis()
        if axis is None:
            raise ValueError("No wavelength axis yet.")
        return _build_uniform_wavelengths(
            start, end, step, float(np.nanmin(axis)), float(np.nanmax(axis))
        )

    def _own_si_grid(self) -> np.ndarray:
        """Step 6 grid when not using the reflectance grid (not limited)."""
        start, end, step = self._read_grid_fields(self._si_start, self._si_end, self._si_step)
        if step <= 0:
            raise ValueError("Step must be positive.")
        if end < start:
            raise ValueError("End must be greater than or equal to Start.")
        grid = np.arange(start, end + step / 2.0, step, dtype=np.float64)
        if grid.size < 2:
            raise ValueError("The grid needs at least two points.")
        return grid

    @staticmethod
    def _problem(build: Callable[[], object]) -> Optional[str]:
        try:
            build()
        except ValueError as exc:
            return str(exc).rstrip(".") + "."
        return None

    def _requirements(self, step: str) -> list[str]:
        """What a step still needs, in plain words (empty list = ready)."""
        needed: list[str] = []
        if step == "average":
            if self.calculation is None:
                needed.append("A choice in step 1: what to calculate")
            elif self.calculation == "nk" and self._missing_measurements():
                needed.append(
                    "Measurements: " + ", ".join(self._missing_measurements())
                    + " (capture in the main window, or Import Measurement File)"
                )
            elif self.calculation == "absolute" and not self._complete_modes():
                missing = self._missing_measurements()
                needed.append(
                    "Baseline, Standard and Sample for R₀, R₆₀ or both ("
                    + ("" if len(missing) == 6 else "still missing: " + ", ".join(missing) + "; ")
                    + "capture in the main window, or Import Measurement File)"
                )
            elif self._axis() is None:
                needed.append("A wavelength axis (capture or import a trace)")
            else:
                shortest = min(
                    np.asarray(self._measurement(mode, slot)).size
                    for mode in self._planned_modes()
                    for slot in _ND_NK_SLOTS
                )
                if self._window_spin.value() > shortest:
                    needed.append(f"A window no longer than the traces ({shortest} points)")
        elif step == "relative":
            if self.averages is None:
                needed.append("Averaged traces (step 2: Average All Traces)")
        elif step == "interp_refl":
            if self.relative is None:
                needed.append("Relative reflectance (step 3)")
            if self._axis() is not None:
                problem = self._problem(self._reflectance_grid)
                if problem:
                    needed.append(f"A valid grid: {problem}")
        elif step == "interp_si":
            if self.si_wavelength is None:
                needed.append("Si values (step 5: Import Si Values)")
            if self._si_same_grid.isChecked():
                if self.relative_interp is None:
                    needed.append(
                        "The interpolated reflectance grid (step 4), or untick "
                        "'Use the reflectance grid' and enter a grid"
                    )
            else:
                problem = self._problem(self._own_si_grid)
                if problem:
                    needed.append(f"A valid Si grid: {problem}")
        elif step == "absolute":
            if self.relative_interp is None:
                needed.append("Interpolated reflectance (step 4)")
            if self.si_interp is None:
                needed.append("Interpolated Si values (step 6)")
            if (
                self.relative_interp is not None
                and self.si_interp is not None
                and not self._grids_match()
            ):
                needed.append(
                    "Steps 4 and 6 on the same wavelengths: tick 'Use the reflectance "
                    "grid' in step 6 and interpolate Si again"
                )
        elif step == "plot":
            if self.absolute is None:
                needed.append("Absolute reflectance (step 7)")
        elif step in ("nd_fit", "nk_fit"):
            if self.absolute is None:
                needed.append("Absolute reflectance (step 7)")
            elif any(mode not in self.absolute for mode, _angle in _ND_NK_MODES):
                needed.append(
                    "Absolute reflectance for both R₀ and R₆₀ (it was calculated for "
                    f"{self._modes_text(self._result_modes())} only: with both "
                    "measured, press step 2 again and redo steps 3–7)"
                )
        elif step == "save_graph":
            if not self.plotted:
                needed.append("A plotted graph (step 8: Plot Absolute Reflectance)")
        elif step == "plot_nk":
            if self.nk_fit is None:
                needed.append(
                    "Stored n₁ and k₁ (step 10: run the Nk fit, then press Finish in "
                    "the Nk Fit window)"
                )
        elif step == "save_nk_graph":
            if not self.nk_plotted:
                needed.append("A plotted graph (step 11: Plot n₁ and k₁)")
        elif step == "export":
            if self._model.columnCount() == 0:
                needed.append("Something to export: measurements, or imported Si values")
        return needed

    def _grids_match(self) -> bool:
        a = np.asarray(self.relative_interp["grid"])
        b = np.asarray(self.si_interp["grid"])
        return a.shape == b.shape and bool(np.allclose(a, b, rtol=0.0, atol=1e-9))

    def _update_footnotes(self, *_args: object) -> None:
        complete = self._complete_modes()
        if self.calculation is None:
            self._measurement_footnote.setText(
                f"<span style='color:{self._NEEDS_COLOR}'>Choose what to calculate "
                "above.</span>"
            )
        elif self.calculation == "nk":
            self._set_footnote(
                self._measurement_footnote,
                [
                    f"{name} (capture in the main window, or Import Measurement File)"
                    for name in self._missing_measurements()
                ],
                "All six traces are available.",
            )
        elif not complete:
            self._set_footnote(
                self._measurement_footnote,
                ["Baseline, Standard and Sample for R₀, R₆₀ or both (capture in the "
                 "main window, or Import Measurement File)"],
                "",
            )
        else:
            incomplete = [
                mode for mode, _angle in _ND_NK_MODES
                if mode not in complete and self._missing_measurements((mode,)) != [
                    f"{_display_mode(mode)} {slot}" for slot in _ND_NK_SLOTS
                ]
            ]
            text = f"{self._modes_text(complete)} complete: will be calculated."
            if incomplete:
                text += " " + "; ".join(
                    f"{_display_mode(mode)} is incomplete (missing "
                    + ", ".join(self._missing_measurements((mode,))) + ") and will be skipped"
                    for mode in incomplete
                ) + "."
            self._set_footnote(self._measurement_footnote, [], text)
        def ready(result: object, done_text: str, ready_text: str) -> str:
            return f"{done_text} Press again to redo." if result is not None else ready_text

        window_text = (
            f"Done for {self._modes_text(self._result_modes())} "
            f"(window {self.averages['window']} points)."
            if self.averages
            else ""
        )
        self._set_footnote(
            self._average_footnote,
            self._requirements("average"),
            ready(self.averages, window_text, "Ready to average."),
        )
        self._set_footnote(
            self._relative_footnote,
            self._requirements("relative"),
            ready(self.relative, "Done.", "Ready to calculate."),
        )
        grid_text = ""
        if self._axis() is not None and not self._problem(self._reflectance_grid):
            grid_text = f"Grid: {self._reflectance_grid().size} points. "
        self._set_footnote(
            self._interp_refl_footnote,
            self._requirements("interp_refl"),
            grid_text + ready(self.relative_interp, "Done.", "Ready to interpolate."),
        )
        if self.si_wavelength is None:
            self._set_footnote(
                self._si_footnote,
                ["Si values: click Import Si Values (an Excel or CSV file with "
                 "wavelength, n and k)"],
                "",
            )
        else:
            self._set_footnote(
                self._si_footnote,
                [],
                f"{self.si_wavelength.size} Si points from {self._si_source} "
                f"({np.nanmin(self.si_wavelength):.1f}–{np.nanmax(self.si_wavelength):.1f} nm).",
            )
        self._set_footnote(
            self._interp_si_footnote,
            self._requirements("interp_si"),
            ready(self.si_interp, "Done.", "Ready to interpolate."),
        )
        self._set_footnote(
            self._absolute_footnote,
            self._requirements("absolute"),
            ready(self.absolute, "Done.", "Ready to calculate."),
        )
        self._set_footnote(
            self._plot_footnote,
            self._requirements("plot"),
            f"Plotted (see the {_ABS_GRAPH_TAB} tab). Press again to redraw."
            if self.plotted
            else "Ready to plot.",
        )
        if self.nd_fit is not None:
            fit_text = (
                f"Last fit at {self.nd_fit['wavelength']:.3f} nm: n₁ = "
                f"{self.nd_fit['n1']:.6f}, d₁ = {self.nd_fit['d1']:.4f} nm."
            )
        else:
            fit_text = "Ready. Opens the Nd Fit window."
        self._set_footnote(self._nd_fit_footnote, self._requirements("nd_fit"), fit_text)
        if self._nd_fit_dialog is not None:
            self._nd_fit_dialog.refresh()
        nk_parts = []
        if self.nk_fit is not None:
            done = int(np.isfinite(self.nk_fit["n1"]).sum())
            nk_parts.append(f"Stored: {done} wavelengths (n₁ and k₁ are in the table).")
        if self.nk_segments:
            count = len(self.nk_segments)
            nk_parts.append(
                f"{count} run{'s' if count != 1 else ''} not stored yet (press Finish "
                "in the Nk Fit window)."
            )
        nk_text = " ".join(nk_parts) or "Ready. Opens the Nk Fit window."
        self._set_footnote(self._nk_fit_footnote, self._requirements("nk_fit"), nk_text)
        if self._nk_fit_dialog is not None:
            self._nk_fit_dialog.refresh()
        self._set_footnote(
            self._plot_nk_footnote,
            self._requirements("plot_nk"),
            f"Plotted (see the {_NK_GRAPH_TAB} tab). Press again to redraw."
            if self.nk_plotted
            else "Ready to plot.",
        )
        self._set_footnote(self._save_graph_footnote, self._requirements("save_graph"), "")
        self._set_footnote(
            self._save_nk_graph_footnote, self._requirements("save_nk_graph"), ""
        )
        export_needed = self._requirements("export")
        if export_needed:
            self._set_footnote(self._export_footnote, export_needed, "")
        else:
            self._export_footnote.setText("")

    def _show_requirements_popup(self, title: str, needed: list[str]) -> None:
        QMessageBox.information(
            self,
            title,
            "This step needs:\n\n" + "\n".join(f"  •  {item}" for item in needed),
        )

    # ------------------------------------------------------------------
    # Measurements (always read live from the main window)
    # ------------------------------------------------------------------

    def _measurement(self, mode: str, slot: str) -> Optional[np.ndarray]:
        return self._main.spectra_data.get(mode, {}).get(slot)

    def _measurement_inputs(self) -> tuple[Optional[np.ndarray], ...]:
        return tuple(
            self._measurement(mode, slot)
            for mode, _angle in _ND_NK_MODES
            for slot in _ND_NK_SLOTS
        )

    def _axis(self) -> Optional[np.ndarray]:
        axis = self._main._wavelengths
        if axis is None:
            axis = self._main._latest_wavelengths
        return None if axis is None else np.asarray(axis, dtype=np.float64)

    def _has_results(self) -> bool:
        return any(
            value is not None
            for value in (
                self.averages, self.relative, self.relative_interp, self.si_interp, self.absolute
            )
        )

    def sync_measurements(self) -> None:
        """Update the measurement status and table; clear results if R0/R60 changed.

        Called whenever the main window refreshes its status bar, which it
        does after every capture, import and reset.
        """
        for mode, _angle in _ND_NK_MODES:
            parts = []
            for slot in _ND_NK_SLOTS:
                present = self._measurement(mode, slot) is not None
                mark = "✓" if present else "–"
                color = self._READY_COLOR if present else "#6d7480"
                parts.append(f"{slot} <span style='color:{color}'>{mark}</span>")
            self._measurement_labels[mode].setText(
                f"<b>{_display_mode(mode)}</b>: " + "&nbsp;&nbsp;".join(parts)
            )

        current = self._measurement_inputs()
        if self._results_inputs is not None and self._has_results():
            changed = [
                f"{_display_mode(mode)} {slot}"
                for (mode, slot), old, new in zip(
                    (
                        (mode, slot)
                        for mode, _angle in _ND_NK_MODES
                        for slot in _ND_NK_SLOTS
                    ),
                    self._results_inputs,
                    current,
                )
                if old is not new
            ]
            if changed:
                self._clear_from("measurements")
                self._refresh_table()
                self._summary.setText(
                    "Measurements changed (" + ", ".join(changed) + ") — all "
                    "calculations cleared. Start again from step 2."
                )
                return
        if self._shown_inputs is None or any(
            old is not new for old, new in zip(self._shown_inputs, current)
        ):
            self._refresh_table()
        else:
            self._update_footnotes()

    def _clear_from(self, step: str) -> None:
        """Clear the results of every step that depends on ``step``."""
        if step == "measurements":
            self.averages = None
            self.si_interp = None
            self._results_inputs = None
        if step in ("measurements", "average"):
            self.relative = None
        if step in ("measurements", "average", "relative"):
            self.relative_interp = None
        if step == "import_si":
            self.si_interp = None
        if step != "plot":
            self.absolute = None
            had_fit = self.nd_fit is not None
            self.nd_fit = None  # the Nd fit used step 7's values
            if self._nd_fit_dialog is not None:
                self._nd_fit_dialog.results_cleared(had_fit)
            had_nk = (
                self.nk_fit is not None
                or bool(self.nk_segments)
                or (self._nk_fit_dialog is not None and self._nk_fit_dialog._run is not None)
            )
            self.nk_fit = None  # so did the Nk runs and their stored result
            self.nk_segments = []
            self._clear_nk_plot()
            if self._nk_fit_dialog is not None:
                self._nk_fit_dialog.results_cleared(had_nk)
        self.plotted = False
        self._plot_widget.clear()

    def _nk_d1_in_use(self) -> Optional[float]:
        """d1 used by the Nk run in progress, the kept runs or the stored result."""
        dialog = self._nk_fit_dialog
        if dialog is not None and dialog._run is not None:
            return float(dialog._run["d1"])
        if self.nk_segments:
            return float(self.nk_segments[0]["d1"])
        if self.nk_fit is not None:
            return float(self.nk_fit["d1"])
        return None

    def nd_d1_changed(self, old_d1: float, new_d1: float) -> None:
        """A new Nd fit gave another d1: the Nk inputs changed, so its runs
        (also one in progress) and its stored result are cleared."""
        self.nk_fit = None
        self.nk_segments = []
        self._clear_nk_plot()
        if self._nk_fit_dialog is not None:
            self._nk_fit_dialog.results_cleared(
                True,
                f"d₁ changed in the Nd fit ({old_d1:.4f} → {new_d1:.4f} nm): Nk runs "
                "and stored n₁, k₁ cleared.",
            )
        self._refresh_table()
        self._update_footnotes()

    def showEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        super().showEvent(event)
        self._load_si_from_post_process_if_empty()
        self.sync_measurements()

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        # Only hide: the main window keeps this window, so nothing is lost.
        event.accept()

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def _on_step_clicked(self, step: str, title: str) -> None:
        needed = self._requirements(step)
        if needed:
            self._show_requirements_popup("Not Ready Yet", needed)
            return
        action = {
            "average": self._average_all,
            "relative": self._calculate_relative,
            "interp_refl": self._interpolate_reflectance,
            "interp_si": self._interpolate_si,
            "absolute": self._calculate_absolute,
        }[step]
        before = self._revision
        popup = ProgressPopup(title, parent=self)
        try:
            popup.run_handler(action, lambda: self._revision != before)
        finally:
            popup.deleteLater()

    def _step_done(self, message: str) -> bool:
        self._revision += 1
        if self._results_inputs is None:
            # Any calculated result is cleared when the measurements change.
            self._results_inputs = self._measurement_inputs()
        self._refresh_table()
        self._summary.setText(message)
        return True

    def _average_all(self) -> bool:
        """Step 2: centred moving average of the traces of every mode in use."""
        axis = self._axis()
        window = int(self._window_spin.value())
        inputs = self._measurement_inputs()
        modes = self._planned_modes()
        averages: dict[object, object] = {"window": window, "modes": modes}
        try:
            for mode in modes:
                for slot in _ND_NK_SLOTS:
                    trace = np.asarray(self._measurement(mode, slot), dtype=np.float64)
                    if trace.shape != axis.shape:
                        raise ValueError(
                            f"{_display_mode(mode)} {slot} has {trace.size} points but the "
                            f"wavelength axis has {axis.size}."
                        )
                    averages[(mode, slot)] = _moving_average_centered(trace, window)
        except ValueError as exc:
            QMessageBox.warning(self, "Averaging Failed", str(exc))
            return False
        self._clear_from("average")
        self.averages = averages
        self._results_inputs = inputs
        return self._step_done(
            f"Averaged the {self._modes_text(modes)} traces ({3 * len(modes)} traces, "
            f"window {window} points)."
        )

    def _calculate_relative(self) -> bool:
        """Step 3: relative reflectance on the detector axis."""
        relative = {
            mode: _reflectance_ratio(
                self.averages[(mode, CaptureMode.SAMPLE.value)],
                self.averages[(mode, CaptureMode.BASELINE.value)],
                self.averages[(mode, CaptureMode.STANDARD.value)],
            )
            for mode in self._result_modes()
        }
        self._clear_from("relative")
        self.relative = relative
        counts = ", ".join(
            f"{_display_mode(mode)} {int(np.isfinite(values).sum())} points"
            for mode, values in relative.items()
        )
        return self._step_done(
            f"Relative reflectance calculated: {counts} (edge points are N/A after "
            "averaging)."
        )

    def _interpolate_reflectance(self) -> bool:
        """Step 4: resample relative reflectance onto the grid."""
        axis = self._axis()
        method = self._refl_method_combo.currentText()
        try:
            grid = self._reflectance_grid()
            result: dict[str, object] = {"grid": grid, "method": method}
            for mode in self._result_modes():
                result[mode] = _interpolate_reflectance_series(
                    axis, self.relative[mode], grid, method
                )
        except ValueError as exc:
            QMessageBox.warning(self, "Interpolation Failed", str(exc))
            return False
        self._clear_from("interp_refl")
        self.relative_interp = result
        note = ""
        if (
            self.si_interp is not None
            and self.si_interp.get("same_grid")
            and not self._grids_match()
        ):
            self.si_interp = None
            note = " The grid changed, so the interpolated Si values were cleared (redo step 6)."
        return self._step_done(
            f"Relative reflectance interpolated onto {grid.size} points "
            f"({grid[0]:.3f}–{grid[-1]:.3f} nm, step {grid[1] - grid[0]:.3f} nm, {method}).{note}"
        )

    def _interpolate_si(self) -> bool:
        """Step 6: resample Si n and k onto the grid."""
        same = self._si_same_grid.isChecked()
        method = self._si_method_combo.currentText()
        try:
            grid = (
                np.asarray(self.relative_interp["grid"]) if same else self._own_si_grid()
            )
            n_values = _interpolate_si_series(self.si_wavelength, self.si_n, grid, method)
            k_values = _interpolate_si_series(self.si_wavelength, self.si_k, grid, method)
        except ValueError as exc:
            QMessageBox.warning(self, "Interpolation Failed", str(exc))
            return False
        self._clear_from("interp_si")
        self.si_interp = {
            "grid": grid.copy(),
            "n": n_values,
            "k": k_values,
            "method": method,
            "same_grid": same,
        }
        where = "the reflectance grid" if same else "its own grid"
        return self._step_done(
            f"Si n and k interpolated onto {where}: {grid.size} points "
            f"({grid[0]:.3f}–{grid[-1]:.3f} nm, {method})."
        )

    def _calculate_absolute(self) -> bool:
        """Step 7: interpolated relative reflectance x bare-Si reflectance."""
        grid = np.asarray(self.relative_interp["grid"])
        result: dict[str, object] = {"grid": grid}
        angles = dict(_ND_NK_MODES)
        for mode in self._result_modes():
            angle = angles[mode]
            # Same method as the lab's Excel sheet (Si_Std_Fit_Full): every step
            # rounded to 4 decimals, with true pi (thin_film.PI). See thin_film.py.
            si_r = thin_film.si_wafer_reflectance_excel(
                self.si_interp["n"], self.si_interp["k"], angle
            )
            result[f"si_r_{mode}"] = si_r
            result[mode] = np.asarray(self.relative_interp[mode]) * si_r
        self._clear_from("absolute")
        self.absolute = result
        modes = self._result_modes()
        finite = np.ones(grid.size, dtype=bool)
        for mode in modes:
            finite &= np.isfinite(np.asarray(result[mode]))
        return self._step_done(
            f"Absolute reflectance ({self._modes_text(modes)}) calculated on {grid.size} "
            f"points ({int(finite.sum())} with a value). Next: step 8, Plot."
        )

    def _on_plot_clicked(self) -> None:
        """Step 8: plot absolute reflectance vs wavelength (its graph tab)."""
        needed = self._requirements("plot")
        if needed:
            self._show_requirements_popup("Not Ready Yet", needed)
            return
        self._plot_widget.clear()
        grid = np.asarray(self.absolute["grid"])
        shown = [(mode, angle) for mode, angle in _ND_NK_MODES if mode in self.absolute]
        curves = [
            (mode, f"{_display_mode(mode)} absolute ({angle:g}°{', s-pol' if angle else ''})",
             np.asarray(self.absolute[mode]), Qt.PenStyle.SolidLine, 1)
            for mode, angle in shown
        ] + [
            (mode, f"Si {_display_mode(mode)} ({angle:g}°{', s-pol' if angle else ''})",
             np.asarray(self.absolute[f"si_r_{mode}"]), Qt.PenStyle.DashLine, 0)
            for mode, angle in shown
        ]
        for mode, name, values, style, z in curves:
            curve = self._plot_widget.plot(
                grid,
                values,
                pen=pg.mkPen(color=self._PLOT_COLORS[mode], width=2, style=style),
                name=name,
                connect="finite",
            )
            curve.setZValue(z)  # sample curves on top of the Si curves
            # Refit the axes when a curve is hidden or shown from the legend.
            curve.visibleChanged.connect(self._refit_plot)
        self._plot_widget.enableAutoRange()
        self.plotted = True
        self._results_tabs.setCurrentIndex(1)
        self._update_footnotes()
        self._summary.setText(
            f"Absolute reflectance and bare-Si reflectance plotted ({_ABS_GRAPH_TAB} tab)."
        )

    def _refit_plot(self) -> None:
        """Fit the axes to the curves that are visible (after a legend click)."""
        view = self._plot_widget.getViewBox()
        view.enableAutoRange()
        view.updateAutoRange()

    def _clear_nk_plot(self) -> None:
        """Remove step 11's curves (the stored n1 / k1 changed or was cleared)."""
        item = self._nk_plot_widget.getPlotItem()
        for curve in self._nk_plot_curves:
            self._nk_plot_legend.removeItem(curve)
            item.removeItem(curve)
        self._nk_plot_curves = []
        self.nk_plotted = False

    @staticmethod
    def _with_gaps(wavelengths: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Drop missing values and break the line where wavelengths are missing
        (runs kept as separate segments), so no line is drawn across a gap."""
        ok = np.isfinite(values) & np.isfinite(wavelengths)
        x = np.asarray(wavelengths, dtype=np.float64)[ok]
        y = np.asarray(values, dtype=np.float64)[ok]
        if x.size < 3:
            return x, y
        steps = np.diff(x)
        breaks = np.flatnonzero(steps > 1.5 * float(np.median(steps)))
        if breaks.size:
            x = np.insert(x, breaks + 1, np.nan)
            y = np.insert(y, breaks + 1, np.nan)
        return x, y

    def _on_plot_nk_clicked(self) -> None:
        """Step 11: plot the stored n1 and k1 vs wavelength (n1 / k1 tab)."""
        needed = self._requirements("plot_nk")
        if needed:
            self._show_requirements_popup("Not Ready Yet", needed)
            return
        self._clear_nk_plot()
        item = self._nk_plot_widget.getPlotItem()
        fit = self.nk_fit
        wavelengths = np.asarray(fit["wavelength"], dtype=np.float64)
        for key, name in (("n1", "n₁"), ("k1", "k₁")):
            x, y = self._with_gaps(wavelengths, np.asarray(fit[key], dtype=np.float64))
            curve = pg.PlotDataItem(
                x, y, pen=pg.mkPen(color=_NK_PLOT_COLORS[key], width=2), connect="finite"
            )
            item.addItem(curve)  # (no name: the legend entry is added below)
            self._nk_plot_legend.addItem(curve, name)
            # Refit the axes when a curve is hidden or shown from the legend.
            curve.visibleChanged.connect(self._refit_nk_plot)
            self._nk_plot_curves.append(curve)
        self._refit_nk_plot()
        self.nk_plotted = True
        self._results_tabs.setCurrentIndex(
            self._results_tabs.indexOf(self._nk_plot_widget.parentWidget())
        )
        self._update_footnotes()
        self._summary.setText(f"n₁ and k₁ plotted ({_NK_GRAPH_TAB} tab).")

    def _refit_nk_plot(self) -> None:
        """Fit the axes to the visible curves (after a legend click)."""
        view = self._nk_plot_widget.getPlotItem().vb
        view.enableAutoRange()
        view.updateAutoRange()

    def _save_nk_graph_image(self) -> None:
        needed = self._requirements("save_nk_graph")
        if needed:
            self._show_requirements_popup("Nothing to Save", needed)
            return
        default_name = f"n1_k1_vs_wavelength_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        path_str, selected = QFileDialog.getSaveFileName(
            self,
            "Save Graph Image",
            str(Path.home() / default_name),
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg)",
        )
        if not path_str:
            return
        path = Path(path_str)
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            path = path.with_suffix(".jpg" if "JPEG" in (selected or "") else ".png")
        try:
            ImageExporter(self._nk_plot_widget.plotItem).export(str(path))
        except Exception as exc:
            QMessageBox.critical(self, "Save Failed", str(exc))
            return
        QMessageBox.information(self, "Graph Saved", f"Saved graph image to:\n{path}")

    def _save_graph_image(self) -> None:
        needed = self._requirements("save_graph")
        if needed:
            self._show_requirements_popup("Nothing to Save", needed)
            return
        default_name = f"absolute_reflectance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        path_str, selected = QFileDialog.getSaveFileName(
            self,
            "Save Graph Image",
            str(Path.home() / default_name),
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg)",
        )
        if not path_str:
            return
        path = Path(path_str)
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            path = path.with_suffix(".jpg" if "JPEG" in (selected or "") else ".png")
        try:
            ImageExporter(self._plot_widget.plotItem).export(str(path))
        except Exception as exc:
            QMessageBox.critical(self, "Save Failed", str(exc))
            return
        QMessageBox.information(self, "Graph Saved", f"Saved graph image to:\n{path}")

    # ------------------------------------------------------------------
    # Silicon reference (step 5)
    # ------------------------------------------------------------------

    def _set_si_values(
        self, wavelengths: np.ndarray, n_values: np.ndarray, k_values: np.ndarray, source: str
    ) -> None:
        self.si_wavelength = np.asarray(wavelengths, dtype=np.float64).copy()
        self.si_n = np.asarray(n_values, dtype=np.float64).copy()
        self.si_k = np.asarray(k_values, dtype=np.float64).copy()
        self._si_source = source

    def _load_si_from_post_process_if_empty(self) -> None:
        """Reuse Si values imported in Post Process if this window has none."""
        if self.si_wavelength is not None:
            return
        wavelengths = getattr(self._main, "si_wavelength", None)
        n_values = getattr(self._main, "si_n", None)
        k_values = getattr(self._main, "si_k", None)
        if wavelengths is None or n_values is None or k_values is None:
            return
        self._set_si_values(wavelengths, n_values, k_values, "Post Process")
        self._refresh_table()

    def _import_si_values(self) -> None:
        path_str, _selected = QFileDialog.getOpenFileName(
            self,
            "Import Si Values",
            str(Path.home()),
            "Excel/CSV Files (*.xlsx *.xls *.csv);;All Files (*)",
        )
        if not path_str:
            return
        try:
            wavelengths, n_values, k_values = parse_silicon_reference(Path(path_str))
        except MeasurementCsvError as exc:
            QMessageBox.warning(self, "Import Failed", str(exc))
            return
        had_results = self.si_interp is not None or self.absolute is not None
        self._set_si_values(wavelengths, n_values, k_values, Path(path_str).name)
        self._clear_from("import_si")
        self._refresh_table()
        message = f"Loaded {wavelengths.size} Si reference points from {Path(path_str).name}."
        if had_results:
            message += " Steps 6–8 were cleared because the Si values changed."
        self._summary.setText(message)

    def _import_measurements(self) -> None:
        """Load R0 / R60 Baseline, Standard and Sample traces from a file.

        The file is read by the original parser (the Excel import block makes
        it accept Excel too) and stored by the main window's own
        _store_imported_dataset, exactly as Import CSV Data stores it: same
        wavelength-axis check, same "Replace Measurements" question, and the
        main window's plot and status update too. Only R0 and R60 columns are
        taken; T columns are left alone. The main window's status refresh then
        calls sync_measurements, which clears the calculations.
        """
        path_str, _selected = QFileDialog.getOpenFileName(
            self,
            "Import Measurement File",
            str(Path.home()),
            "CSV or Excel Files (*.csv *.xlsx *.xlsm *.xls);;All Files (*)",
        )
        if not path_str:
            return
        name = Path(path_str).name
        try:
            wavelengths, columns = parse_unified_measurement_csv(Path(path_str))
        except MeasurementCsvError as exc:
            QMessageBox.warning(self, "Import Failed", str(exc))
            return

        selected: dict[str, dict[str, np.ndarray]] = {}
        for mode, _angle in _ND_NK_MODES:
            slots = {
                slot: values
                for slot, values in columns.get(mode, {}).items()
                if slot in _ND_NK_SLOTS
            }
            if slots:
                selected[mode] = slots
        unlabelled = {
            slot: values
            for slot, values in columns.get(None, {}).items()
            if slot in _ND_NK_SLOTS
        }
        if unlabelled:
            mode = self._ask_unlabelled_mode(list(unlabelled), name)
            if mode is None:
                return
            target = selected.setdefault(mode, {})
            for slot, values in unlabelled.items():
                target.setdefault(slot, values)  # a labelled column wins, as in the original
        nd_nk_modes = {mode for mode, _angle in _ND_NK_MODES}
        skipped = [
            _display_mode(key) for key in columns if key is not None and key not in nd_nk_modes
        ]
        skipped_text = (
            " " + ", ".join(skipped) + " columns were skipped (this window only uses "
            "R₀ and R₆₀; use Import CSV Data in the main window for those)."
            if skipped
            else ""
        )
        if not selected:
            QMessageBox.warning(
                self,
                "Nothing to Import",
                f"No R₀ or R₆₀ Baseline, Standard or Sample columns were found in {name}."
                + skipped_text,
            )
            return

        loaded = self._main._store_imported_dataset(wavelengths, selected)
        if not loaded:
            return  # the main window already explained why (or the user said No)
        QMessageBox.information(
            self,
            "Import Complete",
            f"Loaded {', '.join(loaded)} from {name} ({wavelengths.size} points)."
            + skipped_text,
        )

    def _ask_unlabelled_mode(self, slots: list[str], file_name: str) -> Optional[str]:
        """Ask whether columns without an R0_ / R60_ prefix are R0 or R60."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Which Measurement?")
        box.setText(
            f"The {', '.join(slots)} column(s) in {file_name} don't say whether "
            "they are R₀ or R₆₀."
        )
        box.setInformativeText("Load them as:")
        r0_btn = box.addButton("R₀ (0°)", QMessageBox.ButtonRole.AcceptRole)
        r60_btn = box.addButton("R₆₀ (60°)", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked is r0_btn:
            return "R0"
        if clicked is r60_btn:
            return "R60"
        return None

    # ------------------------------------------------------------------
    # Step 9: Nd fit
    # ------------------------------------------------------------------

    def _open_nd_fit(self) -> None:
        needed = self._requirements("nd_fit")
        if needed:
            self._show_requirements_popup("Not Ready Yet", needed)
            return
        if self._nd_fit_dialog is None:
            self._nd_fit_dialog = NdFitDialog(self)
        self._nd_fit_dialog.refresh()
        _show_maximized(self._nd_fit_dialog)

    def _open_nk_fit(self) -> None:
        needed = self._requirements("nk_fit")
        if needed:
            self._show_requirements_popup("Not Ready Yet", needed)
            return
        if self._nk_fit_dialog is None:
            self._nk_fit_dialog = NkFitDialog(self)
        self._nk_fit_dialog.refresh()
        _show_maximized(self._nk_fit_dialog)

    def nd_fit_values(self, wavelength: float) -> dict[str, object]:
        """R0, R60 (step 7) and Si n2, k2 (step 6) at one wavelength.

        Taken straight from the table when the wavelength is on the step 4 grid;
        otherwise linearly interpolated between the two grid points either side.
        Raises ValueError (plain-words message) when that is not possible.
        """
        if self.absolute is None:
            raise ValueError("Absolute reflectance is not calculated yet (step 7).")
        grid = np.asarray(self.absolute["grid"], dtype=np.float64)
        series = {
            "r0": np.asarray(self.absolute["R0"], dtype=np.float64),
            "r60": np.asarray(self.absolute["R60"], dtype=np.float64),
            "n2": np.asarray(self.si_interp["n"], dtype=np.float64),
            "k2": np.asarray(self.si_interp["k"], dtype=np.float64),
        }
        if not np.isfinite(wavelength) or wavelength < grid[0] - 1e-9 or wavelength > grid[-1] + 1e-9:
            raise ValueError(
                f"Pick a wavelength between {grid[0]:.3f} and {grid[-1]:.3f} nm "
                "(the step 4 grid)."
            )
        hit = np.flatnonzero(np.abs(grid - wavelength) <= 1e-6)
        if hit.size:
            index = int(hit[0])
            values = {key: float(array[index]) for key, array in series.items()}
            source = f"from the table at {grid[index]:.3f} nm"
        else:
            right = int(np.searchsorted(grid, wavelength))
            left = right - 1
            values = {
                key: float(np.interp(wavelength, grid[left:right + 1], array[left:right + 1]))
                for key, array in series.items()
            }
            source = (
                f"interpolated between the table rows at {grid[left]:.3f} and "
                f"{grid[right]:.3f} nm"
            )
        missing = [
            name for key, name in (("r0", "R₀"), ("r60", "R₆₀"), ("n2", "Si n"), ("k2", "Si k"))
            if not np.isfinite(values[key])
        ]
        if missing:
            raise ValueError(
                f"{', '.join(missing)} is N/A at {wavelength:g} nm in the table; pick a "
                "wavelength inside the measured range."
            )
        values["source"] = source
        return values

    # ------------------------------------------------------------------
    # Table and export
    # ------------------------------------------------------------------

    def _refresh_table(self) -> None:
        """Columns, left to right, in workflow order."""
        columns: list[tuple[str, np.ndarray]] = []
        axis = self._axis()
        present = [
            (mode, slot)
            for mode, _angle in _ND_NK_MODES
            for slot in _ND_NK_SLOTS
            if self._measurement(mode, slot) is not None
        ]
        if axis is not None and present:
            columns.append(("Wavelength (nm)", axis))
            for mode, slot in present:  # 1. measurements
                columns.append((f"{mode}_{slot} (counts)", self._measurement(mode, slot)))
            if self.averages is not None:  # 2. averages
                window = self.averages["window"]
                for mode in self._result_modes():
                    for slot in _ND_NK_SLOTS:
                        columns.append(
                            (f"{mode}_{slot}_Avg_W{window} (counts)", self.averages[(mode, slot)])
                        )
            if self.relative is not None:  # 3. relative reflectance
                for mode in self.relative:
                    columns.append((f"{mode} Relative Reflectance", self.relative[mode]))
        if self.relative_interp is not None:  # 4. interpolated reflectance
            columns.append(("Interp Wavelength (nm)", self.relative_interp["grid"]))
            for mode in self._result_modes():
                columns.append(
                    (f"{mode} Interpolated Relative Reflectance", self.relative_interp[mode])
                )
        if self.si_wavelength is not None:  # 5. imported Si values (right of step 4)
            columns += [
                ("Imported Si Wavelength (nm)", self.si_wavelength),
                ("Imported Si n", self.si_n),
                ("Imported Si k", self.si_k),
            ]
        if self.si_interp is not None:  # 6. interpolated Si values
            columns += [
                ("Si Interp Wavelength (nm)", self.si_interp["grid"]),
                ("Interpolated Si n", self.si_interp["n"]),
                ("Interpolated Si k", self.si_interp["k"]),
            ]
        if self.absolute is not None:  # 7. absolute reflectance
            si_headers = {"R0": "Si R0 (0°)", "R60": "Si R60 (60°, s-pol)"}
            modes = [mode for mode, _angle in _ND_NK_MODES if mode in self.absolute]
            columns += [(si_headers[mode], self.absolute[f"si_r_{mode}"]) for mode in modes]
            columns += [
                (f"{mode} Absolute Reflectance", self.absolute[mode]) for mode in modes
            ]
        if self.nk_fit is not None:  # 10. Nk fit (d1 from the Nd fit: one value, one cell)
            columns += [
                ("d1 (nm)", np.array([float(self.nk_fit["d1"])], dtype=np.float64)),
                ("Nk Wavelength (nm)", self.nk_fit["wavelength"]),
                ("Fitted n1", self.nk_fit["n1"]),
                ("Fitted k1", self.nk_fit["k1"]),
            ]
        self._model.set_columns(columns)
        self._shown_inputs = self._measurement_inputs()
        if not self._has_results():
            self._summary.setText(
                "Choose what to calculate in step 1, then follow the steps in order. "
                "Measurements come from the main window's captures or from Import "
                "Measurement File in step 1."
                if self.calculation is None
                else "Follow the steps in order. Measurements come from the main "
                "window's captures or from Import Measurement File in step 1."
            )
        self._update_footnotes()

    def _export_csv(self) -> None:
        """Export the table: Excel (table + graph + settings / fits) or CSV (table)."""
        needed = self._requirements("export")
        if needed:
            self._show_requirements_popup("Nothing to Export", needed)
            return
        excel_filter = "Excel workbook: table, graph, settings and fits (*.xlsx)"
        csv_filter = "CSV file: table only (*.csv)"
        default_name = f"post_process_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        path_str, selected = QFileDialog.getSaveFileName(
            self,
            "Export Post Process Results",
            str(Path.home() / default_name),
            f"{excel_filter};;{csv_filter}",
        )
        if not path_str:
            return
        path = Path(path_str)
        suffix = path.suffix.lower()
        if suffix not in (".xlsx", ".csv"):
            suffix = ".csv" if selected == csv_filter else ".xlsx"
            path = path.with_name(path.name + suffix)
        try:
            if suffix == ".csv":
                headers, rows = self._model.export_rows()
                with path.open("w", newline="", encoding="utf-8-sig") as handle:
                    handle.write("sep=,\n")
                    writer = csv.writer(handle)
                    writer.writerow(headers)
                    writer.writerows(rows)
                message = f"Saved the table ({len(rows)} rows) to:\n{path}"
            else:
                columns = list(self._model._columns)
                charts = []
                chart = self._export_chart_spec(columns)
                if chart is not None:
                    charts.append((_ABS_GRAPH_SHEET, chart))
                nk_chart = self._export_nk_chart_spec(columns)
                if nk_chart is not None:
                    charts.append((_NK_GRAPH_SHEET, nk_chart))
                _write_results_workbook(path, columns, charts, self._export_info_rows())
                message = (
                    f"Saved to:\n{path}\n\nSheets: Data (the table)"
                    + "".join(f", {name} (chart from the Data sheet)" for name, _c in charts)
                    + ", Settings and fits."
                )
        except OSError as exc:
            QMessageBox.critical(self, "Export Failed", str(exc))
            return
        QMessageBox.information(self, "Export Complete", message)

    def _export_chart_spec(
        self, columns: list[tuple[str, np.ndarray]]
    ) -> Optional[dict[str, object]]:
        """The Graph tab's four curves as an Excel chart (needs step 7)."""
        if self.absolute is None:
            return None
        headers = [header for header, _values in columns]
        names = {
            "x": "Interp Wavelength (nm)",
            "R0": "R0 Absolute Reflectance",
            "R60": "R60 Absolute Reflectance",
            "si_R0": "Si R0 (0°)",
            "si_R60": "Si R60 (60°, s-pol)",
        }
        if names["x"] not in headers:
            return None
        grid = np.asarray(self.absolute["grid"])
        colors = {mode: "%02X%02X%02X" % self._PLOT_COLORS[mode] for mode in ("R0", "R60")}
        series = []
        for key, label, mode, dashed in (
            ("R0", "R₀ absolute (0°)", "R0", False),
            ("R60", "R₆₀ absolute (60°, s-pol)", "R60", False),
            ("si_R0", "Si R₀ (0°)", "R0", True),
            ("si_R60", "Si R₆₀ (60°, s-pol)", "R60", True),
        ):
            if names[key] not in headers:
                continue  # that mode was not calculated
            series.append(
                {
                    "name": label, "color": colors[mode], "dashed": dashed,
                    "x_column": headers.index(names["x"]),
                    "y_column": headers.index(names[key]),
                    "length": grid.size,
                }
            )
        if not series:
            return None
        return {
            "title": "Absolute reflectance and bare-Si reflectance vs wavelength",
            "x_title": "Wavelength (nm)",
            "y_title": "Reflectance",
            "x_min": float(grid[0]),
            "x_max": float(grid[-1]),
            "series": series,
            "note": f"Drawn from the Data sheet: the same curves as the app's {_ABS_GRAPH_TAB} tab.",
        }

    def _export_nk_chart_spec(
        self, columns: list[tuple[str, np.ndarray]]
    ) -> Optional[dict[str, object]]:
        """The stored n1 and k1 (same axis) as an Excel chart."""
        if self.nk_fit is None:
            return None
        headers = [header for header, _values in columns]
        if not all(h in headers for h in ("Nk Wavelength (nm)", "Fitted n1", "Fitted k1")):
            return None
        wavelengths = np.asarray(self.nk_fit["wavelength"], dtype=np.float64)
        # Blocks of rows without a wavelength gap (runs kept as separate
        # segments): one series each, so Excel draws no line across a gap.
        blocks = [(0, wavelengths.size)]
        if wavelengths.size >= 3:
            steps = np.diff(wavelengths)
            cuts = (np.flatnonzero(steps > 1.5 * float(np.median(steps))) + 1).tolist()
            edges = [0] + cuts + [wavelengths.size]
            blocks = [(a, b - a) for a, b in zip(edges[:-1], edges[1:])]
        series = [
            {
                "name": name, "color": "%02X%02X%02X" % _NK_PLOT_COLORS[key], "dashed": False,
                "x_column": headers.index("Nk Wavelength (nm)"),
                "y_column": headers.index(column),
                "first": first,
                "length": count,
                "legend": block == 0,
            }
            for key, name, column in (("n1", "n₁", "Fitted n1"), ("k1", "k₁", "Fitted k1"))
            for block, (first, count) in enumerate(blocks)
        ]
        return {
            "title": "n₁ and k₁ vs wavelength",
            "x_title": "Wavelength (nm)",
            "y_title": "n₁ and k₁",
            "x_min": float(np.nanmin(wavelengths)),
            "x_max": float(np.nanmax(wavelengths)),
            "series": series,
            "note": f"Drawn from the Data sheet: the same curves as the app's {_NK_GRAPH_TAB} tab.",
        }

    def _export_info_rows(self) -> list[tuple[object, object, bool]]:
        """Settings and fit results for the "Settings and fits" sheet."""
        rows: list[tuple[object, object, bool]] = [
            ("Post Process results", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), True),
            (
                "Calculation (step 1)",
                {"absolute": "Absolute reflectance", "nk": "n₁ and k₁"}.get(
                    self.calculation or "", "not chosen"
                ),
                False,
            ),
            ("Modes", self._modes_text(self._result_modes()) or "none yet", False),
            ("π used", "true π (3.14159265…)", False),
            ("", "", False),
            ("Calculation settings", "", True),
        ]
        if self.averages:
            rows.append(("Step 2: averaging window (points)", self.averages["window"], False))
        if self.relative_interp is not None:
            grid = np.asarray(self.relative_interp["grid"])
            rows += [
                ("Step 4: grid start (nm)", float(grid[0]), False),
                ("Step 4: grid end (nm)", float(grid[-1]), False),
                ("Step 4: grid points", int(grid.size), False),
                ("Step 4: method", self.relative_interp["method"], False),
            ]
        if self.si_interp is not None:
            rows.append(("Step 6: Si interpolation method", self.si_interp["method"], False))
        if self.absolute is not None:
            rows.append(
                (
                    "Step 7: bare-Si reflectance",
                    "lab Excel sheet method (each step rounded to 4 decimals), with π",
                    False,
                )
            )
        fit = self.nd_fit
        if fit is not None:
            rows += [
                ("", "", False),
                ("Nd fit (step 9)", "", True),
                ("Wavelength (nm)", fit["wavelength"], False),
                ("R₀, R₆₀ and Si n₂, k₂", fit.get("source", "from the table"), False),
                ("n₁ start", fit["n1_start"], False),
                ("n₁ range (Δn₁)", fit["n1_range"], False),
                ("k₁", fit["k1"], False),
                ("d₁ start (nm)", fit["d1_start"], False),
                ("d₁ range (Δd₁, nm)", fit["d1_range"], False),
                ("Si n₂", fit["n2"], False),
                ("Si k₂", fit["k2"], False),
                ("Result: n₁", fit["n1"], True),
                ("Result: d₁ (nm)", fit["d1"], True),
                ("Error (ΔR₆₀² + ΔR₀²)", fit["error"], False),
                ("R₆₀ measured", fit["r60_measured"], False),
                ("R₆₀ fitted", fit["r60_fit"], False),
                ("R₀ measured", fit["r0_measured"], False),
                ("R₀ fitted", fit["r0_fit"], False),
            ]
        nk = self.nk_fit
        if nk is not None:
            wavelengths = np.asarray(nk["wavelength"])
            segment_of = np.asarray(nk["segment"])
            rows += [
                ("", "", False),
                ("Nk fit (step 10)", "", True),
                (
                    "Stored as",
                    "Complete (all runs merged)"
                    if nk["mode"] == "complete"
                    else "Segments (chosen runs)",
                    False,
                ),
                ("Where runs overlap", "the value with the smaller error is kept", False),
                ("Wavelengths stored", f"{wavelengths.size} ({wavelengths[0]:g}–{wavelengths[-1]:g} nm)", False),
                ("Wavelengths fitted", int(np.isfinite(np.asarray(nk["n1"])).sum()), False),
                ("Results", "Fitted n1 and Fitted k1 columns of the Data sheet", False),
            ]
            for run in nk["runs"]:
                used = int((segment_of == run["id"]).sum())
                rows += [
                    (
                        f"Run {run['id']}",
                        f"{run['start']:g}–{run['end']:g} nm · d₁ {run['d1']:.6f} nm · n₁ "
                        f"{_plain_number(run['n1_start'])} + {_plain_number(run['n1_range'])} · "
                        f"k₁ {_plain_number(run['k1_start'])} + {_plain_number(run['k1_range'])} · "
                        f"{used} stored value{'s' if used != 1 else ''}",
                        False,
                    ),
                ]
        return rows


class NdFitDialog(QDialog):
    """Nd Fit window (non-modal), opened from step 9 of the Nd and Nk window.

    Inputs: wavelength, n1 start and range, k1, d1 start and range. R0 and R60
    (measured) and Si n2, k2 are read from the Post Process window's table at that
    wavelength. Run does thin_film.fit_nd and shows n1, d1, the error, and the
    measured and fitted R60 and R0. The inputs and the last result stay when
    the window is closed; the result is cleared when step 7's values change.
    """

    # key, label, default (the values in Optical_Nd_V7S.py)
    _FIELDS = (
        ("wavelength", "Wavelength (nm)", "700"),
        ("n1_start", "n₁ start", "1.9"),
        ("n1_range", "n₁ range (Δn₁)", "0.2"),
        ("k1", "k₁", "0"),
        ("d1_start", "d₁ start (nm)", "300"),
        ("d1_range", "d₁ range (Δd₁, nm)", "20"),
    )
    _OUTPUTS = (
        ("n1", "n₁"),
        ("d1", "d₁"),
        ("error", "Error (ΔR₆₀² + ΔR₀²)"),
        ("r60_measured", "R₆₀ measured"),
        ("r60_fit", "R₆₀ fitted"),
        ("r0_measured", "R₀ measured"),
        ("r0_fit", "R₀ fitted"),
    )

    def __init__(self, owner: NdNkWindow) -> None:
        super().__init__(owner, Qt.WindowType.Window | Qt.WindowType.WindowCloseButtonHint)
        self._owner = owner
        self._revision = 0
        self.setWindowTitle("Nd Fit")
        self.setModal(False)
        self.setStyleSheet(_ND_NK_STYLE)

        # Everything but the Close button scrolls, so small screens still work.
        content = QWidget()
        root = QVBoxLayout(content)
        root.setContentsMargins(14, 14, 14, 6)
        root.setSpacing(10)
        intro = NdNkWindow._note(
            "Finds the film's refractive index n₁ and thickness d₁ at one "
            "wavelength (k₁ fixed) so that the calculated R₀ and R₆₀ match the "
            "absolute R₀ and R₆₀ from step 7. Same search as Optical_Nd_V7S.py, "
            "with π."
        )
        root.addWidget(intro)

        inputs = QGroupBox("Inputs")
        inputs_layout = QVBoxLayout(inputs)
        inputs_layout.setSpacing(8)
        grid = QGridLayout()  # (a form layout cuts off wrapped text)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)
        grid.setColumnStretch(1, 1)
        self._edits: dict[str, QLineEdit] = {}
        row = 0
        for key, label, default in self._FIELDS:
            edit = QLineEdit(default)
            validator = QDoubleValidator(0.0, 1.0e6, 6, edit)
            validator.setNotation(QDoubleValidator.Notation.StandardNotation)
            validator.setLocale(QLocale.c())
            edit.setValidator(validator)
            edit.textChanged.connect(self.refresh)
            self._edits[key] = edit
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(edit, row, 1)
            row += 1
            if key == "wavelength":
                # Short lines broken by hand; no word wrap, so the full height shows.
                self._table_values = QLabel("")
                self._table_values.setTextFormat(Qt.TextFormat.RichText)
                grid.addWidget(self._table_values, row, 1)
                row += 1
        inputs_layout.addLayout(grid)
        inputs_layout.addWidget(
            NdNkWindow._note(
                "n₁ is searched from start to start + range, and d₁ likewise, on a "
                "100 × 100 grid; then twice more on a finer grid around the best "
                "point (3 rounds, as in the script)."
            )
        )
        # Fixed height: on a short screen the window scrolls instead of squeezing.
        inputs.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        root.addWidget(inputs)

        self._run_btn = QPushButton("Run Nd Fit")
        self._run_btn.setObjectName("accentBtn")
        self._run_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._run_btn.setDefault(True)  # Enter in a field runs the fit
        self._run_btn.clicked.connect(self._on_run_clicked)
        root.addWidget(self._run_btn)
        self._run_footnote = NdNkWindow._footnote()
        root.addWidget(self._run_footnote)

        output = QGroupBox("Output")
        out_layout = QVBoxLayout(output)
        out_form = QFormLayout()
        out_form.setSpacing(6)
        self._output_values: dict[str, QLabel] = {}
        for key, label in self._OUTPUTS:
            value = QLabel("—")
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            value.setStyleSheet("font-weight: 600;")
            self._output_values[key] = value
            out_form.addRow(label, value)
        out_layout.addLayout(out_form)
        self._output_details = NdNkWindow._note("No fit yet.")
        out_layout.addWidget(self._output_details)
        self._output_warning = NdNkWindow._footnote()
        out_layout.addWidget(self._output_warning)
        output.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        root.addWidget(output)
        root.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content.setMaximumWidth(760)  # a readable column, centred, when maximized
        scroll.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(content)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 14)
        outer.setSpacing(6)
        outer.addWidget(scroll, stretch=1)
        close_row = QHBoxLayout()
        close_row.setContentsMargins(14, 0, 14, 0)
        close_row.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.setObjectName("accentBtn")
        close_btn.setAutoDefault(False)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.clicked.connect(self.close)
        close_row.addWidget(close_btn)
        outer.addLayout(close_row)

        if owner.nd_fit is not None:
            self._show_result(owner.nd_fit)
        self.refresh()
        # Full height when the screen allows it (the result lines included).
        content.setMinimumWidth(560)
        wanted = content.sizeHint().height() + 70
        screen = self.screen() or QApplication.primaryScreen()
        if screen is not None:
            wanted = min(wanted, screen.availableGeometry().height() - 80)
        self.resize(620, max(400, wanted))

    # Inputs ------------------------------------------------------------

    def _read_inputs(self) -> tuple[dict[str, float], list[str]]:
        values: dict[str, float] = {}
        problems: list[str] = []
        labels = {key: label for key, label, _default in self._FIELDS}
        for key, edit in self._edits.items():
            try:
                values[key] = float(edit.text().strip())
            except ValueError:
                problems.append(f"{labels[key]}: enter a number")
        rules = (
            ("n1_start", lambda v: v > 0, "must be above 0"),
            ("n1_range", lambda v: v > 0, "must be above 0"),
            ("k1", lambda v: v >= 0, "must be 0 or more"),
            ("d1_start", lambda v: v >= 0, "must be 0 or more"),
            ("d1_range", lambda v: v > 0, "must be above 0"),
        )
        for key, ok, text in rules:
            if key in values and not ok(values[key]):
                problems.append(f"{labels[key]}: {text}")
        return values, problems

    def _needed(self) -> tuple[list[str], Optional[dict[str, object]], dict[str, float]]:
        """(what is still needed, the table values at the wavelength, the inputs)."""
        needed = list(self._owner._requirements("nd_fit"))
        values, problems = self._read_inputs()
        table: Optional[dict[str, object]] = None
        if not needed and "wavelength" in values:
            try:
                table = self._owner.nd_fit_values(values["wavelength"])
            except ValueError as exc:
                needed.append(str(exc))
        return needed + problems, table, values

    def refresh(self, *_args: object) -> None:
        """Update the table values shown under Wavelength and the Run footnote."""
        needed, table, _values = self._needed()
        if table is not None:
            self._table_values.setText(
                f"<span style='color:{NdNkWindow._READY_COLOR}'>R₀ = {table['r0']:.6f}, "
                f"R₆₀ = {table['r60']:.6f}<br>Si n₂ = {table['n2']:.6f}, "
                f"k₂ = {table['k2']:.6f}<br>({table['source']})</span>"
            )
        else:
            self._table_values.setText(
                f"<span style='color:{NdNkWindow._NEEDS_COLOR}'>R₀ and R₆₀ at this "
                "wavelength: not available</span>"
            )
        self._owner._set_footnote(self._run_footnote, needed, "Ready to run.")

    # Run ---------------------------------------------------------------

    def _on_run_clicked(self) -> None:
        needed, table, values = self._needed()
        if needed:
            self._owner._show_requirements_popup("Not Ready Yet", needed)
            return
        before = self._revision
        popup = ProgressPopup("Running the Nd fit", parent=self)
        try:
            popup.run_handler(lambda: self._run(table, values), lambda: self._revision != before)
        finally:
            popup.deleteLater()

    def _run(self, table: dict[str, object], values: dict[str, float]) -> bool:
        try:
            result = thin_film.fit_nd(
                values["wavelength"],
                table["r0"],
                table["r60"],
                table["n2"],
                table["k2"],
                values["n1_start"],
                values["n1_range"],
                values["d1_start"],
                values["d1_range"],
                k1=values["k1"],
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Nd Fit Failed", str(exc))
            return False
        result["source"] = table["source"]
        owner = self._owner
        nk_d1 = owner._nk_d1_in_use()  # d1 the Nk results were made with
        owner.nd_fit = result
        nk_reset = nk_d1 is not None and float(result["d1"]) != nk_d1
        if nk_reset:
            owner.nd_d1_changed(nk_d1, float(result["d1"]))
        if owner._nk_fit_dialog is not None:
            owner._nk_fit_dialog.nd_fit_changed()
        self._revision += 1
        self._show_result(result)
        owner._update_footnotes()
        owner._summary.setText(
            f"Nd fit at {result['wavelength']:.3f} nm: n₁ = {result['n1']:.6f}, "
            f"d₁ = {result['d1']:.4f} nm (error {result['error']:.3e})."
            + (" d₁ changed, so the Nk results were cleared." if nk_reset else "")
        )
        if nk_reset:
            QMessageBox.information(
                self,
                "Nk Results Cleared",
                f"The new Nd fit gives d₁ = {result['d1']:.4f} nm (the Nk fit used "
                f"{nk_d1:.4f} nm).\n\nNk uses d₁ as an input, so its runs and the "
                "stored n₁ and k₁ were cleared. Run the Nk fit again with the new d₁.",
            )
        return True

    # Output ------------------------------------------------------------

    def _show_result(self, result: dict[str, object]) -> None:
        texts = {
            "n1": f"{result['n1']:.6f}",
            "d1": f"{result['d1']:.4f} nm",
            "error": f"{result['error']:.3e}",
            "r60_measured": f"{result['r60_measured']:.6f}",
            "r60_fit": f"{result['r60_fit']:.6f}",
            "r0_measured": f"{result['r0_measured']:.6f}",
            "r0_fit": f"{result['r0_fit']:.6f}",
        }
        for key, label in self._output_values.items():
            label.setText(texts[key])
        n1_end = result["n1_start"] + result["n1_range"]
        d1_end = result["d1_start"] + result["d1_range"]
        self._output_details.setText(
            f"At {result['wavelength']:.3f} nm: R₀, R₆₀ and Si n₂, k₂ "
            f"{result.get('source', 'from the table')}. "
            f"Searched n₁ {result['n1_start']:g}–{n1_end:g} and d₁ "
            f"{result['d1_start']:g}–{d1_end:g} nm, k₁ = {result['k1']:g}, "
            f"Si n₂ = {result['n2']:.6f}, k₂ = {result['k2']:.6f}."
        )
        warnings = []
        edges = result["edges"]
        if edges["n1"] == "low":
            warnings.append(
                f"n₁ ended at the low end of its range in round 1, so the best n₁ may "
                f"be below {result['n1_start']:g}: try a lower n₁ start."
            )
        elif edges["n1"] == "high":
            warnings.append(
                f"n₁ ended at the high end of its range in round 1, so the best n₁ may "
                f"be above {n1_end:g}: try a higher n₁ start or a larger range."
            )
        if edges["d1"] == "low":
            warnings.append(
                f"d₁ ended at the low end of its range in round 1, so the best d₁ may "
                f"be below {result['d1_start']:g} nm: try a lower d₁ start."
            )
        elif edges["d1"] == "high":
            warnings.append(
                f"d₁ ended at the high end of its range in round 1, so the best d₁ may "
                f"be above {d1_end:g} nm: try a higher d₁ start or a larger range."
            )
        self._output_warning.setText(
            "".join(
                f"<p style='color:{NdNkWindow._NEEDS_COLOR}; margin:0 0 4px 0'>{text}</p>"
                for text in warnings
            )
        )

    def results_cleared(self, had_result: bool) -> None:
        """Step 7's values changed: the shown fit no longer applies."""
        for label in self._output_values.values():
            label.setText("—")
        self._output_warning.setText("")
        self._output_details.setText(
            "The absolute reflectance changed, so the last fit was cleared. Run the "
            "fit again."
            if had_result
            else "No fit yet."
        )


_NK_SEGMENT_COLORS: tuple[tuple[int, int, int], ...] = (
    (79, 195, 247), (255, 183, 77), (129, 199, 132), (229, 115, 115),
    (186, 104, 200), (255, 213, 79), (77, 182, 172), (240, 98, 146),
)
_NK_RESULT_KEYS = ("n1", "k1", "error", "r0_fit", "r60_fit")
# Runs list in the Nk Fit window: a tick box per row shows what is on the graph.
_NK_RUNS_LIST_STYLE = """
QListWidget#nkRunsList::item { padding: 4px 2px; }
QListWidget#nkRunsList::indicator {
    width: 14px; height: 14px; border-radius: 3px;
    border: 1px solid #5c6370; background-color: #1e2229;
}
QListWidget#nkRunsList::indicator:checked { border: 1px solid #4fc3f7; background-color: #4fc3f7; }
QLineEdit#nkD1Field { background-color: #161a20; color: #c8ccd4; border: 1px dashed #3a414d; }
"""


def _merge_nk_segments(segments: list[dict[str, object]]) -> dict[str, np.ndarray]:
    """Merge Nk runs into one result, sorted by wavelength.

    Where runs overlap, the value with the smaller error at that wavelength is
    kept (on an exact tie, the later run). A wavelength no run could fit is
    kept as N/A. ``segment`` says which run each value came from.
    """
    best: dict[float, tuple[float, int, int]] = {}  # wavelength -> (error, run, row)
    for run_index, segment in enumerate(segments):
        errors = np.asarray(segment["error"], dtype=np.float64)
        for row, wavelength in enumerate(np.asarray(segment["wavelength"], dtype=np.float64)):
            error = float(errors[row]) if np.isfinite(errors[row]) else math.inf
            key = round(float(wavelength), 9)
            if key not in best or error <= best[key][0]:
                best[key] = (error, run_index, row)
    keys = sorted(best)
    out: dict[str, np.ndarray] = {
        name: np.full(len(keys), np.nan) for name in ("wavelength",) + _NK_RESULT_KEYS
    }
    out["segment"] = np.zeros(len(keys), dtype=int)
    out["n1_edge"] = np.full(len(keys), "", dtype=object)
    out["k1_edge"] = np.full(len(keys), "", dtype=object)
    for i, key in enumerate(keys):
        _error, run_index, row = best[key]
        segment = segments[run_index]
        out["wavelength"][i] = segment["wavelength"][row]
        for name in _NK_RESULT_KEYS:
            out[name][i] = segment[name][row]
        out["segment"][i] = segment["id"]
        out["n1_edge"][i] = segment["n1_edge"][row]
        out["k1_edge"][i] = segment["k1_edge"][row]
    return out


def _plain_number(value: float) -> str:
    """0.000001 rather than 1e-06; no trailing zeros."""
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-0") else "0"


def _nk_segment_text(segment: dict[str, object]) -> str:
    fitted = int(np.isfinite(np.asarray(segment["n1"])).sum())
    n1_low, k1_low = segment["n1_start"], segment["k1_start"]
    return (
        f"Run {segment['id']}: {segment['start']:g}–{segment['end']:g} nm · "
        f"{fitted} wavelengths · n₁ {_plain_number(n1_low)}–"
        f"{_plain_number(n1_low + segment['n1_range'])} · k₁ {_plain_number(k1_low)}–"
        f"{_plain_number(k1_low + segment['k1_range'])} · d₁ {segment['d1']:.4f} nm"
    )


def _nk_edge_text(segment: dict[str, object]) -> str:
    """Plain-words note on where n1 / k1 ended at the end of their range."""
    wavelengths = np.asarray(segment["wavelength"])
    notes = []
    for key, name, low, high in (
        ("n1_edge", "n₁", "n₁ may be below n₁ start", "n₁ may be above n₁ start + range"),
        ("k1_edge", "k₁", "k₁ ≈ 0 there if k₁ start is near 0", "k₁ may be above k₁ start + range"),
    ):
        edges = np.asarray(segment[key], dtype=object)
        for side, meaning in (("low", low), ("high", high)):
            hits = wavelengths[edges == side]
            if hits.size:
                span = f"{hits[0]:g} nm" if hits.size == 1 else f"{hits[0]:g}–{hits[-1]:g} nm"
                notes.append(
                    f"{name} at the {side} end of its range at {hits.size} wavelength"
                    f"{'s' if hits.size != 1 else ''} ({span}): {meaning}"
                )
    return "; ".join(notes)


class _NkFitThread(QThread):
    """Background Nk fit over one wavelength range, one wavelength at a time.

    Emits point_done(index, result) after each wavelength (result is the
    thin_film.fit_nk dict, or None where R0, R60 or Si n, k is N/A). Setting
    stop_requested ends the loop after the current wavelength.
    """

    point_done = pyqtSignal(int, object)

    def __init__(self, data: tuple[np.ndarray, ...], values: dict[str, float]) -> None:
        super().__init__()
        self._data = data
        self._values = values
        self.stop_requested = False

    def run(self) -> None:  # background thread: maths only
        wavelengths, r0, r60, n2, k2 = self._data
        v = self._values
        for index in range(wavelengths.size):
            if self.stop_requested:
                return
            inputs = (float(r0[index]), float(r60[index]), float(n2[index]), float(k2[index]))
            result = None
            if all(math.isfinite(x) for x in inputs):
                try:
                    result = thin_film.fit_nk(
                        float(wavelengths[index]), *inputs, v["d1"],
                        v["n1_start"], v["n1_range"], v["k1_start"], v["k1_range"],
                    )
                except ValueError:
                    result = None
            self.point_done.emit(index, result)


class _NkFinishDialog(QDialog):
    """Finish: store all runs merged (Complete) or only the ticked ones (Segments)."""

    def __init__(self, parent: QWidget, segments: list[dict[str, object]]) -> None:
        super().__init__(parent)
        self.setWindowTitle("Finish Nk Fit")
        self.setModal(True)
        self.setStyleSheet(_ND_NK_STYLE)
        self.setMinimumWidth(640)
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.addWidget(QLabel(f"<b>How should the {len(segments)} run(s) be stored?</b>"))
        self.complete = QRadioButton("Complete: merge all runs into one result")
        self.complete.setChecked(True)
        self.pick = QRadioButton("Segments: choose which runs to keep")
        layout.addWidget(self.complete)
        layout.addWidget(self.pick)
        self.runs = QListWidget()
        for segment in segments:
            item = QListWidgetItem(_nk_segment_text(segment))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            item.setForeground(QColor(*_NK_SEGMENT_COLORS[(segment["id"] - 1) % len(_NK_SEGMENT_COLORS)]))
            item.setData(Qt.ItemDataRole.UserRole, segment["id"])
            self.runs.addItem(item)
        self.runs.setEnabled(False)
        self.pick.toggled.connect(self.runs.setEnabled)
        layout.addWidget(self.runs)
        layout.addWidget(
            NdNkWindow._note(
                "Where runs overlap, the value with the smaller error at each "
                "wavelength is kept. The stored result goes into the Post Process "
                "table (Fitted n1, Fitted k1) and the Excel export; the runs list "
                "is then emptied for the next set."
            )
        )
        row = QHBoxLayout()
        row.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.setObjectName("accentBtn")
        cancel.clicked.connect(self.reject)
        store = QPushButton("Store")
        store.setObjectName("accentBtn")
        store.setDefault(True)
        store.clicked.connect(self._on_store)
        row.addWidget(cancel)
        row.addWidget(store)
        layout.addLayout(row)

    def chosen_ids(self) -> list[int]:
        if self.complete.isChecked():
            return [self.runs.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.runs.count())]
        return [
            self.runs.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.runs.count())
            if self.runs.item(i).checkState() == Qt.CheckState.Checked
        ]

    def mode(self) -> str:
        return "complete" if self.complete.isChecked() else "segments"

    def _on_store(self) -> None:
        if not self.chosen_ids():
            QMessageBox.information(self, "Nothing Ticked", "Tick at least one run to keep.")
            return
        self.accept()


class NkFitDialog(QDialog):
    """Nk Fit window (non-modal), opened from step 10 of the Post Process window.

    Each Run fits n1 and k1 (d1 fixed) at the interpolated wavelengths inside
    the chosen wavelength range, in the background; n1 and k1 are plotted live
    (stacked plots) as each wavelength is done, with a progress bar above the
    plots (Stop discards that run). Every finished run is kept as a segment
    until Finish, which stores either all runs merged (Complete) or the ticked
    ones (Segments); where runs overlap, the smaller error wins. The stored
    result goes into the Post Process table. Runs and the stored result are
    cleared when step 7 changes.

    The graph shows only the latest run (the one in progress, else the newest
    kept run; with no runs, the stored result). Clicking another run or the
    stored result in the list shows it alongside, clicking again hides it;
    each new run starts with a clean graph.
    """

    # key, label, default (n1 / k1 values from Optical_NK_V7S.py; d1 from the Nd fit)
    _FIELDS = (
        ("d1", "d₁ (nm)", ""),
        ("wl_start", "Wavelength start (nm)", ""),
        ("wl_end", "Wavelength end (nm)", ""),
        ("n1_start", "n₁ start", "1.45"),
        ("n1_range", "n₁ range (Δn₁)", "0.05"),
        ("k1_start", "k₁ start", "0.000001"),
        ("k1_range", "k₁ range (Δk₁)", "0.1"),
    )
    _STORED_COLOR = (235, 235, 235)

    def __init__(self, owner: NdNkWindow) -> None:
        super().__init__(
            owner,
            Qt.WindowType.Window
            | Qt.WindowType.WindowMinMaxButtonsHint
            | Qt.WindowType.WindowCloseButtonHint,
        )
        self._owner = owner
        self._thread: Optional[_NkFitThread] = None
        self._run: Optional[dict[str, object]] = None  # the run in progress
        self._paused_plots: list[QWidget] = []
        self._curves: list[object] = []  # segment and stored-result curves
        self._live_curves: list[object] = []
        # Runs (ids) or "stored" the user clicked to show beside the latest run;
        # emptied when a new run starts.
        self._shown_extra: set[object] = set()
        self.setWindowTitle("Nk Fit")
        self.setModal(False)
        self.setStyleSheet(_ND_NK_STYLE + _NK_RUNS_LIST_STYLE)

        # Left: inputs, Run, runs list, Finish (scrolls on small screens).
        left = QWidget()
        root = QVBoxLayout(left)
        root.setContentsMargins(14, 14, 10, 14)
        root.setSpacing(10)
        root.addWidget(
            NdNkWindow._note(
                "Finds the film's n₁ and k₁ (d₁ fixed) at the interpolated "
                "wavelengths in a range, so the calculated R₀ and R₆₀ match the "
                "absolute R₀ and R₆₀ from step 7 (search of Optical_NK_V7S.py, with "
                "π). Run several ranges with their own n₁ / k₁ ranges, then Finish."
            )
        )
        inputs = QGroupBox("Inputs")
        inputs_layout = QVBoxLayout(inputs)
        inputs_layout.setSpacing(8)
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)
        grid.setColumnStretch(1, 1)
        self._edits: dict[str, QLineEdit] = {}
        row = 0
        for key, label, default in self._FIELDS:
            edit = QLineEdit(default)
            decimals = 3 if key.startswith("wl_") else 6
            validator = QDoubleValidator(0.0, 1.0e6, decimals, edit)
            validator.setNotation(QDoubleValidator.Notation.StandardNotation)
            validator.setLocale(QLocale.c())
            edit.setValidator(validator)
            edit.textChanged.connect(self.refresh)
            self._edits[key] = edit
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(edit, row, 1)
            row += 1
            if key == "d1":
                # d1 comes from the Nd fit (step 9) only: one d1 for every run.
                edit.setReadOnly(True)
                edit.setObjectName("nkD1Field")
                edit.setToolTip("d₁ comes from the Nd fit (step 9). Run the Nd fit again to change it.")
                self._d1_source = QLabel("")
                self._d1_source.setTextFormat(Qt.TextFormat.RichText)
                self._d1_source.setWordWrap(True)
                grid.addWidget(self._d1_source, row, 1)
                row += 1
            if key == "wl_end":
                self._wavelengths_label = QLabel("")
                self._wavelengths_label.setTextFormat(Qt.TextFormat.RichText)
                self._wavelengths_label.setWordWrap(True)
                grid.addWidget(self._wavelengths_label, row, 1)
                row += 1
        inputs_layout.addLayout(grid)
        inputs_layout.addWidget(
            NdNkWindow._note(
                "At each wavelength, n₁ is searched from start to start + range and "
                "k₁ likewise on a 100 × 100 grid, then twice more on a finer grid "
                "around the best point (3 rounds, as in the script). About 12 ms "
                "per wavelength."
            )
        )
        inputs.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        root.addWidget(inputs)

        self._run_btn = QPushButton("Run Nk Fit")
        self._run_btn.setObjectName("accentBtn")
        self._run_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._run_btn.setDefault(True)
        self._run_btn.clicked.connect(self._on_run_clicked)
        root.addWidget(self._run_btn)
        self._run_footnote = NdNkWindow._footnote()
        root.addWidget(self._run_footnote)

        runs = QGroupBox("Runs (kept until Finish)")
        runs_layout = QVBoxLayout(runs)
        runs_layout.addWidget(
            NdNkWindow._note(
                "The graph shows the latest run. Click another run (or the stored "
                "result) to show it alongside; click again to hide it. Each new run "
                "starts with a clean graph."
            )
        )
        self._runs_list = QListWidget()
        self._runs_list.setObjectName("nkRunsList")
        self._runs_list.setMinimumHeight(120)
        self._runs_list.setWordWrap(True)
        self._runs_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self._runs_list.setCursor(Qt.CursorShape.PointingHandCursor)
        self._runs_list.itemClicked.connect(self._on_run_item_clicked)
        runs_layout.addWidget(self._runs_list)
        self._finish_btn = QPushButton("Finish…")
        self._finish_btn.setObjectName("accentBtn")
        self._finish_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._finish_btn.setAutoDefault(False)
        self._finish_btn.clicked.connect(self._on_finish_clicked)
        runs_layout.addWidget(self._finish_btn)
        self._stored_label = NdNkWindow._note("")
        runs_layout.addWidget(self._stored_label)
        root.addWidget(runs, stretch=1)

        close_row = QHBoxLayout()
        close_row.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.setObjectName("accentBtn")
        close_btn.setAutoDefault(False)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.clicked.connect(self.close)
        close_row.addWidget(close_btn)
        root.addLayout(close_row)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setWidget(left)
        left.setMaximumWidth(450)  # never wider than the column (no cut-off text)
        left_scroll.setFixedWidth(460)

        # Right: progress (not a pop-up) above the live n1 / k1 plots.
        right = QVBoxLayout()
        right.setContentsMargins(4, 14, 14, 14)
        right.setSpacing(8)
        progress_row = QHBoxLayout()
        self._progress_label = QLabel("No fit running.")
        self._progress_label.setMinimumWidth(360)
        progress_row.addWidget(self._progress_label)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1000)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFormat("%p%")
        progress_row.addWidget(self._progress_bar, stretch=1)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setObjectName("accentBtn")
        self._stop_btn.setAutoDefault(False)
        self._stop_btn.setEnabled(False)
        self._stop_btn.setToolTip("Stop this run; its points are discarded.")
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        progress_row.addWidget(self._stop_btn)
        right.addLayout(progress_row)

        self._plots = pg.GraphicsLayoutWidget()
        self._n_plot = self._plots.addPlot(row=0, col=0)
        self._k_plot = self._plots.addPlot(row=1, col=0)
        self._k_plot.setXLink(self._n_plot)
        for plot, label in ((self._n_plot, "n₁"), (self._k_plot, "k₁")):
            plot.setLabel("left", label)
            plot.showGrid(x=True, y=True, alpha=0.25)
            plot.getAxis("bottom").enableAutoSIPrefix(False)
            plot.getAxis("left").enableAutoSIPrefix(False)
        self._k_plot.setLabel("bottom", "Wavelength (nm)")
        self._n_plot.setTitle("n₁ and k₁ vs wavelength (each point drawn as it is calculated)")
        self._legend = self._n_plot.addLegend(
            offset=(-10, 10), brush=pg.mkBrush(26, 29, 35, 230), pen=pg.mkPen(80, 86, 96)
        )
        right.addWidget(self._plots, stretch=1)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(left_scroll)
        outer.addLayout(right, stretch=1)

        # Redraw the live plot at most every 80 ms (not on every point).
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setInterval(80)
        self._redraw_timer.timeout.connect(self._redraw_live)
        self._elapsed = QElapsedTimer()
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._stop_and_wait)

        self.nd_fit_changed()
        self._redraw_saved()
        self.refresh()
        self.resize(1280, 860)

    # Inputs ------------------------------------------------------------

    def nd_fit_changed(self) -> None:
        """Show d1 from the latest Nd fit (empty when there is none)."""
        fit = self._owner.nd_fit
        self._edits["d1"].setText("" if fit is None else f"{fit['d1']:.6f}")
        self.refresh()

    def _grid(self) -> Optional[np.ndarray]:
        absolute = self._owner.absolute
        return None if absolute is None else np.asarray(absolute["grid"], dtype=np.float64)

    def _read_inputs(self) -> tuple[dict[str, float], list[str]]:
        values: dict[str, float] = {}
        problems: list[str] = []
        labels = {key: label for key, label, _default in self._FIELDS}
        for key, edit in self._edits.items():
            if key == "d1":  # exactly the Nd fit's d1 (the box shows it rounded)
                fit = self._owner.nd_fit
                if fit is None:
                    problems.append("d₁: run the Nd fit (step 9) first")
                else:
                    values["d1"] = float(fit["d1"])
                continue
            try:
                values[key] = float(edit.text().strip())
            except ValueError:
                problems.append(f"{labels[key]}: enter a number")
        rules = (
            ("d1", lambda v: v > 0, "must be above 0"),
            ("n1_start", lambda v: v > 0, "must be above 0"),
            ("n1_range", lambda v: v > 0, "must be above 0"),
            ("k1_start", lambda v: v >= 0, "must be 0 or more"),
            ("k1_range", lambda v: v > 0, "must be above 0"),
        )
        for key, ok, text in rules:
            if key in values and not ok(values[key]):
                problems.append(f"{labels[key]}: {text}")
        grid = self._grid()
        if grid is not None and "wl_start" in values and "wl_end" in values:
            if values["wl_end"] < values["wl_start"]:
                problems.append("Wavelength end: must not be below the start")
            elif self._range_indexes(values).size == 0:
                problems.append(
                    f"Wavelength range: no interpolated wavelength inside it (the grid "
                    f"is {grid[0]:g}–{grid[-1]:g} nm)"
                )
        return values, problems

    def _range_indexes(self, values: dict[str, float]) -> np.ndarray:
        grid = self._grid()
        if grid is None:
            return np.zeros(0, dtype=int)
        return np.flatnonzero(
            (grid >= values["wl_start"] - 1e-9) & (grid <= values["wl_end"] + 1e-9)
        )

    def _needed(self) -> tuple[list[str], dict[str, float]]:
        needed = list(self._owner._requirements("nk_fit"))
        values, problems = self._read_inputs()
        return needed + problems, values

    def refresh(self, *_args: object) -> None:
        """Update the notes under d1 and the wavelength range, and the footnotes."""
        ready, needs = NdNkWindow._READY_COLOR, NdNkWindow._NEEDS_COLOR
        fit = self._owner.nd_fit
        if fit is not None:
            self._d1_source.setText(
                f"<span style='color:{ready}'>From the Nd fit at "
                f"{fit['wavelength']:.3f} nm.</span> <span style='color:#8a93a3'>A new "
                "Nd fit with another d₁ clears the Nk runs.</span>"
            )
        else:
            self._d1_source.setText(
                f"<span style='color:{needs}'>No Nd fit yet: run step 9 first</span>"
            )
        grid = self._grid()
        if grid is not None:
            for key, value in (("wl_start", grid[0]), ("wl_end", grid[-1])):
                if not self._edits[key].text().strip():
                    self._edits[key].blockSignals(True)
                    self._edits[key].setText(f"{value:g}")
                    self._edits[key].blockSignals(False)
            values, _problems = self._read_inputs()
            step = (grid[-1] - grid[0]) / (grid.size - 1) if grid.size > 1 else 0.0
            if "wl_start" in values and "wl_end" in values:
                count = self._range_indexes(values).size
                self._wavelengths_label.setText(
                    f"<span style='color:{ready if count else needs}'>{count} interpolated "
                    f"wavelengths in this range<br>(grid {grid[0]:g}–{grid[-1]:g} nm, "
                    f"step {step:g} nm)</span>"
                )
        else:
            self._wavelengths_label.setText(
                f"<span style='color:{needs}'>Interpolated wavelengths: not available "
                "yet (step 7)</span>"
            )
        running = self._thread is not None
        needed, _values = self._needed()
        if running:
            self._run_footnote.setText(
                f"<span style='color:{needs}'>A run is in progress (Stop it, or wait).</span>"
            )
        else:
            self._owner._set_footnote(self._run_footnote, needed, "Ready to run.")
        self._run_btn.setEnabled(not running)
        self._finish_btn.setEnabled(bool(self._owner.nk_segments) and not running)
        stored = self._owner.nk_fit
        if stored is not None:
            used = sorted(set(int(s) for s in np.asarray(stored["segment"])))
            mode = "Complete" if stored["mode"] == "complete" else "Segments"
            self._stored_label.setText(
                f"Stored in the table: {int(np.isfinite(stored['n1']).sum())} wavelengths "
                f"({mode}: run{'s' if len(used) != 1 else ''} "
                f"{', '.join(str(u) for u in used)} of that set)."
            )
        else:
            self._stored_label.setText("Nothing stored yet: press Finish when done.")

    # Run ---------------------------------------------------------------

    def _on_run_clicked(self) -> None:
        if self._thread is not None:
            return
        needed, values = self._needed()
        if needed:
            self._owner._show_requirements_popup("Not Ready Yet", needed)
            return
        owner = self._owner
        indexes = self._range_indexes(values)
        data = tuple(
            np.array(np.asarray(array, dtype=np.float64)[indexes])
            for array in (
                owner.absolute["grid"], owner.absolute["R0"], owner.absolute["R60"],
                owner.si_interp["n"], owner.si_interp["k"],
            )
        )
        run_id = owner.nk_run_counter + 1
        owner.nk_run_counter = run_id
        count = data[0].size
        self._run = {
            "id": run_id, "start": values["wl_start"], "end": values["wl_end"],
            "d1": values["d1"], "n1_start": values["n1_start"], "n1_range": values["n1_range"],
            "k1_start": values["k1_start"], "k1_range": values["k1_range"],
            "wavelength": data[0].copy(),
            **{name: np.full(count, np.nan) for name in _NK_RESULT_KEYS},
            "n1_edge": np.full(count, "", dtype=object),
            "k1_edge": np.full(count, "", dtype=object),
            "done": 0, "dirty": False,
        }
        # A new run starts with a clean graph: only this run is drawn (earlier
        # runs can be clicked back on while it runs).
        self._shown_extra.clear()
        self._redraw_curves()
        color = _NK_SEGMENT_COLORS[(run_id - 1) % len(_NK_SEGMENT_COLORS)]
        name = f"Run {run_id}: {values['wl_start']:g}–{values['wl_end']:g} nm (running)"
        self._live_curves = [
            self._n_plot.plot([], [], pen=pg.mkPen(color, width=2), symbol="o", symbolSize=5,
                              symbolBrush=pg.mkBrush(*color), symbolPen=None, name=name),
            self._k_plot.plot([], [], pen=pg.mkPen(color, width=2), symbol="o", symbolSize=5,
                              symbolBrush=pg.mkBrush(*color), symbolPen=None),
        ]
        for curve in self._live_curves:
            curve.setZValue(10)  # the latest run is drawn on top
        grid = self._grid()
        self._n_plot.setXRange(float(grid[0]), float(grid[-1]), padding=0.02)
        # While the fit runs, the other plots stop repainting: the main window's
        # live plot redraws several antialiased curves on every spectrometer
        # frame, and that holds up the background calculation (about 3x slower).
        # Data keeps arriving; their painting resumes when the run ends.
        self._paused_plots = [
            plot
            for window in (owner._main, owner)
            for plot in window.findChildren(pg.PlotWidget)
            if plot.updatesEnabled()
        ]
        for plot in self._paused_plots:
            plot.setUpdatesEnabled(False)
        self._thread = _NkFitThread(data, values)
        self._thread.point_done.connect(self._on_point)
        self._thread.finished.connect(self._on_run_finished)
        self._progress_bar.setValue(0)
        self._progress_label.setText(f"Run {run_id}: 0 of {count} wavelengths")
        self._stop_btn.setEnabled(True)
        for edit in self._edits.values():
            edit.setEnabled(False)
        self._elapsed.start()
        self._redraw_timer.start()
        self._thread.start()
        self.refresh()

    def _on_point(self, index: int, result: Optional[dict[str, object]]) -> None:
        run = self._run
        if run is None:
            return
        if result is not None:
            for name in _NK_RESULT_KEYS:
                run[name][index] = result[name]
            run["n1_edge"][index] = result["edges"]["n1"] or ""
            run["k1_edge"][index] = result["edges"]["k1"] or ""
        run["done"] = index + 1
        run["dirty"] = True

    def _redraw_live(self) -> None:
        run = self._run
        if run is None or run.get("discard") or not run["dirty"]:
            return  # (a discarded run's last points are not drawn)
        run["dirty"] = False
        done = run["done"]
        count = run["wavelength"].size
        wavelengths = run["wavelength"][:done]
        for curve, key in zip(self._live_curves, ("n1", "k1")):
            values = run[key][:done]
            ok = np.isfinite(values)
            curve.setData(wavelengths[ok], values[ok])
        self._progress_bar.setValue(int(round(1000 * done / max(count, 1))))
        seconds = self._elapsed.elapsed() / 1000.0
        left = seconds * (count - done) / done if done else 0.0
        self._progress_label.setText(
            f"Run {run['id']}: {done} of {count} wavelengths · {seconds:.1f} s elapsed · "
            f"about {left:.1f} s left"
        )

    def _on_stop_clicked(self) -> None:
        if self._thread is not None:
            self._thread.stop_requested = True
            self._stop_btn.setEnabled(False)
            self._progress_label.setText("Stopping…")

    def _on_run_finished(self) -> None:
        thread, run = self._thread, self._run
        self._thread = None
        self._run = None
        self._redraw_timer.stop()
        for plot in self._paused_plots:
            plot.setUpdatesEnabled(True)
            plot.update()
        self._paused_plots = []
        for edit in self._edits.values():
            edit.setEnabled(True)
        self._stop_btn.setEnabled(False)
        if thread is not None:
            thread.deleteLater()
        stopped = thread is None or thread.stop_requested or run is None or run.get("discard")
        for curve in self._live_curves:
            for plot in (self._n_plot, self._k_plot):
                if curve in plot.listDataItems():
                    plot.removeItem(curve)
        self._live_curves = []
        if stopped:
            if run is not None and not run.get("discard"):
                self._progress_label.setText(
                    f"Run {run['id']} stopped: its points were discarded."
                )
                self._progress_bar.setValue(0)
        else:
            run.pop("dirty", None)
            run.pop("done", None)
            seconds = self._elapsed.elapsed() / 1000.0
            self._owner.nk_segments.append(run)
            self._progress_bar.setValue(1000)
            self._progress_label.setText(
                f"Run {run['id']} done: {run['wavelength'].size} wavelengths in "
                f"{seconds:.1f} s."
            )
            self._owner._summary.setText(
                f"Nk fit run {run['id']} done ({run['start']:g}–{run['end']:g} nm). "
                "Press Finish in the Nk Fit window when all ranges are done."
            )
        self._redraw_saved()
        self._owner._update_footnotes()
        self.refresh()

    def _stop_and_wait(self) -> None:
        """App closing: end a running fit cleanly."""
        thread = self._thread
        if thread is not None:
            thread.stop_requested = True
            thread.wait(5000)

    # Runs list, stored result and plots ----------------------------------

    def _redraw_saved(self) -> None:
        """Rebuild the runs list and redraw the graph."""
        self._rebuild_runs_list()
        self._redraw_curves()

    def _fixed_key(self) -> Optional[object]:
        """What the graph always shows: the newest kept run (or, with no runs,
        the stored result). None while a run is in progress (that run is the
        latest, drawn live)."""
        if self._run is not None:
            return None
        segments = self._owner.nk_segments
        if segments:
            return segments[-1]["id"]
        return "stored" if self._owner.nk_fit is not None else None

    def _shown_keys(self) -> set[object]:
        keys = set(self._shown_extra)
        fixed = self._fixed_key()
        if fixed is not None:
            keys.add(fixed)
        return keys

    def _rebuild_runs_list(self) -> None:
        self._runs_list.clear()
        stored = self._owner.nk_fit
        if stored is not None:
            mode = "Complete" if stored["mode"] == "complete" else "Segments"
            text = (
                f"Stored result: {int(np.isfinite(np.asarray(stored['n1'])).sum())} "
                f"wavelengths ({mode})"
            )
            item = QListWidgetItem(text)
            item.setForeground(QColor(*self._STORED_COLOR))
            item.setData(Qt.ItemDataRole.UserRole, "stored")
            item.setData(Qt.ItemDataRole.UserRole + 1, text)
            item.setToolTip("The result stored by the last Finish (it is in the Post Process table).")
            self._runs_list.addItem(item)
        for segment in self._owner.nk_segments:
            color = _NK_SEGMENT_COLORS[(segment["id"] - 1) % len(_NK_SEGMENT_COLORS)]
            text = _nk_segment_text(segment)
            edges = _nk_edge_text(segment)
            if edges:
                text += "  ⚠"
            item = QListWidgetItem(text)
            item.setForeground(QColor(*color))
            item.setData(Qt.ItemDataRole.UserRole, segment["id"])
            item.setData(Qt.ItemDataRole.UserRole + 1, text)
            if edges:
                item.setToolTip(edges)
            self._runs_list.addItem(item)
        for row in range(self._runs_list.count()):
            item = self._runs_list.item(row)
            # The tick only shows what is on the graph; clicking the row toggles it.
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
        self._sync_list_checks()

    def _sync_list_checks(self) -> None:
        shown = self._shown_keys()
        fixed = self._fixed_key()
        for row in range(self._runs_list.count()):
            item = self._runs_list.item(row)
            key = item.data(Qt.ItemDataRole.UserRole)
            on = key in shown
            item.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
            font = item.font()
            font.setBold(on)
            item.setFont(font)
            base = item.data(Qt.ItemDataRole.UserRole + 1)
            if key == fixed and key != "stored":
                base += "  · latest"
            item.setText(base)

    def _on_run_item_clicked(self, item: QListWidgetItem) -> None:
        """Show a run (or the stored result) beside the latest run, or hide it."""
        key = item.data(Qt.ItemDataRole.UserRole)
        if key is None or key == self._fixed_key():
            return  # the latest run is always on the graph
        if key in self._shown_extra:
            self._shown_extra.discard(key)
        else:
            self._shown_extra.add(key)
        self._redraw_curves()

    def _redraw_curves(self) -> None:
        """Draw the latest run plus the clicked runs (own colours) and, when
        shown, the stored result (white)."""
        for curve in self._curves:
            for plot in (self._n_plot, self._k_plot):
                if curve in plot.listDataItems():
                    plot.removeItem(curve)
        self._curves = []
        shown = self._shown_keys()
        fixed = self._fixed_key()
        stored = self._owner.nk_fit
        if stored is not None and "stored" in shown:
            pen = pg.mkPen(self._STORED_COLOR, width=2)
            for plot, key, name in ((self._n_plot, "n1", "Stored result"), (self._k_plot, "k1", None)):
                values = np.asarray(stored[key])
                ok = np.isfinite(values)
                curve = plot.plot(np.asarray(stored["wavelength"])[ok], values[ok], pen=pen,
                                  symbol="o", symbolSize=4, symbolBrush=pg.mkBrush(*self._STORED_COLOR),
                                  symbolPen=None, name=name)
                curve.setZValue(10 if fixed == "stored" else 0)
                self._curves.append(curve)
        for segment in self._owner.nk_segments:
            if segment["id"] not in shown:
                continue
            color = _NK_SEGMENT_COLORS[(segment["id"] - 1) % len(_NK_SEGMENT_COLORS)]
            latest = segment["id"] == fixed
            label = f"Run {segment['id']}: {segment['start']:g}–{segment['end']:g} nm" + (" (latest)" if latest else "")
            for plot, key, name in ((self._n_plot, "n1", label), (self._k_plot, "k1", None)):
                values = np.asarray(segment[key])
                ok = np.isfinite(values)
                curve = plot.plot(np.asarray(segment["wavelength"])[ok], values[ok],
                                  pen=pg.mkPen(color, width=2), symbol="o", symbolSize=4,
                                  symbolBrush=pg.mkBrush(*color), symbolPen=None, name=name)
                curve.setZValue(10 if latest else 0)
                self._curves.append(curve)
        grid = self._grid()
        if grid is not None:
            self._n_plot.setXRange(float(grid[0]), float(grid[-1]), padding=0.02)
        self._sync_list_checks()

    def _on_finish_clicked(self) -> None:
        segments = list(self._owner.nk_segments)
        if not segments or self._thread is not None:
            return
        dialog = _NkFinishDialog(self, segments)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.finish(dialog.mode(), dialog.chosen_ids())

    def finish(self, mode: str, chosen_ids: list[int]) -> None:
        """Store the chosen runs (merged, smaller error wins) and empty the list."""
        segments = [s for s in self._owner.nk_segments if s["id"] in chosen_ids]
        if not segments:
            return
        merged = _merge_nk_segments(segments)
        merged["d1"] = float(segments[0]["d1"])  # one d1 for all runs (from the Nd fit)
        merged["mode"] = mode
        merged["runs"] = [
            {key: s[key] for key in ("id", "start", "end", "d1", "n1_start", "n1_range", "k1_start", "k1_range")}
            for s in segments
        ]
        owner = self._owner
        owner.nk_fit = merged
        owner.nk_segments = []
        owner._clear_nk_plot()  # a new stored result: plot it again in step 11
        self._shown_extra.clear()
        owner._refresh_table()
        fitted = int(np.isfinite(merged["n1"]).sum())
        owner._summary.setText(
            f"Nk fit stored: {fitted} wavelengths from run(s) "
            f"{', '.join(str(s['id']) for s in segments)} ({'Complete' if mode == 'complete' else 'Segments'}). "
            "Fitted n1 and k1 are in the table."
        )
        self._progress_label.setText(f"Stored {fitted} wavelengths.")
        self._redraw_saved()
        self.refresh()

    def results_cleared(self, had_result: bool, message: Optional[str] = None) -> None:
        """Step 7's values (or the Nd fit's d1) changed: runs and the stored
        result no longer apply."""
        if self._thread is not None:
            self._thread.stop_requested = True
            if self._run is not None:
                self._run["discard"] = True
        self._shown_extra.clear()
        self._redraw_saved()
        self._progress_bar.setValue(0)
        self._progress_label.setText(
            message
            if message is not None
            else "The absolute reflectance changed, so the runs and the stored result "
            "were cleared."
            if had_result
            else "No fit running."
        )
        self.nd_fit_changed()  # d1 box follows the Nd fit (cleared with step 7)


class SpectrometerAppWithNdNk(SpectrometerApp):
    """Main window with two additions (nothing in the original code changes):

    * the Post Process button opens the merged Post Process window (step 1:
      absolute reflectance, or n1 and k1; T is a placeholder);
    * a big Capture button right beside the Measurement Mode and Optical Data
      buttons, with the "Active context ... traces captured" line to its right
      (see _build_ui below; delete that method to go back to the old layout).
    """

    # Created the first time the button is pressed, then kept (remembering).
    _nd_nk_window: Optional[NdNkWindow] = None

    _BIG_CAPTURE_STYLE = """
        QPushButton {
            background-color: #0f5563; color: #ffffff; font-size: 19px; font-weight: 700;
            border: 2px solid #00bcd4; border-radius: 8px; padding: 8px 22px;
        }
        QPushButton:hover { background-color: #137083; }
        QPushButton:pressed { background-color: #0b3f4a; }
        QPushButton:disabled {
            background-color: #1e2229; color: #6b7280; border: 1px solid #2a2f3a;
        }
    """

    def show(self) -> None:
        """main() calls show() once at start-up: open maximized instead."""
        self.showMaximized()

    def _build_ui(self) -> None:
        """Build the original window, then move the Capture button and the
        "traces captured" line next to the mode / trace buttons.

        Before: [mode row .......... traces captured]
                [trace row ........................ Capture]
        After:  [mode row ]  [  big    ]  traces captured
                [trace row]  [ Capture ]
        The same widgets (the original code still updates their text); only
        their place, and the Capture button's size and look, change.
        """
        super()._build_ui()
        root = self.centralWidget().layout()
        mode_row = root.itemAt(0).layout()
        trace_row = root.itemAt(1).layout()
        if (
            mode_row is None
            or trace_row is None
            or mode_row.indexOf(self._geometry_hint) < 0
            or trace_row.indexOf(self._capture_btn) < 0
        ):
            return  # layout not as expected: leave the original one
        root.takeAt(0)
        root.takeAt(0)
        mode_row.removeWidget(self._geometry_hint)
        mode_row.addStretch(1)  # R0 / R60 / T keep their original size
        trace_row.removeWidget(self._capture_btn)

        buttons = QVBoxLayout()
        buttons.setSpacing(10)
        buttons.addLayout(mode_row)
        buttons.addLayout(trace_row)

        capture = self._capture_btn
        capture.setStyleSheet(self._BIG_CAPTURE_STYLE)
        capture.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        # Wide enough for the longest text, so the button never changes size.
        current = capture.text()
        widest = 0
        for verb in ("Capture", "Recapture"):
            for mode in CaptureMode:
                for _key, display in OPTICAL_MODES:
                    capture.setText(f"{verb} {mode.value} ({display})")
                    capture.ensurePolished()
                    widest = max(widest, capture.sizeHint().width())
        capture.setText(current)
        capture.setMinimumSize(max(320, widest), 86)

        top = QHBoxLayout()
        top.setSpacing(16)
        top.addLayout(buttons)
        top.addWidget(capture)
        top.addWidget(self._geometry_hint)
        top.addStretch(1)
        root.insertLayout(0, top)

    def _build_controls_row(self) -> QHBoxLayout:
        """The original Post Process button now opens the merged Post Process
        window (absolute reflectance, or n1 and k1). The button itself, its
        place and look are the original ones; only what it opens changes."""
        row = super()._build_controls_row()
        button = self._post_process_btn
        button.clicked.disconnect()  # was: the original Post Process dialog
        button.clicked.connect(self._open_nd_nk_window)
        button.setEnabled(True)
        button.setToolTip(
            "Post Process: absolute reflectance (R₀, R₆₀ or both), or n₁ and k₁ "
            "(R₀ and R₆₀)"
        )
        self._nd_nk_btn = button  # older name, kept for the tests and other code
        return row

    def _open_nd_nk_window(self) -> None:
        if self._nd_nk_window is None:
            self._nd_nk_window = NdNkWindow(self)
        _show_maximized(self._nd_nk_window)

    def _refresh_status(self) -> None:
        super()._refresh_status()
        # The original enables Post Process only when the selected mode has a
        # capture; the merged window shows what is missing itself, so keep it on.
        self._post_process_btn.setEnabled(True)
        # The main window refreshes its status after every capture, import
        # and reset, so this keeps the Post Process window in step with the data.
        if self._nd_nk_window is not None:
            self._nd_nk_window.sync_measurements()


# ===========================================================================
# END OF ADDED FEATURE: Nd and Nk window
# ===========================================================================


# ===========================================================================
# ADDED FEATURE: Excel import
# ---------------------------------------------------------------------------
# Lets every import button read Excel files as well as CSV files:
#   * Import CSV Data (main window) and Import (Post Process): measurement
#     files can be .xlsx / .xlsm (and old .xls when the xlrd package is
#     installed). The file chooser now lists CSV and Excel files.
#   * Import Si Values (Post Process and the Nd and Nk window): Excel files are
#     still read by the original code (pandas) whenever that works. Only when it
#     cannot (pandas or openpyxl missing, for example in the .exe) is the sheet
#     read here instead, with openpyxl or a small built-in reader, giving the
#     same rows pandas would.
#   * Also fixed on the way in: CSV files saved by Excel in the Windows code
#     page instead of UTF-8, and Excel files renamed to .csv, which the
#     original import could not read.
#
# How it works: the two original reader functions are wrapped. An Excel sheet
# is turned into the same rows a CSV file would give and handed to the
# ORIGINAL parser, so column detection, checks and messages are exactly the
# same as for CSV files. Files that already worked go to the original code
# unchanged. For measurement files, if the sheet as it is fails, it is tried
# again without any title rows above the header row. The first sheet that
# contains data is used. Exporting is not changed.
#
# To remove this feature: delete this block.
# ===========================================================================

import io  # noqa: E402  (imports used only by this block)
import os  # noqa: E402
import tempfile  # noqa: E402
import warnings  # noqa: E402
import xml.etree.ElementTree as ElementTree  # noqa: E402
import zipfile  # noqa: E402
from contextlib import contextmanager  # noqa: E402

_EXCEL_SUFFIXES = frozenset({".xlsx", ".xlsm", ".xls"})
_ZIP_SIGNATURE = b"PK\x03\x04"  # .xlsx / .xlsm files are zip archives
_OLE_SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"  # old binary .xls files

# The original functions, kept so the wrappers below can call them.
_original_parse_unified_measurement_csv = parse_unified_measurement_csv
_original_load_table_rows = _load_table_rows


def _file_signature(path: Path, size: int = 8) -> bytes:
    try:
        with open(path, "rb") as handle:
            return handle.read(size)
    except OSError:
        return b""


def _looks_like_excel(path: Path) -> bool:
    signature = _file_signature(path)
    return (
        path.suffix.lower() in _EXCEL_SUFFIXES
        or signature.startswith(_ZIP_SIGNATURE)
        or signature.startswith(_OLE_SIGNATURE)
    )


def _excel_cell_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _xlsx_sheets_openpyxl(path: Path) -> list[list[list[str]]]:
    import openpyxl  # ImportError -> the built-in reader is used instead

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Opened from bytes so the file name's extension does not matter.
        workbook = openpyxl.load_workbook(
            io.BytesIO(path.read_bytes()), read_only=True, data_only=True
        )
    try:
        return [
            [
                [_excel_cell_text(value) for value in row]
                for row in sheet.iter_rows(values_only=True)
            ]
            for sheet in workbook.worksheets
        ]
    finally:
        workbook.close()


def _xlsx_sheets_builtin(path: Path) -> list[list[list[str]]]:
    """Minimal .xlsx reader (values only) using just the standard library."""
    main_ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    rel_ns = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
    pkg_ns = "{http://schemas.openxmlformats.org/package/2006/relationships}"
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        shared: list[str] = []
        if "xl/sharedStrings.xml" in names:
            root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.iter(f"{main_ns}si"):
                shared.append("".join(t.text or "" for t in item.iter(f"{main_ns}t")))
        targets: dict[str, str] = {}
        if "xl/_rels/workbook.xml.rels" in names:
            rels = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
            for rel in rels.iter(f"{pkg_ns}Relationship"):
                target = rel.get("Target", "").lstrip("/")
                if not target.startswith("xl/"):
                    target = "xl/" + target
                targets[rel.get("Id", "")] = target
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        sheet_paths = [
            targets.get(sheet.get(f"{rel_ns}id", ""), "")
            for sheet in workbook.iter(f"{main_ns}sheet")
        ]
        sheets: list[list[list[str]]] = []
        for sheet_path in sheet_paths:
            if sheet_path not in names:
                continue
            root = ElementTree.fromstring(archive.read(sheet_path))
            rows: list[list[str]] = []
            for row in root.iter(f"{main_ns}row"):
                cells: dict[int, str] = {}
                for index, cell in enumerate(row.iter(f"{main_ns}c")):
                    ref = cell.get("r", "")
                    letters = "".join(ch for ch in ref if ch.isalpha())
                    column = index
                    if letters:
                        column = 0
                        for ch in letters.upper():
                            column = column * 26 + (ord(ch) - ord("A") + 1)
                        column -= 1
                    kind = cell.get("t", "n")
                    value_el = cell.find(f"{main_ns}v")
                    text = value_el.text if value_el is not None and value_el.text else ""
                    if kind == "s" and text:
                        text = shared[int(text)]
                    elif kind == "inlineStr":
                        text = "".join(t.text or "" for t in cell.iter(f"{main_ns}t"))
                    cells[column] = text.strip()
                width = max(cells) + 1 if cells else 0
                rows.append([cells.get(column, "") for column in range(width)])
            sheets.append(rows)
        return sheets


def _xls_sheets(path: Path) -> list[list[list[str]]]:
    try:
        import xlrd
    except ImportError as exc:
        raise MeasurementCsvError(
            "This is an old-format Excel file (.xls). Open it in Excel and use "
            "File > Save As > Excel Workbook (.xlsx), then import the .xlsx file."
        ) from exc
    book = xlrd.open_workbook(str(path))
    return [
        [
            [_excel_cell_text(sheet.cell_value(row, column)) for column in range(sheet.ncols)]
            for row in range(sheet.nrows)
        ]
        for sheet in book.sheets()
    ]


def _drop_blank_rows(rows: list[list[str]]) -> list[list[str]]:
    """Drop blank rows and pad every row to the same width, as pandas does.

    pandas gives every row the full width of the sheet, so a title row like
    "Silicon data" still has empty cells under the n and k columns; the
    original Si parser relies on that.
    """
    trimmed = []
    for row in rows:
        while row and not row[-1]:
            row = row[:-1]
        if any(cell for cell in row):
            trimmed.append(list(row))
    width = max((len(row) for row in trimmed), default=0)
    return [row + [""] * (width - len(row)) for row in trimmed]


def _trim_excel_rows(rows: list[list[str]]) -> list[list[str]]:
    """Blank rows dropped, and title rows above the header removed.

    Only used as a second try for measurement files: the header is taken to be
    the row just above the first row whose cells are all numbers.
    """
    cleaned = _drop_blank_rows(rows)
    first_data = next(
        (index for index, row in enumerate(cleaned) if _row_looks_numeric(row)), None
    )
    if first_data is not None and first_data > 1:
        cleaned = cleaned[first_data - 1 :]  # keep the header row just above the data
    return cleaned


def _read_excel_rows(path: Path) -> list[list[str]]:
    """Rows (as strings) of the first sheet in an Excel file that has data.

    Every row of the sheet is kept (blank rows aside), as pandas would give
    them; nothing above the header is removed here.
    """
    try:
        if _file_signature(path).startswith(_OLE_SIGNATURE):
            sheets = _xls_sheets(path)
        else:
            try:
                sheets = _xlsx_sheets_openpyxl(path)
            except ImportError:
                sheets = _xlsx_sheets_builtin(path)
            except Exception as openpyxl_error:
                # Some programs write workbooks openpyxl rejects; try the
                # built-in reader before giving up.
                try:
                    sheets = _xlsx_sheets_builtin(path)
                except Exception:
                    raise openpyxl_error
    except MeasurementCsvError:
        raise
    except Exception as exc:
        raise MeasurementCsvError(f"Could not read Excel file: {exc}") from exc
    for rows in sheets:
        cleaned = _drop_blank_rows(rows)
        if cleaned:
            return cleaned
    raise MeasurementCsvError("The Excel file does not contain any data.")


def _read_text_any_encoding(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")  # never fails


@contextmanager
def _temporary_csv(rows: Optional[list[list[str]]] = None, text: Optional[str] = None):
    """A temporary UTF-8 CSV file with these rows (or this text)."""
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".csv", delete=False, encoding="utf-8", newline=""
    )
    try:
        with handle:
            if rows is not None:
                csv.writer(handle).writerows(rows)
            else:
                handle.write(text or "")
        yield Path(handle.name)
    finally:
        try:
            os.remove(handle.name)
        except OSError:
            pass


def parse_unified_measurement_csv(  # noqa: F811  (wraps the original)
    path: Path,
) -> tuple[np.ndarray, dict[Optional[str], dict[str, np.ndarray]]]:
    """Excel-aware wrapper around the original measurement-file parser."""
    path = Path(path)
    if _looks_like_excel(path):
        rows = _read_excel_rows(path)
        try:
            with _temporary_csv(rows=rows) as temp_path:
                return _original_parse_unified_measurement_csv(temp_path)
        except MeasurementCsvError as first_error:
            trimmed = _trim_excel_rows(rows)  # second try: skip title rows
            if trimmed == rows:
                raise
            try:
                with _temporary_csv(rows=trimmed) as temp_path:
                    return _original_parse_unified_measurement_csv(temp_path)
            except MeasurementCsvError:
                raise first_error from None
    try:
        return _original_parse_unified_measurement_csv(path)
    except UnicodeDecodeError:
        try:
            text = _read_text_any_encoding(path)
        except OSError as exc:
            raise MeasurementCsvError(f"Could not read file: {exc}") from exc
        with _temporary_csv(text=text) as temp_path:
            return _original_parse_unified_measurement_csv(temp_path)


def _load_table_rows(path: Path) -> list[list[str]]:  # noqa: F811  (wraps the original)
    """Excel-aware wrapper around the original Si-file reader.

    .xlsx / .xls files go to the original (pandas) first and its rows are used
    unchanged. Only if it fails, or for Excel files it would read as text
    (.xlsm, or an Excel file renamed .csv), is the sheet read here.
    """
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        try:
            return _original_load_table_rows(path)
        except Exception:  # pandas / openpyxl missing or could not read it
            return _read_excel_rows(path)
    if _looks_like_excel(path):
        return _read_excel_rows(path)
    try:
        return _original_load_table_rows(path)
    except UnicodeDecodeError:
        try:
            text = _read_text_any_encoding(path)
        except OSError as exc:
            raise MeasurementCsvError(f"Could not read file: {exc}") from exc
        return list(csv.reader(text.splitlines()))


# File chooser: the two measurement import buttons only offered CSV files.
# While one of them runs, the CSV-only filter is swapped for one that also
# lists Excel files.
_CSV_ONLY_FILTER = "CSV Files (*.csv);;All Files (*)"
_CSV_AND_EXCEL_FILTER = (
    "CSV or Excel Files (*.csv *.xlsx *.xlsm *.xls);;CSV Files (*.csv);;"
    "Excel Files (*.xlsx *.xlsm *.xls);;All Files (*)"
)


@contextmanager
def _file_chooser_with_excel():
    original = QFileDialog.getOpenFileName

    def get_open_file_name(*args, **kwargs):
        args = list(args)
        if len(args) >= 4 and args[3] == _CSV_ONLY_FILTER:
            args[3] = _CSV_AND_EXCEL_FILTER
        if kwargs.get("filter") == _CSV_ONLY_FILTER:
            kwargs["filter"] = _CSV_AND_EXCEL_FILTER
        return original(*args, **kwargs)

    QFileDialog.getOpenFileName = get_open_file_name
    try:
        yield
    finally:
        QFileDialog.getOpenFileName = original


def _with_excel_in_file_chooser(method: Callable[..., object]) -> Callable[..., object]:
    def wrapper(self, *_signal_args: object) -> object:
        with _file_chooser_with_excel():
            return method(self)

    wrapper.__name__ = method.__name__
    wrapper.__doc__ = method.__doc__
    return wrapper


SpectrometerApp._import_csv_data = _with_excel_in_file_chooser(
    SpectrometerApp._import_csv_data
)
PostProcessDialog._import_measurement_csv = _with_excel_in_file_chooser(
    PostProcessDialog._import_measurement_csv
)

# ===========================================================================
# END OF ADDED FEATURE: Excel import
# ===========================================================================


def main() -> int:
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName("Multi-Function Optical Analyzer")
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    window = SpectrometerAppWithNdNk()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
