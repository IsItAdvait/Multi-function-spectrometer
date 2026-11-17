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
#
# The post-processing sidebar is split into two tabs ("Original" and
# "Nd and Nk"). The Nd and Nk tab holds Silicon Calibration and the new
# Si Reflectance step (bare-silicon reflectance at 0° and 60°, s-polarized).
# ===========================================================================

import threading  # noqa: E402  (imports used only by this block)
import time  # noqa: E402
from functools import partial  # noqa: E402

from PyQt6.QtCore import QElapsedTimer, QObject, pyqtSignal, pyqtSlot  # noqa: E402
from PyQt6.QtWidgets import QProgressBar, QTabWidget  # noqa: E402

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


# ---------------------------------------------------------------------------
# Silicon reflectance (used by the "Nd and Nk" tab)
# ---------------------------------------------------------------------------

# Angles of incidence in degrees, matching the R₀ and R₆₀ optical modes.
_SI_R_ANGLES_DEG: tuple[float, float] = (0.0, 60.0)
# Table column kinds used for the calculated Si reflectance.
_SI_R_KINDS: frozenset[str] = frozenset({"si_r_wl", "si_r0", "si_r60"})


def bare_substrate_reflectance_s(
    n2: np.ndarray, k2: np.ndarray, angle_deg: float, n0: float = 1.0
) -> np.ndarray:
    """Reflectance |r|² of an uncoated substrate in air, s-polarized.

    This is the formula of Optical_Nd_V7S.py / Optical_NK_V7S.py with no film
    (d1 = 0): the input admittance Y then equals the substrate admittance η2,
    so r = (η0 − η2) / (η0 + η2) with η = N·cos θ (s-wave) and N = n − ik.
    At 0° the s and p polarizations are identical. Works element-wise on
    arrays of n2 and k2; NaN inputs give NaN.
    """
    with np.errstate(all="ignore"):
        n_sub = np.asarray(n2, dtype=np.float64) - 1j * np.asarray(k2, dtype=np.float64)
        n_amb = complex(n0, 0.0)
        theta0 = math.radians(angle_deg)
        theta2 = np.arcsin(n_amb / n_sub * math.sin(theta0))  # Snell's law
        eta0 = n_amb * math.cos(theta0)
        eta2 = n_sub * np.cos(theta2)
        r = (eta0 - eta2) / (eta0 + eta2)
        return (r * np.conj(r)).real


def _replace_model_columns(
    model: SpectrumTableModel,
    kinds: frozenset[str],
    new_columns: list[tuple[str, np.ndarray, str]],
) -> None:
    """Remove all columns of ``kinds`` from the table and append ``new_columns``.

    Same approach as SpectrumTableModel's own Si methods (full model reset;
    the row count grows if a new column is longer than the table).
    """
    model.beginResetModel()
    for _header, series, _kind in new_columns:
        model._n = max(model._n, int(np.asarray(series).size))
    model._columns = [column for column in model._columns if column[2] not in kinds]
    model._columns.extend(new_columns)
    model.endResetModel()


class PostProcessDialogWithProgress(PostProcessDialog):
    """PostProcessDialog with progress pop-ups, sidebar tabs and new Nd/Nk tools.

    The original handler methods are called unchanged. Only the signal
    connections of these buttons/menu items are swapped for wrappers:
    Average trace (Baseline/Standard/Sample), Average All,
    Interpolate (Baseline/Standard/Sample), Interpolate all,
    Relative Reflectance, Interpolate Relative Reflectance,
    Interpolate Si (n and k), and the new Calculate Si Reflectance.
    Import, Export, Import Si Values and Back keep their original wiring.

    New long-running features should call ``self.run_long_task(...)``.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Calculated Si reflectance: {"wavelength", "R0", "R60", "source"}.
        self.si_reflectance: Optional[dict[str, object]] = None
        # The Si wavelength / n / k arrays it was calculated from.
        self._si_reflectance_inputs: Optional[tuple[np.ndarray, ...]] = None
        self._attach_progress_popups()

    # -- Sidebar tabs (added) -------------------------------------------
    # Tab titles: rename here later if needed.
    _ORIGINAL_TAB_TITLE = "Original"
    _ND_NK_TAB_TITLE = "Nd and Nk"

    def _build_sidebar(self) -> QWidget:
        """Two-tab sidebar instead of one scrolling column.

        Overrides PostProcessDialog._build_sidebar (which stays unchanged) and
        reuses the original group builders, so every control keeps its
        original behaviour. The table on the right stays visible on both tabs.
          "Original"  : Average, Interpolation, Reflectance
          "Nd and Nk" : Silicon Calibration, Si Reflectance
                        (future n / k / d tools go here)
        """
        self._sidebar_tabs = QTabWidget()
        self._sidebar_tabs.setObjectName("sidebarTabs")
        self._sidebar_tabs.setStyleSheet(
            """
            QTabWidget#sidebarTabs::pane {
                border: 1px solid #3d4450;
                border-radius: 6px;
                top: -1px;
            }
            QTabWidget#sidebarTabs > QTabBar::tab {
                background-color: #1a1e26;
                color: #8b93a3;
                border: 1px solid #3a4150;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                padding: 6px 16px;
                margin-right: 2px;
                font-weight: 600;
            }
            QTabWidget#sidebarTabs > QTabBar::tab:selected {
                background-color: #123e4f;
                color: #e8fbff;
                border: 1px solid #00bcd4;
                border-bottom: none;
            }
            QTabWidget#sidebarTabs > QTabBar::tab:hover:!selected {
                background-color: #353b48;
                color: #e8eaed;
            }
            """
        )
        self._sidebar_tabs.addTab(
            self._make_sidebar_page(
                self._build_average_group(),
                self._build_interpolation_group(),
                self._build_reflectance_group(),
            ),
            self._ORIGINAL_TAB_TITLE,
        )
        self._sidebar_tabs.addTab(
            self._make_sidebar_page(
                self._build_silicon_group(),
                self._build_si_reflectance_group(),
            ),
            self._ND_NK_TAB_TITLE,
        )
        self._sidebar_tabs.setFixedWidth(336)
        self._sidebar_tabs.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding
        )
        return self._sidebar_tabs

    @staticmethod
    def _make_sidebar_page(*groups: QWidget) -> QScrollArea:
        """One scrollable tab page, laid out like the original sidebar."""
        content = QWidget()
        content.setMinimumWidth(300)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(6, 8, 6, 6)
        layout.setSpacing(10)
        for group in groups:
            layout.addWidget(group)
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        return scroll
    # -- End of sidebar tabs --------------------------------------------

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
            (
                self._si_reflectance_btn.clicked,
                "Calculating Si Reflectance",
                self._calculate_si_reflectance,
            ),
        )
        for signal, title, action in wiring:
            signal.disconnect()  # remove the original direct connection
            signal.connect(partial(self._run_with_progress, title, action))

        # Keep the Si reflectance consistent when Si values or measurements
        # are imported again. These extra connections run after the original
        # handlers and only clear the added Si reflectance columns.
        self._import_si_btn.clicked.connect(self._sync_si_reflectance)
        self._import_csv_btn.clicked.connect(self._sync_si_reflectance)

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
        self._sync_si_reflectance()

    # -- Si reflectance (added) -----------------------------------------
    def _build_si_reflectance_group(self) -> QGroupBox:
        """Nd and Nk tab: calculate the reflectance of bare silicon."""
        group = QGroupBox("Si Reflectance")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)
        note = QLabel(
            "Bare silicon in air, s-polarized, at 0° and 60°. Uses the "
            "interpolated Si n and k if available, otherwise the imported values."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #9aa3b2;")
        layout.addWidget(note)

        self._si_reflectance_btn = QPushButton("Calculate Si Reflectance")
        self._si_reflectance_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._si_reflectance_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._si_reflectance_btn.setToolTip(
            "Add Si R0 (0°) and Si R60 (60°, s-pol) columns calculated from Si n and k"
        )
        self._si_reflectance_btn.clicked.connect(self._calculate_si_reflectance)
        layout.addWidget(self._si_reflectance_btn)
        return group

    def _si_reflectance_source(
        self,
    ) -> Optional[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
        """Interpolated Si n, k when available, otherwise the imported values."""
        if (
            self.si_interp_wavelength is not None
            and self.si_interp_n is not None
            and self.si_interp_k is not None
            and self.si_interp_wavelength.size > 0
        ):
            return (
                "interpolated Si n, k",
                self.si_interp_wavelength,
                self.si_interp_n,
                self.si_interp_k,
            )
        if (
            self.si_wavelength is not None
            and self.si_n is not None
            and self.si_k is not None
            and self.si_wavelength.size > 0
        ):
            return ("imported Si n, k", self.si_wavelength, self.si_n, self.si_k)
        return None

    def _calculate_si_reflectance(self) -> bool:
        """Add Si R0 (0°) and Si R60 (60°, s-pol) columns to the table."""
        source = self._si_reflectance_source()
        if source is None:
            QMessageBox.warning(
                self,
                "Silicon Data Required",
                "Import Si values before calculating Si reflectance.",
            )
            return False
        label, wavelengths, n_values, k_values = source
        try:
            r0 = bare_substrate_reflectance_s(n_values, k_values, _SI_R_ANGLES_DEG[0])
            r60 = bare_substrate_reflectance_s(n_values, k_values, _SI_R_ANGLES_DEG[1])
        except Exception as exc:
            logger.exception("Si reflectance calculation failed")
            QMessageBox.critical(self, "Calculation Failed", str(exc))
            return False

        wl = np.asarray(wavelengths, dtype=np.float64).copy()
        _replace_model_columns(
            self._model,
            _SI_R_KINDS,
            [
                ("Si R Wavelength (nm)", wl, "si_r_wl"),
                ("Si R0 (0°)", r0, "si_r0"),
                ("Si R60 (60°, s-pol)", r60, "si_r60"),
            ],
        )
        self.si_reflectance = {
            "wavelength": wl.copy(),
            "R0": r0.copy(),
            "R60": r60.copy(),
            "source": label,
        }
        self._si_reflectance_inputs = (wavelengths, n_values, k_values)

        self._table.resizeColumnsToContents()
        first_col = next(
            index
            for index, column in enumerate(self._model._columns)
            if column[2] == "si_r_wl"
        )
        self._table.scrollTo(self._model.index(0, first_col))
        finite = np.isfinite(r0) & np.isfinite(r60)
        ranges = ""
        if finite.any():
            ranges = (
                f"  ·  R0 {r0[finite].min():.4f}–{r0[finite].max():.4f}"
                f"  ·  R60 {r60[finite].min():.4f}–{r60[finite].max():.4f}"
            )
        self._summary.setText(
            f"Optical mode: {self._mode_display}  ·  Si reflectance: "
            f"{wl.size} points from {label} "
            f"({wl[0]:.3f}–{wl[-1]:.3f} nm){ranges}"
        )
        return True

    def _sync_si_reflectance(self, *_signal_args: object) -> None:
        """Keep the Si reflectance columns consistent with the Si n, k data.

        - Si n, k changed (re-import / re-interpolate): clear the result.
        - Columns removed by the original measurement Import (it drops all
          added columns) while Si n, k is unchanged: put them back, since
          Si reflectance does not depend on the measurements.
        """
        if self._si_reflectance_inputs is None or self.si_reflectance is None:
            return
        source = self._si_reflectance_source()
        same_inputs = source is not None and all(
            current is used
            for current, used in zip(source[1:], self._si_reflectance_inputs)
        )
        has_columns = any(kind in _SI_R_KINDS for _h, _s, kind in self._model._columns)
        if same_inputs:
            if not has_columns:
                stored = self.si_reflectance
                _replace_model_columns(
                    self._model,
                    _SI_R_KINDS,
                    [
                        ("Si R Wavelength (nm)", np.array(stored["wavelength"]), "si_r_wl"),
                        ("Si R0 (0°)", np.array(stored["R0"]), "si_r0"),
                        ("Si R60 (60°, s-pol)", np.array(stored["R60"]), "si_r60"),
                    ],
                )
                self._table.resizeColumnsToContents()
            return
        if has_columns:
            _replace_model_columns(self._model, _SI_R_KINDS, [])
            self._table.resizeColumnsToContents()
        self.si_reflectance = None
        self._si_reflectance_inputs = None
        self._summary.setText(
            self._summary.text()
            + "  ·  Si reflectance cleared (Si values changed, recalculate)"
        )
    # -- End of Si reflectance ------------------------------------------


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


def main() -> int:
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName("Multi-Function Optical Analyzer")
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    window = SpectrometerApp()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())