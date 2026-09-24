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
from typing import Optional

import numpy as np
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter
from PyQt6.QtCore import (
    QAbstractTableModel,
    QMetaObject,
    QModelIndex,
    Qt,
    QThread,
    QTimer,
    Q_ARG,
)
from PyQt6.QtGui import QAction, QCloseEvent, QFont, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
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
            if kind == "avg":
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
    ) -> None:
        super().__init__(parent)
        self._mode_display = mode_display
        self._mode_key = mode_key
        self._raw_wavelengths = np.asarray(wavelengths, dtype=np.float64)
        self._wl_min = float(self._raw_wavelengths.min())
        self._wl_max = float(self._raw_wavelengths.max())
        self.setWindowTitle("Post-Processing - Data Viewer")
        self.setModal(True)
        self.resize(1000, 720)
        self.setStyleSheet(
            """
            QDialog {
                background-color: #12141a;
                color: #e8eaed;
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
                padding: 8px 16px;
                min-height: 36px;
            }
            QPushButton:hover {
                background-color: #353b48;
            }
            QToolButton {
                background-color: #2a2f3a;
                border: 1px solid #3d4450;
                border-radius: 6px;
                padding: 8px 16px;
                min-height: 36px;
                color: #e8eaed;
            }
            QToolButton:hover {
                background-color: #353b48;
            }
            QToolButton::menu-indicator {
                subcontrol-origin: padding;
                subcontrol-position: center right;
                padding-right: 4px;
            }
            QMenu {
                background-color: #1e2229;
                color: #e8eaed;
                border: 1px solid #3d4450;
            }
            QMenu::item:selected {
                background-color: #2a4568;
            }
            QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #1e2229;
                border: 1px solid #3d4450;
                border-radius: 4px;
                padding: 4px 8px;
                min-height: 26px;
                color: #e8eaed;
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
        self._summary.setStyleSheet("font-weight: 600;")
        root.addWidget(self._summary)

        root.addWidget(self._build_interpolation_group())
        root.addLayout(self._build_average_controls())

        self._model = SpectrumTableModel(
            wavelengths, baseline, standard, sample, parent=self
        )
        self._table = self._make_table_view()
        self._table.setModel(self._model)
        root.addWidget(self._table, stretch=1)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        self._back_btn = QPushButton("Back")
        self._back_btn.setMinimumWidth(120)
        self._back_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._back_btn.setDefault(True)
        self._back_btn.clicked.connect(self.accept)
        button_row.addWidget(self._back_btn)
        root.addLayout(button_row)

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

    def _build_interpolation_group(self) -> QGroupBox:
        group = QGroupBox("Grid Interpolation")
        layout = QHBoxLayout(group)
        layout.setSpacing(10)

        start_default = float(math.ceil(self._wl_min))
        end_default = float(math.floor(self._wl_max))
        if start_default >= end_default:
            start_default = self._wl_min
            end_default = self._wl_max

        layout.addWidget(QLabel("Start Wavelength (nm):"))
        self._start_wl_spin = QDoubleSpinBox()
        self._start_wl_spin.setDecimals(3)
        self._start_wl_spin.setRange(self._wl_min, self._wl_max)
        self._start_wl_spin.setValue(start_default)
        self._start_wl_spin.setSingleStep(1.0)
        layout.addWidget(self._start_wl_spin)

        layout.addWidget(QLabel("End Wavelength (nm):"))
        self._end_wl_spin = QDoubleSpinBox()
        self._end_wl_spin.setDecimals(3)
        self._end_wl_spin.setRange(self._wl_min, self._wl_max)
        self._end_wl_spin.setValue(end_default)
        self._end_wl_spin.setSingleStep(1.0)
        layout.addWidget(self._end_wl_spin)

        layout.addWidget(QLabel("Step Size (nm):"))
        self._step_wl_spin = QDoubleSpinBox()
        self._step_wl_spin.setDecimals(3)
        self._step_wl_spin.setRange(0.1, 50.0)
        self._step_wl_spin.setValue(1.0)
        self._step_wl_spin.setSingleStep(1.0)
        layout.addWidget(self._step_wl_spin)

        layout.addWidget(QLabel("Trace:"))
        self._interp_trace_combo = QComboBox()
        self._interp_trace_combo.setMinimumWidth(110)
        layout.addWidget(self._interp_trace_combo)

        layout.addWidget(QLabel("Method:"))
        self._interp_method_combo = QComboBox()
        self._interp_method_combo.addItems(["Linear", "Cubic Spline"])
        self._interp_method_combo.setMinimumWidth(120)
        layout.addWidget(self._interp_method_combo)

        self._interpolate_btn = QPushButton("Interpolate Grid")
        self._interpolate_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._interpolate_btn.clicked.connect(self._interpolate_grid)
        layout.addWidget(self._interpolate_btn)

        layout.addStretch(1)
        return group

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
        self._interpolate_btn.setEnabled(self._interp_trace_combo.count() > 0)

    def _build_average_controls(self) -> QHBoxLayout:
        controls = QHBoxLayout()
        controls.setSpacing(10)

        controls.addWidget(QLabel("Window Size (Points):"))
        self._window_spin = QSpinBox()
        self._window_spin.setRange(2, 50)
        self._window_spin.setValue(5)
        self._window_spin.setSingleStep(1)
        self._window_spin.setMinimumWidth(80)
        controls.addWidget(self._window_spin)

        self._avg_menu_btn = QToolButton(self)
        self._avg_menu_btn.setText("Average Trace")
        self._avg_menu_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._avg_menu_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self._avg_menu_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._avg_menu_btn.setMinimumWidth(140)
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
        controls.addWidget(self._avg_menu_btn)

        self._avg_all_btn = QPushButton("Average All")
        self._avg_all_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._avg_all_btn.setToolTip(
            "Average Baseline, Standard, and Sample at once "
            "(each result inserted beside its source column)"
        )
        self._avg_all_btn.clicked.connect(self._average_all_traces)
        controls.addWidget(self._avg_all_btn)

        self._export_table_btn = QPushButton("Export Processed Table (CSV)")
        self._export_table_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._export_table_btn.setToolTip(
            "Export all table columns (raw, averages, and interpolation) side-by-side"
        )
        self._export_table_btn.clicked.connect(self._export_processed_table)
        controls.addWidget(self._export_table_btn)

        controls.addStretch(1)
        return controls

    def _make_table_view(self) -> QTableView:
        table = QTableView(self)
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QTableView.SelectionMode.ExtendedSelection)
        table.setSortingEnabled(False)
        table.verticalHeader().setVisible(True)
        table.verticalHeader().setDefaultSectionSize(22)
        table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        table.setWordWrap(False)
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
            interp_wl = _build_uniform_wavelengths(
                self._start_wl_spin.value(),
                self._end_wl_spin.value(),
                self._step_wl_spin.value(),
                self._wl_min,
                self._wl_max,
            )
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
            f"step {self._step_wl_spin.value():.3f} nm, {method_key})"
        )

    def _export_processed_table(self) -> None:
        headers, rows = self._model.export_rows()
        # Ensure every row has exactly one string per column (blank for padded cells).
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
        self._latest_wavelengths: Optional[np.ndarray] = None
        self._latest_intensities: Optional[np.ndarray] = None

        # Hierarchical storage: optical mode key → capture slot → intensities
        self.spectra_data: dict[str, dict[str, Optional[np.ndarray]]] = (
            _build_spectra_data()
        )
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

        integ_label = QLabel("Integration time (ms):")
        row.addWidget(integ_label)

        self._integration_spin = QSpinBox()
        self._integration_spin.setRange(1, 10_000)
        self._integration_spin.setValue(100)
        self._integration_spin.setSuffix(" ms")
        self._integration_spin.setMinimumWidth(110)
        self._integration_spin.valueChanged.connect(self._on_integration_changed)
        row.addWidget(self._integration_spin)

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

    def _reset_current_geometry(self) -> None:
        geometry = self._active_geometry
        self.spectra_data[geometry] = _empty_geometry_slot()
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
        dialog = PostProcessDialog(
            mode_display=self._active_display(),
            mode_key=self._active_geometry,
            wavelengths=wavelengths,
            baseline=slot[CaptureMode.BASELINE.value],
            standard=slot[CaptureMode.STANDARD.value],
            sample=slot[CaptureMode.SAMPLE.value],
            parent=self,
        )
        # Modal exec keeps live acquisition running on the worker thread;
        # stored capture arrays are only read, never mutated.
        dialog.exec()
        self._refresh_status()

    def _on_integration_changed(self, value_ms: int) -> None:
        microseconds = int(value_ms) * 1000
        QMetaObject.invokeMethod(
            self._worker,
            "set_integration_time",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(int, microseconds),
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
                self._on_integration_changed(self._integration_spin.value())
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
