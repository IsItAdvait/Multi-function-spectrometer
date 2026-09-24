"""
Hardware backend for Ocean Optics FLAME spectrometers.

Provides a QThread-friendly ``SpectrometerWorker`` that acquires spectra via
python-seabreeze, with a synthetic ``MockSpectrometer`` fallback when no
device is available. Contains no UI widgets.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Deque, Optional, Protocol, runtime_checkable

import numpy as np
from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal, pyqtSlot

logger = logging.getLogger(__name__)

# Minimum polling interval so very short integration times do not starve the
# event loop (microseconds).
_MIN_POLL_INTERVAL_US: int = 10_000  # 10 ms

_MOCK_WAVELENGTH_START_NM: float = 350.0
_MOCK_WAVELENGTH_END_NM: float = 1000.0
_MOCK_NUM_PIXELS: int = 2048


@runtime_checkable
class SpectrometerDevice(Protocol):
    """Minimal interface shared by real and mock spectrometers."""

    def wavelengths(self) -> np.ndarray: ...

    def intensities(self) -> np.ndarray: ...

    def integration_time_micros(self, microseconds: int) -> None: ...

    def close(self) -> None: ...


class MockSpectrometer:
    """Synthetic spectrometer producing a Gaussian peak over 350–1000 nm."""

    def __init__(self) -> None:
        self._wavelengths: np.ndarray = np.linspace(
            _MOCK_WAVELENGTH_START_NM,
            _MOCK_WAVELENGTH_END_NM,
            _MOCK_NUM_PIXELS,
            dtype=np.float64,
        )
        self._integration_time_us: int = 100_000
        # Peak centred in the green band; amplitude scales lightly with
        # integration time so mock behaviour feels plausible.
        self._peak_center_nm: float = 550.0
        self._peak_width_nm: float = 40.0
        self._base_amplitude: float = 40_000.0
        self._noise_level: float = 200.0

    def wavelengths(self) -> np.ndarray:
        return self._wavelengths.copy()

    def intensities(self) -> np.ndarray:
        scale = self._integration_time_us / 100_000.0
        amplitude = self._base_amplitude * max(scale, 0.05)
        spectrum = amplitude * np.exp(
            -0.5
            * ((self._wavelengths - self._peak_center_nm) / self._peak_width_nm) ** 2
        )
        noise = np.random.normal(0.0, self._noise_level, size=self._wavelengths.shape)
        return np.clip(spectrum + noise, 0.0, 65535.0).astype(np.float64)

    def integration_time_micros(self, microseconds: int) -> None:
        if microseconds <= 0:
            raise ValueError("Integration time must be positive")
        self._integration_time_us = int(microseconds)

    def close(self) -> None:
        pass


def _open_spectrometer() -> tuple[SpectrometerDevice, bool]:
    """
    Try to open the first available Ocean Optics device.

    Returns ``(device, is_hardware)``. On failure, returns a
    ``MockSpectrometer`` with ``is_hardware=False``.
    """
    try:
        from seabreeze.spectrometers import Spectrometer

        device = Spectrometer.from_first_available()
        logger.info("Connected to hardware spectrometer: %s", device.model)
        return device, True
    except Exception as exc:  # seabreeze raises SeaBreezeError / OSError / etc.
        logger.warning(
            "No Ocean Optics spectrometer found (%s); using MockSpectrometer",
            exc,
        )
        return MockSpectrometer(), False


class SpectrometerWorker(QObject):
    """
    QObject worker that owns spectrometer I/O and must live on a ``QThread``.

    Typical wiring from the GUI thread::

        thread = QThread()
        worker = SpectrometerWorker()
        worker.moveToThread(thread)
        thread.started.connect(worker.initialize)
        worker.data_acquired.connect(on_spectrum)
        thread.start()
    """

    data_acquired = pyqtSignal(np.ndarray, np.ndarray)
    status_message = pyqtSignal(str)
    device_error = pyqtSignal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._device: Optional[SpectrometerDevice] = None
        self._is_hardware: bool = False
        self._wavelengths: Optional[np.ndarray] = None
        self._integration_time_us: int = 100_000
        self._scans_to_average: int = 1
        self._scan_buffer: Deque[np.ndarray] = deque(maxlen=1)
        self._acquiring: bool = False
        self._timer: Optional[QTimer] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @pyqtSlot()
    def initialize(self) -> None:
        """Open hardware (or mock) and prepare the acquisition timer.

        Call this once after the worker has been moved onto its thread
        (e.g. connect to ``QThread.started``). Safe to call again after
        ``stop_acquisition`` to reconnect.
        """
        if self._acquiring:
            self._acquiring = False
            if self._timer is not None:
                self._timer.stop()

        self._release_device()
        self._clear_scan_buffer()
        self._device, self._is_hardware = _open_spectrometer()
        try:
            self._device.integration_time_micros(self._integration_time_us)
            self._wavelengths = np.asarray(self._device.wavelengths(), dtype=np.float64)
        except Exception as exc:
            message = f"Failed to initialise spectrometer: {exc}"
            logger.error(message)
            self.device_error.emit(message)
            self._release_device()
            return

        if self._timer is None:
            self._timer = QTimer(self)
            self._timer.setTimerType(Qt.TimerType.PreciseTimer)
            self._timer.timeout.connect(self._acquire_once)

        mode = "hardware" if self._is_hardware else "mock"
        self.status_message.emit(
            f"Spectrometer ready ({mode}): {self._wavelengths.size} pixels"
        )

    # ------------------------------------------------------------------
    # Thread-safe control slots
    # ------------------------------------------------------------------

    @pyqtSlot(int)
    def set_integration_time(self, microseconds: int) -> None:
        """Update integration time; adjusts the poll timer if running."""
        if microseconds <= 0:
            self.device_error.emit(
                f"Invalid integration time: {microseconds} µs (must be > 0)"
            )
            return

        self._integration_time_us = int(microseconds)
        self._clear_scan_buffer()
        if self._device is None:
            return

        try:
            self._device.integration_time_micros(self._integration_time_us)
        except Exception as exc:
            message = f"Failed to set integration time: {exc}"
            logger.error(message)
            self.device_error.emit(message)
            return

        if self._acquiring and self._timer is not None:
            self._timer.setInterval(self._poll_interval_ms())

        self.status_message.emit(
            f"Integration time set to {self._integration_time_us} µs"
        )

    @pyqtSlot(int)
    def set_scans_to_average(self, count: int) -> None:
        """Set how many consecutive scans to average (pixel-wise mean).

        Clears the FIFO buffer so spectra from the previous window are not
        mixed with the new setting. ``count == 1`` is passthrough.
        """
        if count < 1:
            self.device_error.emit(
                f"Invalid scans_to_average: {count} (must be >= 1)"
            )
            return

        self._scans_to_average = int(count)
        self._clear_scan_buffer()
        self.status_message.emit(
            f"Scans to average set to {self._scans_to_average}"
        )

    @pyqtSlot()
    def start_acquisition(self) -> None:
        """Start continuous spectrum acquisition on the worker thread."""
        if self._device is None or self._wavelengths is None or self._timer is None:
            self.device_error.emit("Spectrometer not initialised; call initialize() first")
            return

        if self._acquiring:
            self.status_message.emit("Acquisition already running")
            return

        self._clear_scan_buffer()
        self._acquiring = True
        interval_ms = self._poll_interval_ms()
        self._timer.start(interval_ms)
        self.status_message.emit(
            f"Acquisition started (poll interval {interval_ms} ms)"
        )

    @pyqtSlot()
    def stop_acquisition(self) -> None:
        """Halt acquisition and release the spectrometer device."""
        self._acquiring = False
        if self._timer is not None:
            self._timer.stop()

        self._clear_scan_buffer()
        self._release_device()
        self.status_message.emit("Acquisition stopped; device released")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clear_scan_buffer(self) -> None:
        """Drop buffered scans (called when averaging or timing settings change)."""
        self._scan_buffer = deque(maxlen=max(1, self._scans_to_average))

    def _average_intensities(self, intensities: np.ndarray) -> np.ndarray:
        """Append ``intensities`` and return the pixel-wise mean of the buffer."""
        if self._scans_to_average <= 1:
            return intensities

        if self._scan_buffer.maxlen != self._scans_to_average:
            self._clear_scan_buffer()

        self._scan_buffer.append(intensities)
        # Vectorized mean over filled frames; O(n_pixels) via NumPy BLAS.
        stacked = np.stack(self._scan_buffer, axis=0)
        return np.mean(stacked, axis=0, dtype=np.float64)

    def _poll_interval_ms(self) -> int:
        """Match integration time, floored to a minimum polling rate."""
        interval_us = max(self._integration_time_us, _MIN_POLL_INTERVAL_US)
        return max(1, interval_us // 1000)

    @pyqtSlot()
    def _acquire_once(self) -> None:
        if not self._acquiring or self._device is None or self._wavelengths is None:
            return

        try:
            raw = np.asarray(self._device.intensities(), dtype=np.float64)
            averaged = self._average_intensities(raw)
            self.data_acquired.emit(self._wavelengths.copy(), averaged)
        except Exception as exc:
            message = f"Spectrometer read error (possible USB disconnect): {exc}"
            logger.error(message)
            self.device_error.emit(message)
            self._acquiring = False
            if self._timer is not None:
                self._timer.stop()

    def _release_device(self) -> None:
        if self._device is not None:
            try:
                self._device.close()
            except Exception as exc:
                logger.warning("Error while closing spectrometer: %s", exc)
            self._device = None
            self._is_hardware = False
            self._wavelengths = None


# ---------------------------------------------------------------------------
# Standalone running test block
# ---------------------------------------------------------------------------
# if __name__ == "__main__":
#     from PyQt6.QtCore import QCoreApplication, QThread
#     import sys
#
#     app = QCoreApplication(sys.argv)
#     thread = QThread()
#     worker = SpectrometerWorker()
#     worker.moveToThread(thread)
#
#     def start_worker():
#         worker.initialize()
#         worker.start_acquisition()
#
#     thread.started.connect(start_worker)
#     worker.status_message.connect(lambda msg: print(f"[STATUS] {msg}"))
#     worker.device_error.connect(lambda err: print(f"[ERROR] {err}"))
#
#     count = 0
#     def handle_data(wavelengths, intensities):
#         global count
#         count += 1
#         print(f"Scan {count}: {len(wavelengths)} points | Peak: {intensities.max():.1f} counts")
#         if count >= 5:
#             worker.stop_acquisition()
#             thread.quit()
#
#     worker.data_acquired.connect(handle_data)
#     thread.finished.connect(app.quit)
#
#     thread.start()
#     sys.exit(app.exec())
