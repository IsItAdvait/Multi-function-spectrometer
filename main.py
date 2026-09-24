import sys
import numpy as np
from PyQt6 import QtWidgets, QtCore
import pyqtgraph as pg
import seabreeze.spectrometers as sb

class SpectrometerApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FLAME Real-Time Spectrum")
        self.resize(800, 600)

        self.spec = sb.Spectrometer.from_first_available()
        self.spec.integration_time_micros(100000)  # 100 ms

        self.wavelengths = self.spec.wavelengths()

        self.plot_widget = pg.PlotWidget()
        self.setCentralWidget(self.plot_widget)
        self.plot_widget.setLabel("left", "Intensity (counts)")
        self.plot_widget.setLabel("bottom", "Wavelength (nm)")
        self.plot_widget.setYRange(0, 65535)
        self.curve = self.plot_widget.plot(pen=pg.mkPen(color="y", width=2))

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(50)

    def update_plot(self):
        intensities = self.spec.intensities()
        self.curve.setData(self.wavelengths, intensities)

    def closeEvent(self, event):
        self.spec.close()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = SpectrometerApp()
    window.show()
    sys.exit(app.exec())