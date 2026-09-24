#!/usr/bin/env python3
"""Generate a professional user documentation Word report for the spectrometer software."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


OUT_PATH = Path(__file__).resolve().parent / "Multi_Function_Optical_Analyzer_User_Guide.docx"


def _set_run_font(run, *, bold: bool = False, italic: bool = False, size: int = 11) -> None:
    run.bold = bold
    run.italic = italic
    run.font.size = Pt(size)
    run.font.name = "Times New Roman"
    r = run._element
    rPr = r.get_or_add_rPr()
    rFonts = rPr.get_or_add_rFonts()
    rFonts.set(qn("w:ascii"), "Times New Roman")
    rFonts.set(qn("w:hAnsi"), "Times New Roman")
    rFonts.set(qn("w:eastAsia"), "Times New Roman")


def add_heading(doc: Document, text: str, level: int = 1) -> None:
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        _set_run_font(run, bold=True, size=14 if level == 1 else 12)
        run.font.color.rgb = RGBColor(0x12, 0x3E, 0x4F)


def add_para(
    doc: Document,
    text: str,
    *,
    bold: bool = False,
    italic: bool = False,
    first_line_indent: bool = True,
) -> None:
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_after = Pt(8)
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    if first_line_indent:
        pf.first_line_indent = Inches(0.3)
    run = p.add_run(text)
    _set_run_font(run, bold=bold, italic=italic, size=11)


def add_bullet(doc: Document, text: str, level: int = 0) -> None:
    p = doc.add_paragraph(style="List Bullet")
    p.clear()
    p.paragraph_format.left_indent = Inches(0.25 + 0.25 * level)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    _set_run_font(run, size=11)


def add_numbered(doc: Document, text: str) -> None:
    p = doc.add_paragraph(style="List Number")
    p.clear()
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    _set_run_font(run, size=11)


def build_document() -> Document:
    doc = Document()

    # Page margins
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # ----- Title page -----
    for _ in range(2):
        doc.add_paragraph()

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = title.add_run("Multi-Function Optical Analyzer")
    _set_run_font(r, bold=True, size=22)
    r.font.color.rgb = RGBColor(0x12, 0x3E, 0x4F)

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = subtitle.add_run(
        "Software User Documentation and Technical Reference Guide"
    )
    _set_run_font(r, italic=True, size=14)

    gap = doc.add_paragraph()
    gap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = gap.add_run(
        "A Desktop Application for Real-Time Spectrometric Acquisition,\n"
        "Multi-Geometry Optical Measurement, and Spectral Post-Processing"
    )
    _set_run_font(r, size=12)

    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta.paragraph_format.space_before = Pt(36)
    r = meta.add_run(
        f"Document Version 1.0\n"
        f"Prepared: {date.today().strftime('%d %B %Y')}\n"
        f"Software Modules: app.py · hardware.py"
    )
    _set_run_font(r, size=11)

    doc.add_page_break()

    # ----- Abstract -----
    add_heading(doc, "1. Abstract", 1)
    add_para(
        doc,
        "This document provides a comprehensive user and technical guide to the "
        "Multi-Function Optical Analyzer, a PyQt6-based desktop application designed "
        "for real-time acquisition and analysis of optical spectra from Ocean Optics "
        "spectrometers (including FLAME-class instruments) via the python-seabreeze "
        "interface. The software supports hierarchical capture of Baseline, Standard, "
        "and Sample traces across three optical measurement modes—specular reflectance "
        "at near-normal incidence (R₀), reflectance at sixty degrees (R₆₀), and "
        "transmission (T)—and offers integrated post-processing capabilities including "
        "centered moving-average smoothing and uniform-grid spectral interpolation. "
        "Data may be exported as comma-separated values (CSV) or as publication-ready "
        "plot images. The architecture isolates spectrometer I/O on a dedicated worker "
        "thread so that the graphical user interface remains responsive during continuous "
        "acquisition. In the absence of physical hardware, the application falls back "
        "transparently to a synthetic mock spectrometer, enabling training, demonstration, "
        "and offline development without interrupting the user workflow.",
    )

    # ----- Introduction -----
    add_heading(doc, "2. Introduction and Purpose", 1)
    add_para(
        doc,
        "Optical spectroscopy is a foundational technique in materials characterization, "
        "quality control, and academic laboratory instruction. Practical measurement "
        "campaigns frequently require coordinated acquisition of reference (baseline), "
        "calibration (standard), and specimen (sample) spectra under multiple optical "
        "geometries. The Multi-Function Optical Analyzer was developed to consolidate "
        "these tasks into a single, coherent desktop environment that couples live "
        "instrument control with structured data capture and lightweight spectral "
        "conditioning.",
    )
    add_para(
        doc,
        "The principal objectives of the software are as follows:",
        first_line_indent=False,
    )
    add_bullet(
        doc,
        "To provide non-blocking, continuous spectral acquisition from an Ocean Optics "
        "spectrometer connected over USB.",
    )
    add_bullet(
        doc,
        "To organize measurements hierarchically by optical geometry (R₀, R₆₀, T) and "
        "by measurement role (Baseline, Standard, Sample).",
    )
    add_bullet(
        doc,
        "To enable rapid visual comparison of live and captured traces on a shared "
        "wavelength–intensity plot.",
    )
    add_bullet(
        doc,
        "To support post-acquisition conditioning (moving-average smoothing and "
        "wavelength-grid interpolation) without altering the underlying raw captures.",
    )
    add_bullet(
        doc,
        "To export analysis-ready tabular data and figure images suitable for "
        "laboratory notebooks, reports, and further computational analysis.",
    )

    # ----- System overview -----
    add_heading(doc, "3. System Overview", 1)
    add_heading(doc, "3.1 Software Architecture", 2)
    add_para(
        doc,
        "The application comprises two primary modules. The graphical front end "
        "(app.py) implements the main window, measurement workflow controls, plotting "
        "surface, and post-processing dialog. The hardware back end (hardware.py) "
        "implements SpectrometerWorker, a QObject designed to execute on a dedicated "
        "QThread. Communication between the user interface and the worker proceeds "
        "exclusively through Qt signals and slots, ensuring thread-safe control of "
        "integration time, scan averaging, and acquisition start/stop without blocking "
        "the event loop.",
    )
    add_para(
        doc,
        "Upon launch, SpectrometerWorker attempts to open the first available Ocean "
        "Optics device using Spectrometer.from_first_available(). If no device is "
        "detected, or if the connection fails, the worker instantiates MockSpectrometer, "
        "which synthesizes a noisy Gaussian spectrum over approximately 350–1000 nm. "
        "Status messages report whether the session is operating in hardware or mock mode.",
    )

    add_heading(doc, "3.2 Supported Instrumentation", 2)
    add_para(
        doc,
        "The software is intended for Ocean Optics spectrometers accessible through "
        "python-seabreeze (for example, FLAME and related USB spectrometers). Pixel "
        "count and calibrated wavelength axes are read from the device. Intensity "
        "values are treated as detector counts on a 16-bit-compatible scale "
        "(0–65535 counts) for display ranging.",
    )

    add_heading(doc, "3.3 Optical Measurement Modes", 2)
    add_para(
        doc,
        "Three optical modes are provided as exclusive, checkable controls. Switching "
        "modes changes the active data context without discarding captures stored under "
        "other modes.",
        first_line_indent=False,
    )
    add_bullet(
        doc,
        "R₀ — Reflectance (or related intensity measurement) associated with near-normal "
        "geometry; internal storage key R0.",
    )
    add_bullet(
        doc,
        "R₆₀ — Reflectance associated with a sixty-degree measurement geometry; "
        "internal storage key R60.",
    )
    add_bullet(
        doc,
        "T — Transmission measurement geometry; internal storage key T.",
    )
    add_para(
        doc,
        "Physical staging of sources, fibers, and detectors remains the operator’s "
        "responsibility; the software records and labels spectra according to the "
        "selected mode and does not itself actuate optomechanics.",
        first_line_indent=False,
    )

    # ----- Services -----
    add_heading(doc, "4. Services and Capabilities", 1)
    add_heading(doc, "4.1 Real-Time Spectral Acquisition", 2)
    add_para(
        doc,
        "The live plot updates continuously from the worker’s data_acquired signal. "
        "The abscissa is wavelength in nanometres; the ordinate is intensity in counts. "
        "Automatic SI prefixing on axis ticks is disabled so that values remain "
        "numerically transparent (for example, 650 nm rather than a scaled prefix). "
        "The live trace is rendered in bright yellow.",
    )

    add_heading(doc, "4.2 Structured Capture Workflow", 2)
    add_para(
        doc,
        "For each optical mode, the operator may capture three roles: Baseline, "
        "Standard, and Sample. Captured traces persist as dashed overlays "
        "(Baseline cyan, Standard green, Sample magenta) while live acquisition "
        "continues. Capture status is indicated on the optical-data buttons and in "
        "the status bar. The hierarchical in-memory store has the form:",
    )
    add_para(
        doc,
        "spectra_data[mode][role] → intensity array | None, with a shared wavelength "
        "axis derived from the detector calibration.",
        italic=True,
        first_line_indent=False,
    )

    add_heading(doc, "4.3 Acquisition Conditioning", 2)
    add_bullet(
        doc,
        "Integration time (ms): converted to microseconds and applied on the worker "
        "thread; the polling timer tracks the integration interval subject to a "
        "minimum floor for UI stability.",
    )
    add_bullet(
        doc,
        "Scans to Average (1–100): pixel-wise mean over a FIFO buffer of recent scans; "
        "the buffer is cleared when averaging depth or integration time changes, "
        "preventing mixture of incompatible settings. Captures snapshot the currently "
        "averaged live spectrum.",
    )

    add_heading(doc, "4.4 Visualization and Figure Export", 2)
    add_para(
        doc,
        "Save Graph Image exports the PlotWidget contents—axes, grid, legend, live "
        "curve, and captured overlays—via pyqtgraph’s ImageExporter to PNG or JPEG. "
        "Suggested filenames incorporate the active mode and a timestamp "
        "(for example, spectrum_R0_YYYYMMDD_HHMMSS.png).",
    )

    add_heading(doc, "4.5 Tabular Data Export", 2)
    add_para(
        doc,
        "Export Data (CSV) is enabled only when Baseline, Standard, and Sample have "
        "all been captured for the active optical mode. The file contains four columns "
        "with explicit units: Wavelength (nm), <Mode>_Baseline (counts), "
        "<Mode>_Standard (counts), and <Mode>_Sample (counts). Files are written for "
        "spreadsheet compatibility (UTF-8 with BOM and a sep=, directive where "
        "applicable in post-process exports).",
    )

    add_heading(doc, "4.6 Post-Processing Services", 2)
    add_para(
        doc,
        "The Post Process dialog opens a modal data viewer for the active mode. It "
        "presents raw wavelengths and intensities in a high-performance table model "
        "and provides two classes of spectral conditioning:",
        first_line_indent=False,
    )
    add_bullet(
        doc,
        "Centered moving average: window size W (2–50 points); results are inserted "
        "adjacent to the source column as <Trace>_Avg_W{W} (counts). Incomplete "
        "edge neighborhoods are marked N/A. Individual traces may be averaged via a "
        "dropdown menu, or all three via Average All.",
    )
    add_bullet(
        doc,
        "Grid interpolation: user-defined start/end wavelength and step size "
        "(0.1–50 nm) with Linear or Cubic Spline methods. The selected trace is "
        "resampled onto a uniform grid clamped within the raw wavelength range. "
        "Interp_Wavelength (nm) and <Trace>_Interp (counts) are appended beside the "
        "raw columns; table row count remains equal to the raw spectrum length, with "
        "trailing interpolation cells left blank when the uniform grid is shorter.",
    )
    add_para(
        doc,
        "Export Processed Table (CSV) writes every visible column—raw, averaged, and "
        "interpolated—in left-to-right order, preserving empty cells for unequal "
        "column lengths.",
        first_line_indent=False,
    )

    # ----- Requirements -----
    add_heading(doc, "5. System Requirements and Dependencies", 1)
    add_para(doc, "Recommended software environment:", first_line_indent=False)
    add_bullet(doc, "Python 3.10 or newer.")
    add_bullet(doc, "PyQt6 (graphical interface and threading).")
    add_bullet(doc, "NumPy (numerical arrays and convolution/interpolation).")
    add_bullet(doc, "pyqtgraph (real-time plotting and image export).")
    add_bullet(doc, "python-seabreeze (Ocean Optics device access).")
    add_bullet(
        doc,
        "Optional: SciPy for CubicSpline interpolation; a NumPy natural-cubic "
        "fallback is used when SciPy is unavailable.",
    )
    add_para(
        doc,
        "Hardware: a compatible Ocean Optics USB spectrometer with appropriate "
        "drivers and permissions. Without hardware, mock mode remains fully "
        "functional for interface familiarization.",
        first_line_indent=False,
    )

    # ----- Launch -----
    add_heading(doc, "6. Installation and Launch", 1)
    add_para(
        doc,
        "From the project directory, activate the project virtual environment and "
        "start the application:",
        first_line_indent=False,
    )
    add_para(
        doc,
        "source ~/flame_env/bin/activate\n"
        "cd /path/to/spectrometer_proj\n"
        "python app.py",
        italic=True,
        first_line_indent=False,
    )
    add_para(
        doc,
        "On successful start, the status bar reports device readiness (hardware or "
        "mock), and continuous acquisition begins automatically. Closing the main "
        "window stops acquisition, releases the device, and joins the worker thread.",
    )

    # ----- UI guide -----
    add_heading(doc, "7. User Interface Guide", 1)
    add_heading(doc, "7.1 Main Window Layout", 2)
    add_bullet(
        doc,
        "Measurement Mode row: exclusive buttons R₀, R₆₀, and T; active mode highlighted "
        "with the application accent color.",
    )
    add_bullet(
        doc,
        "Optical Data row: Baseline, Standard, and Sample selectors; Capture / Recapture "
        "action for the selected role.",
    )
    add_bullet(
        doc,
        "Plot area: live spectrum with optional captured overlays and legend entries "
        "qualified by optical mode.",
    )
    add_bullet(
        doc,
        "Acquisition controls: Integration time and Scans to Average.",
    )
    add_bullet(
        doc,
        "Session controls: Reset Current Mode, Reset All, Save Graph Image, Export Data "
        "(CSV), and Post Process.",
    )
    add_bullet(
        doc,
        "Status bar: device name, active mode, capture completeness, and transient "
        "worker messages.",
    )

    add_heading(doc, "7.2 Plot Conventions", 2)
    add_para(
        doc,
        "Wavelength (nm) appears on the horizontal axis and Intensity (counts) on the "
        "vertical axis. Captured overlays remain visible when switching among "
        "Baseline/Standard/Sample roles within a mode; changing optical mode swaps the "
        "overlay set to that mode’s stored captures while live acquisition continues "
        "uninterrupted.",
    )

    # ----- Workflow -----
    add_heading(doc, "8. Recommended Measurement Procedure", 1)
    add_para(
        doc,
        "The following procedure describes a complete three-role capture under one "
        "optical geometry, suitable for subsequent reflectance or transmittance "
        "reduction in external analysis tools.",
        first_line_indent=False,
    )
    add_numbered(
        doc,
        "Connect the spectrometer and launch the application; confirm hardware mode "
        "in the status bar (or proceed in mock mode for practice).",
    )
    add_numbered(
        doc,
        "Select the desired optical mode (R₀, R₆₀, or T) and configure the physical "
        "optical path accordingly.",
    )
    add_numbered(
        doc,
        "Set Integration time and Scans to Average to obtain adequate signal-to-noise "
        "without saturation (counts approaching 65535).",
    )
    add_numbered(
        doc,
        "Select Baseline, arrange the baseline condition (for example, dark or "
        "reference path as defined by the laboratory protocol), and press Capture "
        "Baseline.",
    )
    add_numbered(
        doc,
        "Select Standard, present the calibration standard, and capture Standard.",
    )
    add_numbered(
        doc,
        "Select Sample, present the specimen, and capture Sample.",
    )
    add_numbered(
        doc,
        "Review overlays on the plot. Optionally save a graph image and/or export "
        "the four-column CSV for the active mode.",
    )
    add_numbered(
        doc,
        "Repeat for remaining optical modes as required. Switching modes preserves "
        "prior captures; Reset Current Mode clears only the active mode; Reset All "
        "clears every mode after confirmation.",
    )
    add_numbered(
        doc,
        "Open Post Process for smoothing or interpolation before exporting an "
        "extended processed table, if needed.",
    )

    # ----- Post-process detail -----
    add_heading(doc, "9. Post-Processing Dialog — Detailed Operation", 1)
    add_heading(doc, "9.1 Data Viewer", 2)
    add_para(
        doc,
        "Post Process is enabled when at least one capture exists for the active "
        "optical mode. The table lists Wavelength (nm), Baseline (counts), Standard "
        "(counts), and Sample (counts). Empty roles appear as blank cells. Row "
        "population uses a lightweight QAbstractTableModel for efficient display of "
        "large pixel arrays (typically 2048+ detector elements).",
    )

    add_heading(doc, "9.2 Moving Average", 2)
    add_para(
        doc,
        "Choose Window Size (Points) and either Average Trace (Baseline, Standard, or "
        "Sample) or Average All. The algorithm is a centered rolling mean: for odd "
        "window W, offset = (W − 1) // 2; valid convolutions from NumPy are placed "
        "starting at that offset, with incomplete edge rows shown as N/A. Each "
        "application inserts a new column immediately to the right of the source "
        "measurement; repeated applications with different windows stack additional "
        "columns without deleting earlier results.",
    )

    add_heading(doc, "9.3 Grid Interpolation", 2)
    add_para(
        doc,
        "In the Grid Interpolation group, specify Start Wavelength, End Wavelength, "
        "Step Size, Trace, and Method (Linear or Cubic Spline). Interpolate Grid "
        "appends Interp_Wavelength (nm) and <Selected_Trace>_Interp (counts) to the "
        "far right of the table. Raw and averaged columns are never overwritten. "
        "Re-running interpolation updates existing interpolation columns for the "
        "shared wavelength grid and the selected intensity series.",
    )

    add_heading(doc, "9.4 Processed Table Export", 2)
    add_para(
        doc,
        "Export Processed Table (CSV) serializes the complete table using Python’s "
        "csv module with comma delimiters, UTF-8 encoding with BOM, and an Excel "
        "compatible sep=, header line so that spreadsheet applications parse columns "
        "reliably across locales.",
    )

    # ----- Data model -----
    add_heading(doc, "10. Data Model and File Formats", 1)
    add_heading(doc, "10.1 In-Memory Hierarchy", 2)
    add_para(
        doc,
        "Captured intensities are stored under optical-mode keys R0, R60, and T, each "
        "containing Baseline, Standard, and Sample slots. Wavelengths are shared "
        "across modes because they originate from the spectrometer’s calibrated "
        "pixel-to-wavelength map.",
    )

    add_heading(doc, "10.2 Mode-Specific Spectrum CSV", 2)
    add_para(
        doc,
        "Header scheme example for R60:",
        first_line_indent=False,
    )
    add_para(
        doc,
        "Wavelength (nm), R60_Baseline (counts), R60_Standard (counts), "
        "R60_Sample (counts)",
        italic=True,
        first_line_indent=False,
    )

    add_heading(doc, "10.3 Processed Table CSV", 2)
    add_para(
        doc,
        "May include, in visual order, raw columns, zero or more "
        "<Trace>_Avg_W{W} (counts) columns adjacent to their sources, "
        "Interp_Wavelength (nm), and one or more <Trace>_Interp (counts) columns. "
        "Numeric fields use fixed six-decimal-place formatting where applicable.",
    )

    # ----- Safety / concurrency -----
    add_heading(doc, "11. Concurrency, Safety, and Reliability", 1)
    add_para(
        doc,
        "All spectrometer reads occur on the worker thread under a precise QTimer. "
        "GUI controls invoke worker slots via queued connections. On window close, "
        "stop_acquisition is invoked with a blocking queued connection, after which "
        "the thread is quit and awaited. USB disconnects or read failures emit "
        "device_error, stop the acquisition timer, and surface a warning dialog "
        "without terminating the application process abruptly.",
    )

    # ----- Troubleshooting -----
    add_heading(doc, "12. Troubleshooting", 1)
    add_bullet(
        doc,
        "Status indicates mock mode: verify USB connection, seabreeze backend "
        "installation, and user permissions for the device node.",
    )
    add_bullet(
        doc,
        "Export Data disabled: capture Baseline, Standard, and Sample for the "
        "currently selected optical mode.",
    )
    add_bullet(
        doc,
        "Saturated spectrum: reduce integration time or optical throughput.",
    )
    add_bullet(
        doc,
        "Noisy spectrum: increase Scans to Average and/or integration time within "
        "saturation limits.",
    )
    add_bullet(
        doc,
        "CSV opens as a single column in a spreadsheet: use files produced by the "
        "current Export Processed Table implementation (sep=, and UTF-8 BOM), or "
        "import with comma as delimiter.",
    )
    add_bullet(
        doc,
        "Interpolation empty: ensure Start ≤ End, step > 0, and the selected "
        "interval intersects the raw wavelength range.",
    )

    # ----- Limitations -----
    add_heading(doc, "13. Scope and Limitations", 1)
    add_para(
        doc,
        "The software records and conditions spectral intensity arrays; it does not "
        "compute derived optical quantities such as absolute reflectance or "
        "absorbance. Such reductions should be performed with laboratory-specific "
        "formulae using the exported Baseline, Standard, and Sample columns. Optical "
        "mode labels encode experimental intent; the correctness of the physical "
        "geometry remains under operator control. MockSpectrometer output is "
        "synthetic and must not be treated as metrological data.",
    )

    # ----- Glossary -----
    add_heading(doc, "14. Glossary", 1)
    add_bullet(
        doc,
        "Baseline — Reference spectrum captured under the laboratory’s defined "
        "baseline condition for the active optical mode.",
    )
    add_bullet(
        doc,
        "Standard — Spectrum of a calibration or reference standard.",
    )
    add_bullet(
        doc,
        "Sample — Spectrum of the specimen under test.",
    )
    add_bullet(
        doc,
        "Integration time — Detector exposure duration per scan, set in milliseconds "
        "in the UI and applied in microseconds on the device.",
    )
    add_bullet(
        doc,
        "Scans to Average — Number of consecutive intensity vectors averaged "
        "pixel-wise before display and capture.",
    )
    add_bullet(
        doc,
        "Centered moving average — Rolling mean aligned to the window midpoint, "
        "with incomplete ends marked unavailable.",
    )
    add_bullet(
        doc,
        "Uniform grid interpolation — Resampling of a selected intensity series onto "
        "an evenly spaced wavelength axis within the measured spectral range.",
    )

    # ----- Closing -----
    add_heading(doc, "15. Closing Remarks", 1)
    add_para(
        doc,
        "The Multi-Function Optical Analyzer integrates instrument control, "
        "structured multi-geometry capture, and practical spectral post-processing "
        "into a single desktop workflow. By maintaining a clear separation between "
        "live acquisition, role-based captures, and optional conditioning columns, "
        "the software supports reproducible laboratory practice while remaining "
        "accessible for instruction and demonstration. Operators are encouraged to "
        "document optical configurations alongside exported files so that subsequent "
        "quantitative analysis retains full experimental context.",
    )

    footer = doc.add_paragraph()
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.paragraph_format.space_before = Pt(24)
    r = footer.add_run(
        "— End of Document —\n"
        "Multi-Function Optical Analyzer · User Documentation and Technical Reference"
    )
    _set_run_font(r, italic=True, size=10)

    return doc


def main() -> None:
    doc = build_document()
    doc.save(OUT_PATH)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
