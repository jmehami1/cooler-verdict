#!/usr/bin/env python3
"""PyQt5 GUI runner for CPU/GPU cooling comparison tests."""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psutil
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from temp_compare import (
    collect_temperatures,
    find_busy_user_processes,
    infer_gpu_vendor,
    start_stressors,
    stop_stressors,
    summarize_cooler_effect_from_csv,
    utc_now_iso,
    wait_until_temperatures_steady,
)


@dataclass
class GuiPhase:
    name: str
    cooled: bool
    stress: bool
    pre_message: str


class CollapsibleSection(QWidget):
    def __init__(self, title: str, expanded: bool = True, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.toggle_button = QToolButton(text=title, checkable=True, checked=expanded)
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.toggle_button.setStyleSheet("font-weight: 600; padding: 6px 0;")

        self.content = QWidget()
        self.content.setVisible(expanded)
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(8)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.toggle_button)
        layout.addWidget(self.content)

        self.toggle_button.toggled.connect(self._set_expanded)

    def _set_expanded(self, expanded: bool) -> None:
        self.toggle_button.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.content.setVisible(expanded)


class TempCompareWindow(QMainWindow):
    status_changed = pyqtSignal(str)
    phase_changed = pyqtSignal(str)
    remaining_changed = pyqtSignal(int)
    warning_changed = pyqtSignal(str)
    output_changed = pyqtSignal(str)
    buttons_changed = pyqtSignal(bool, bool)
    sensor_sample_ready = pyqtSignal(object, float)
    dialog_info_requested = pyqtSignal(str, str)
    dialog_error_requested = pyqtSignal(str, str)
    cooler_confirm_requested = pyqtSignal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Cooling Compare - CPU/GPU Temperature Logger")
        self.resize(1200, 820)

        self.running = False
        self.stop_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None
        self.csv_writer: Optional[csv.writer] = None
        self.phase_summaries: List[dict] = []
        self.run_started_epoch: Optional[float] = None
        self.plot_origin_epoch = time.time()
        self._poll_active = False
        self._poll_in_flight = False
        self._pending_cooler_confirmation: Optional[Tuple[threading.Event, Dict[str, bool]]] = None

        self.sensor_data: Dict[str, List[float]] = {}
        self.sensor_time_data: Dict[str, List[float]] = {}
        self.plot_lines: Dict[str, object] = {}
        self.plot_visibility: Dict[str, bool] = {}
        self.legend_artist_to_sensor: Dict[object, str] = {}
        self.sensor_display_names: Dict[str, str] = {}
        self.sensor_live_labels: Dict[str, QLabel] = {}

        self.figure = Figure(figsize=(9.5, 4.8), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Live Temperature Sensors")
        self.ax.set_xlabel("Elapsed time (min)")
        self.ax.set_ylabel("Temperature (C)")
        self.ax.grid(True, alpha=0.35)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect("pick_event", self._on_plot_pick)

        self.live_poll_timer = QTimer(self)
        self.live_poll_timer.setInterval(3000)
        self.live_poll_timer.timeout.connect(self._poll_live_temps)

        self._build_layout()
        self._connect_signals()
        self._detect_and_set_specs()
        self._set_status("Ready")
        self._set_phase("No test running")
        self._set_remaining_seconds(0)
        self._set_warning("")
        self.output_changed.emit("results/")
        self.buttons_changed.emit(True, False)
        self._start_live_polling()

    def _connect_signals(self) -> None:
        self.status_changed.connect(self.status_value_label.setText)
        self.phase_changed.connect(self.phase_value_label.setText)
        self.remaining_changed.connect(self._apply_remaining)
        self.warning_changed.connect(self.warning_value_label.setText)
        self.output_changed.connect(self.output_value_label.setText)
        self.buttons_changed.connect(self._apply_buttons)
        self.sensor_sample_ready.connect(self._consume_sensor_sample)
        self.dialog_info_requested.connect(self._show_info_dialog)
        self.dialog_error_requested.connect(self._show_error_dialog)
        self.cooler_confirm_requested.connect(self._show_cooler_confirmation)

    def _build_layout(self) -> None:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.setCentralWidget(scroll)

        container = QWidget()
        scroll.setWidget(container)
        outer = QVBoxLayout(container)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(12)

        hardware_section = CollapsibleSection("Detected Hardware", expanded=True)
        hardware_form = QFormLayout()
        hardware_form.setLabelAlignment(Qt.AlignLeft)
        self.cpu_spec_label = QLabel("Detecting CPU...")
        self.gpu_spec_label = QLabel("Detecting GPU...")
        self.storage_spec_label = QLabel("Detecting storage...")
        for label in (self.cpu_spec_label, self.gpu_spec_label, self.storage_spec_label):
            label.setWordWrap(True)
        hardware_form.addRow("CPU:", self.cpu_spec_label)
        hardware_form.addRow("GPU:", self.gpu_spec_label)
        hardware_form.addRow("Storage:", self.storage_spec_label)
        hardware_section.content_layout.addLayout(hardware_form)
        outer.addWidget(hardware_section)

        sensors_section = CollapsibleSection("Detected Sensors (Live)", expanded=True)
        self.sensor_columns_widget = QWidget()
        self.sensor_columns_layout = QGridLayout(self.sensor_columns_widget)
        self.sensor_columns_layout.setContentsMargins(0, 0, 0, 0)
        self.sensor_columns_layout.setSpacing(6)
        sensors_section.content_layout.addWidget(self.sensor_columns_widget)
        outer.addWidget(sensors_section)

        plot_section = CollapsibleSection("Temperature Plot", expanded=True)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        plot_section.content_layout.addWidget(self.canvas)
        # Legend area: a horizontal scroll area containing checkboxes/buttons
        # so the legend always lives below the axes and cannot overlap ticks.
        self.legend_scroll = QScrollArea()
        self.legend_scroll.setWidgetResizable(True)
        self.legend_container = QWidget()
        self.legend_layout = QHBoxLayout(self.legend_container)
        self.legend_layout.setContentsMargins(6, 6, 6, 6)
        self.legend_layout.setSpacing(8)
        self.legend_scroll.setWidget(self.legend_container)
        self.legend_scroll.setFixedHeight(64)
        plot_section.content_layout.addWidget(self.legend_scroll)
        plot_section.content_layout.setStretch(0, 1)
        plot_section.setMinimumHeight(420)
        outer.addWidget(plot_section, stretch=1)

        controls_section = CollapsibleSection("Test Controls", expanded=True)
        controls_layout = QGridLayout()
        controls_layout.setHorizontalSpacing(12)
        controls_layout.setVerticalSpacing(10)

        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(1, 1440)
        self.duration_spin.setValue(20)

        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setRange(1.0, 3600.0)
        self.interval_spin.setDecimals(1)
        self.interval_spin.setSingleStep(0.5)
        self.interval_spin.setValue(1.0)

        self.cpu_workers_spin = QSpinBox()
        self.cpu_workers_spin.setRange(1, max(1, os.cpu_count() or 1))
        self.cpu_workers_spin.setValue(max(1, (os.cpu_count() or 2) - 1))

        self.cpu_threshold_spin = QDoubleSpinBox()
        self.cpu_threshold_spin.setRange(0.1, 100.0)
        self.cpu_threshold_spin.setDecimals(1)
        self.cpu_threshold_spin.setSingleStep(0.5)
        self.cpu_threshold_spin.setValue(3.0)

        self.gpu_stress_cmd_edit = QLineEdit()
        self.enforce_clean_checkbox = QCheckBox("Enforce clean process list")

        controls_layout.addWidget(QLabel("Duration per phase (min)"), 0, 0)
        controls_layout.addWidget(self.duration_spin, 0, 1)
        controls_layout.addWidget(QLabel("Sample interval (sec)"), 0, 2)
        controls_layout.addWidget(self.interval_spin, 0, 3)
        controls_layout.addWidget(QLabel("CPU workers"), 0, 4)
        controls_layout.addWidget(self.cpu_workers_spin, 0, 5)
        controls_layout.addWidget(QLabel("CPU threshold (%)"), 0, 6)
        controls_layout.addWidget(self.cpu_threshold_spin, 0, 7)

        controls_layout.addWidget(QLabel("GPU stress cmd (optional)"), 1, 0)
        controls_layout.addWidget(self.gpu_stress_cmd_edit, 1, 1, 1, 5)
        controls_layout.addWidget(self.enforce_clean_checkbox, 1, 6, 1, 2)

        buttons_row = QHBoxLayout()
        self.start_button = QPushButton("Start Testing")
        self.stop_button = QPushButton("Stop Testing")
        self.check_button = QPushButton("Check Running Processes")
        self.stop_button.setEnabled(False)
        buttons_row.addWidget(self.start_button)
        buttons_row.addWidget(self.stop_button)
        buttons_row.addWidget(self.check_button)
        buttons_row.addStretch(1)
        controls_layout.addLayout(buttons_row, 2, 0, 1, 8)
        controls_section.content_layout.addLayout(controls_layout)
        outer.addWidget(controls_section)

        status_section = CollapsibleSection("Current Status", expanded=True)
        status_form = QFormLayout()
        status_form.setLabelAlignment(Qt.AlignLeft)
        self.status_value_label = QLabel("Ready")
        self.phase_value_label = QLabel("No test running")
        self.remaining_value_label = QLabel("00:00")
        self.output_value_label = QLabel("results/")
        self.warning_value_label = QLabel("")
        self.warning_value_label.setStyleSheet("color: #a94442;")
        for label in (
            self.status_value_label,
            self.phase_value_label,
            self.remaining_value_label,
            self.output_value_label,
            self.warning_value_label,
        ):
            label.setWordWrap(True)
        status_form.addRow("State:", self.status_value_label)
        status_form.addRow("Phase:", self.phase_value_label)
        status_form.addRow("Remaining:", self.remaining_value_label)
        status_form.addRow("Output:", self.output_value_label)
        status_form.addRow("Warnings:", self.warning_value_label)
        status_section.content_layout.addLayout(status_form)
        outer.addWidget(status_section)

        self.start_button.clicked.connect(self.start_testing)
        self.stop_button.clicked.connect(self.stop_testing)
        self.check_button.clicked.connect(self.check_running_processes)

    def _start_live_polling(self) -> None:
        self._poll_active = True
        self._poll_live_temps()
        self.live_poll_timer.start()

    def _detect_and_set_specs(self) -> None:
        try:
            cpu_name = None
            try:
                out = subprocess.check_output(["lscpu"], text=True, stderr=subprocess.DEVNULL)
                for line in out.splitlines():
                    if "Model name:" in line:
                        cpu_name = line.split(":", 1)[1].strip()
                        break
            except Exception:
                pass
            if not cpu_name:
                cpu_name = os.uname().machine
            cpu_cores = psutil.cpu_count(logical=False)
            cpu_threads = psutil.cpu_count(logical=True)
            self.cpu_spec_label.setText(f"{cpu_name} ({cpu_cores} cores, {cpu_threads} threads)")
        except Exception as exc:
            self.cpu_spec_label.setText(f"CPU: Unknown ({exc})")

        gpu_str = "No discrete GPU detected"
        try:
            if subprocess.call(["which", "nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
                try:
                    out = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                        text=True,
                        stderr=subprocess.DEVNULL,
                    )
                    lines = [line.strip() for line in out.splitlines() if line.strip()]
                    if lines:
                        gpu_str = ", ".join(line.split(",", 1)[0].strip() for line in lines)
                except Exception:
                    gpu_str = "NVIDIA GPU present (details unavailable)"
            else:
                try:
                    out = subprocess.check_output(["lspci"], text=True, stderr=subprocess.DEVNULL)
                    gpus = [line for line in out.splitlines() if "VGA compatible controller" in line or "3D controller" in line]
                    if gpus:
                        gpu_str = "; ".join(gpus)
                except Exception:
                    pass
        except Exception as exc:
            gpu_str = f"GPU: Unknown ({exc})"
        self.gpu_spec_label.setText(gpu_str)

        storage_str = "No storage devices detected"
        try:
            drives = []
            if subprocess.call(["which", "lsblk"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
                try:
                    raw = subprocess.check_output(
                        ["lsblk", "-d", "-J", "-o", "NAME,MODEL,SIZE,TYPE"],
                        text=True,
                        stderr=subprocess.DEVNULL,
                    )
                    for dev in json.loads(raw).get("blockdevices", []):
                        if dev.get("type") != "disk":
                            continue
                        model = (dev.get("model") or dev.get("name", "?")).strip()
                        size = dev.get("size") or ""
                        drives.append(f"{model} ({size})" if size else model)
                except Exception:
                    raw = subprocess.check_output(
                        ["lsblk", "-d", "-n", "-o", "NAME,MODEL,SIZE,TYPE"],
                        text=True,
                        stderr=subprocess.DEVNULL,
                    )
                    for line in raw.splitlines():
                        parts = line.split()
                        if len(parts) < 2 or parts[-1] != "disk":
                            continue
                        size = parts[-2] if len(parts) >= 3 else ""
                        model = " ".join(parts[1:-2]).strip() if len(parts) > 3 else parts[0]
                        drives.append(f"{model or parts[0]} ({size})" if size else model or parts[0])
            if not drives:
                for dev in sorted(Path("/sys/block").iterdir()):
                    if any(dev.name.startswith(prefix) for prefix in ("loop", "ram", "sr", "dm-", "md")):
                        continue
                    model_file = dev / "device" / "model"
                    size_file = dev / "size"
                    if model_file.exists():
                        model = model_file.read_text().strip()
                        size_gb = int(size_file.read_text().strip()) * 512 / 1e9 if size_file.exists() else 0
                        drives.append(f"{model} ({size_gb:.0f} GB)" if size_gb > 1 else model)
            storage_str = ", ".join(drives) if drives else "No storage devices detected"
        except Exception as exc:
            storage_str = f"Unknown ({exc})"
        self.storage_spec_label.setText(storage_str)

    def _set_status(self, text: str) -> None:
        self.status_changed.emit(text)

    def _set_phase(self, text: str) -> None:
        self.phase_changed.emit(text)

    def _set_remaining_seconds(self, seconds: int) -> None:
        self.remaining_changed.emit(seconds)

    def _set_warning(self, text: str) -> None:
        self.warning_changed.emit(text)

    def _apply_remaining(self, seconds: int) -> None:
        minutes = max(0, seconds) // 60
        remainder = max(0, seconds) % 60
        self.remaining_value_label.setText(f"{minutes:02d}:{remainder:02d}")

    def _apply_buttons(self, start_enabled: bool, stop_enabled: bool) -> None:
        self.start_button.setEnabled(start_enabled)
        self.stop_button.setEnabled(stop_enabled)

    def _show_info_dialog(self, title: str, message: str) -> None:
        QMessageBox.information(self, title, message)

    def _show_error_dialog(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)

    def _poll_live_temps(self) -> None:
        # Keep polling active even while a test run is in progress so the
        # GUI and logging receive continuous samples. Previously we skipped
        # polling when `self.running` was True which caused gaps between
        # phases when the worker wasn't emitting samples.
        if not self._poll_active or self._poll_in_flight:
            return

        self._poll_in_flight = True

        def _read() -> None:
            try:
                readings = self._deduplicate_sensors(collect_temperatures())
            except Exception:
                readings = {}
            elapsed = time.time() - self.plot_origin_epoch
            self.sensor_sample_ready.emit(readings, elapsed)
            self._poll_in_flight = False

        threading.Thread(target=_read, daemon=True).start()

    def _consume_sensor_sample(self, readings: object, elapsed_s: float) -> None:
        if not isinstance(readings, dict):
            return
        self._append_sensor_sample(elapsed_s, readings)
        self._update_live_temps(readings)
        self._refresh_plot()

    def _append_sensor_sample(self, elapsed_s: float, readings: Dict[str, float]) -> None:
        new_sensors = False
        for sensor, value in readings.items():
            if not sensor.endswith(":temp_c"):
                continue
            self.sensor_time_data.setdefault(sensor, []).append(elapsed_s)
            self.sensor_data.setdefault(sensor, []).append(value)
            if sensor not in self.plot_visibility:
                self.plot_visibility[sensor] = True
                new_sensors = True
        if new_sensors or not self.sensor_display_names:
            self._refresh_sensor_metadata()

    def _compute_legend_columns(self, labels: List[str]) -> int:
        if not labels:
            return 1

        plot_width_px = max(1.0, float(self.ax.get_window_extent().width))
        longest_label = max(len(label) for label in labels)
        # Estimate text + line handle + spacing width for each legend column.
        estimated_col_width_px = max(90.0, 58.0 + (longest_label * 6.2))
        max_cols = int(plot_width_px // estimated_col_width_px)
        return max(1, min(len(labels), max_cols))

    def _refresh_plot(self) -> None:
        self.ax.clear()
        self.ax.set_title("Live Temperature Sensors")
        self.ax.set_xlabel("Elapsed time (min)")
        self.ax.set_ylabel("Temperature (C)")
        self.ax.grid(True, alpha=0.35)
        self.plot_lines = {}
        self.legend_artist_to_sensor = {}
        legend_bottom_margin = 0.08

        plotted: List[str] = []
        plotted_labels: List[str] = []
        for sensor in sorted(self.sensor_time_data):
            xs = [value / 60.0 for value in self.sensor_time_data.get(sensor, [])]
            ys = self.sensor_data.get(sensor, [])
            if not xs or not ys:
                continue
            label = self.sensor_display_names.get(sensor, self._format_sensor_label(sensor))
            (line,) = self.ax.plot(xs, ys, label=label, linewidth=1.4)
            visible = self.plot_visibility.get(sensor, True)
            line.set_visible(visible)
            self.plot_lines[sensor] = line
            plotted.append(sensor)
            plotted_labels.append(label)

        if plotted:
            # Remove any existing figure-level legends to avoid duplicates.
            try:
                for old_leg in list(self.figure.legends):
                    old_leg.remove()
            except Exception:
                pass

            legend_cols = self._compute_legend_columns(plotted_labels)
            # Create a figure-level legend anchored below the axes so it does not
            # overlap tick labels; we'll measure it and expand the bottom margin.
            handles = [self.plot_lines[s] for s in plotted]
            labels = plotted_labels
            # Anchor the legend so its top sits at the very bottom of the
            # figure (y=0.0). We'll then measure its height and push the axes
            # up by that amount so the legend sits entirely below the axes.
            legend = self.figure.legend(
                handles=handles,
                labels=labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.0),
                bbox_transform=self.figure.transFigure,
                mode="expand",
                ncol=legend_cols,
                fontsize=8,
                frameon=True,
                columnspacing=1.0,
                handlelength=1.6,
            )

            # Rebuild the external Qt legend widget below the canvas so it
            # never overlaps the axes. Use checkboxes to toggle visibility.
            # Clear existing legend widgets.
            while self.legend_layout.count():
                item = self.legend_layout.takeAt(0)
                w = item.widget()
                if w is not None:
                    w.deleteLater()

            for sensor, label in zip(plotted, plotted_labels):
                cb = QCheckBox(label)
                cb.setChecked(self.plot_visibility.get(sensor, True))
                cb.setToolTip(sensor)

                def _make_toggled(s):
                    def _toggled(state: int) -> None:
                        visible = bool(state)
                        self.plot_visibility[s] = visible
                        line = self.plot_lines.get(s)
                        if line is not None:
                            line.set_visible(visible)
                        # Redraw only — avoid rebuilding everything.
                        self.canvas.draw_idle()

                    return _toggled

                cb.stateChanged.connect(_make_toggled(sensor))
                self.legend_layout.addWidget(cb)
            self.legend_layout.addStretch(1)

        # Adjust bottom margin to account for the legend height so it doesn't
        # overlap x-axis tick labels or the x-axis title. We need to draw the
        # canvas once so the legend has a renderer and a measurable bbox.
        if plotted and legend is not None:
            # Force a synchronous draw to get an accurate renderer and bboxes.
            self.canvas.draw()
            try:
                renderer = self.figure.canvas.get_renderer()
                legend_bbox = legend.get_window_extent(renderer=renderer)
                fig_bbox = self.figure.get_window_extent(renderer=renderer)

                # Legend height as fraction of figure height.
                legend_height_frac = legend_bbox.height / max(1.0, fig_bbox.height)
                # Small extra fraction to account for tick labels and xlabel.
                extra_frac = 0.06
                new_bottom = min(0.85, legend_height_frac + extra_frac)
                # Ensure there's at least a tiny bottom margin.
                new_bottom = max(new_bottom, 0.02 + legend_height_frac)
                self.figure.subplots_adjust(bottom=new_bottom)
            except Exception:
                # Fallback to tight_layout if measurement fails.
                self.figure.tight_layout(rect=(0, legend_bottom_margin, 1, 1))
            # Final draw to reflect adjusted margins.
            self.canvas.draw_idle()
        else:
            self.figure.tight_layout(rect=(0, legend_bottom_margin, 1, 1))
            self.canvas.draw_idle()

    def _on_plot_pick(self, event) -> None:
        sensor = self.legend_artist_to_sensor.get(event.artist)
        if sensor is None:
            return
        self.plot_visibility[sensor] = not self.plot_visibility.get(sensor, True)
        self._refresh_plot()

    def _categorize_sensor(self, key: str) -> str:
        lowered = key.lower()
        if any(token in lowered for token in ("coretemp", "k10temp", "zenpower", "cpu", "core", "package", "tctl", "tdie", "x86_pkg_temp", "ccd")):
            return "CPU"
        if "gpu" in lowered:
            return "GPU"
        if "nvme" in lowered or "m.2" in lowered:
            return "NVMe / M.2"
        if "ssd" in lowered:
            return "SSD"
        if "sata" in lowered:
            return "SATA"
        if "hdd" in lowered or "disk" in lowered:
            return "HDD"
        if "pch" in lowered or "chipset" in lowered:
            return "Chipset"
        if "acpitz" in lowered or "acpi" in lowered:
            return "System (ACPI)"
        return "Other"

    def _deduplicate_sensors(self, readings: Dict[str, float]) -> Dict[str, float]:
        has_coretemp = any(key.startswith("coretemp:") for key in readings)
        has_k10temp = any(key.startswith(("k10temp:", "zenpower:")) for key in readings)
        result: Dict[str, float] = {}
        for key, value in readings.items():
            if key.startswith("sensors:"):
                label = key.split(":")[1].lower()
                if has_coretemp and (label.startswith("core") or label.startswith("package")):
                    continue
                if has_k10temp and any(label.startswith(prefix) for prefix in ("tctl", "tdie", "ccd", "tccd")):
                    continue
            result[key] = value
        return result

    def _short_sensor_name(self, key: str) -> str:
        parts = key.split(":")
        if len(parts) < 3:
            return key[:24]
        source, label = parts[0], parts[1]
        if source in ("coretemp", "k10temp", "zenpower", "nvme", "linux_thermal") or source.startswith("gpu"):
            return label[:24]
        return f"{source}: {label}"[:24]

    def _compact_gpu_name(self, label: str) -> str:
        compact = label.replace("NVIDIA ", "").replace("GeForce ", "")
        compact = compact.replace("Laptop GPU", "Laptop").replace("Graphics", "")
        compact = " ".join(compact.split())
        if len(compact) > 22:
            tokens = compact.split()
            keep = [token for token in tokens if any(char.isdigit() for char in token)]
            compact = " ".join(keep[-2:] or tokens[-3:])
        return compact[:22] or label[:22]

    def _format_sensor_label(self, key: str) -> str:
        parts = key.split(":")
        if len(parts) < 3:
            return key[:24]

        source, label = parts[0], parts[1]
        lowered = label.lower()
        if source.startswith("gpu"):
            gpu_idx = source[3:] if source[3:].isdigit() else ""
            prefix = f"GPU {gpu_idx}" if gpu_idx else "GPU"
            return f"{prefix} {self._compact_gpu_name(label)}"[:24].strip()
        if source in ("coretemp", "k10temp", "zenpower"):
            if lowered.startswith("package"):
                return "CPU Package"
            if lowered.startswith(("tdie", "tctl")):
                return label.upper()
            return label[:24]
        if source == "linux_thermal":
            aliases = {
                "x86_pkg_temp": "CPU Package",
                "acpitz": "ACPI",
            }
            return aliases.get(lowered, label.replace("_", " ")[:24])
        if source == "nvme":
            return f"NVMe {label}"[:24]
        if source == "sensors":
            return label.replace("_", " ")[:24]
        return self._short_sensor_name(key)

    def _refresh_sensor_metadata(self) -> None:
        deduped = self._deduplicate_sensors({key: 0.0 for key in self.sensor_data})
        sensor_keys = sorted(key for key in deduped if key.endswith(":temp_c"))
        used_names: Dict[str, int] = {}
        self.sensor_display_names = {}
        for key in sensor_keys:
            base_name = self._format_sensor_label(key)
            count = used_names.get(base_name, 0)
            used_names[base_name] = count + 1
            display_name = base_name if count == 0 else f"{base_name} {count + 1}"
            self.sensor_display_names[key] = display_name[:24]
            self.plot_visibility.setdefault(key, True)
        self._build_sensor_columns(sensor_keys)

    def _clear_layout(self, layout: QGridLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                while child_layout.count():
                    child = child_layout.takeAt(0)
                    child_widget = child.widget()
                    if child_widget is not None:
                        child_widget.deleteLater()

    def _build_sensor_columns(self, sensor_keys: List[str]) -> None:
        self._clear_layout(self.sensor_columns_layout)
        self.sensor_live_labels = {}

        groups: Dict[str, List[str]] = {}
        for key in sensor_keys:
            if key.endswith(":temp_c"):
                groups.setdefault(self._categorize_sensor(key), []).append(key)

        cpu_all = sorted(groups.get("CPU", []))
        midpoint = (len(cpu_all) + 1) // 2
        columns = [
            ("CPU", cpu_all[:midpoint]),
            ("CPU (cont.)", cpu_all[midpoint:]),
            ("GPU", sorted(groups.get("GPU", []))),
            (
                "Storage",
                sorted(groups.get("NVMe / M.2", []) + groups.get("SSD", []) + groups.get("SATA", []) + groups.get("HDD", [])),
            ),
            (
                "Other",
                sorted(groups.get("Chipset", []) + groups.get("System (ACPI)", []) + groups.get("Other", [])),
            ),
        ]

        for column_index, (header, sensors) in enumerate(columns):
            frame = QFrame()
            frame.setFrameShape(QFrame.StyledPanel)
            frame.setFrameShadow(QFrame.Raised)
            frame_layout = QVBoxLayout(frame)
            frame_layout.setContentsMargins(8, 8, 8, 8)
            frame_layout.setSpacing(4)

            title = QLabel(header)
            title.setStyleSheet("font-weight: 600;")
            frame_layout.addWidget(title)

            if not sensors:
                empty = QLabel("-")
                empty.setStyleSheet("color: grey;")
                frame_layout.addWidget(empty)
            else:
                for key in sensors:
                    row = QHBoxLayout()
                    row.setContentsMargins(0, 0, 0, 0)
                    row.setSpacing(6)
                    name_label = QLabel(self.sensor_display_names.get(key, self._format_sensor_label(key)))
                    value_label = QLabel("-")
                    value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    value_label.setStyleSheet("color: #0055aa;")
                    row.addWidget(name_label)
                    row.addStretch(1)
                    row.addWidget(value_label)
                    frame_layout.addLayout(row)
                    self.sensor_live_labels[key] = value_label
            frame_layout.addStretch(1)
            self.sensor_columns_layout.addWidget(frame, 0, column_index)

    def _update_live_temps(self, readings: Dict[str, float]) -> None:
        for key, label in self.sensor_live_labels.items():
            value = readings.get(key)
            label.setText(f"{value:.1f} C" if value is not None else "-")

    def check_running_processes(self) -> None:
        offenders = find_busy_user_processes(cpu_threshold=float(self.cpu_threshold_spin.value()))
        if not offenders:
            self._set_warning("")
            self._set_status("No high-CPU user processes detected.")
            QMessageBox.information(self, "Process Check", "No high-CPU user processes detected.")
            return

        lines = ["Potentially interfering processes:"]
        for pid, name, cpu in offenders[:12]:
            lines.append(f"PID={pid:<7} CPU={cpu:>5.1f}%  {name}")
        message = "\n".join(lines)
        self._set_warning("High-CPU user processes detected. See dialog.")
        self._set_status("Process check completed with warnings.")
        QMessageBox.warning(self, "Process Check", message)

    def start_testing(self) -> None:
        if self.running:
            return

        duration_min = int(self.duration_spin.value())
        interval_sec = float(self.interval_spin.value())
        cpu_workers = int(self.cpu_workers_spin.value())
        cpu_threshold = float(self.cpu_threshold_spin.value())

        offenders = find_busy_user_processes(cpu_threshold=cpu_threshold)
        if offenders and self.enforce_clean_checkbox.isChecked():
            names = ", ".join([f"{name}({cpu:.1f}%)" for _, name, cpu in offenders[:6]])
            QMessageBox.critical(
                self,
                "Cannot Start",
                f"High-CPU user processes were detected and enforce-clean is enabled:\n{names}",
            )
            return

        self._set_warning("Proceeding with warnings: active user processes detected." if offenders else "")
        self.running = True
        self.stop_event.clear()
        self.phase_summaries = []
        self.run_started_epoch = time.time()
        self.plot_origin_epoch = self.run_started_epoch

        known_sensors = sorted(self.sensor_display_names) or sorted(self.sensor_data)
        self.sensor_data = {sensor: [] for sensor in known_sensors}
        self.sensor_time_data = {sensor: [] for sensor in known_sensors}
        self._refresh_sensor_metadata()
        self._refresh_plot()

        self.buttons_changed.emit(False, True)

        self.worker_thread = threading.Thread(
            target=self._run_test_worker,
            args=(duration_min, interval_sec, cpu_workers, self.gpu_stress_cmd_edit.text().strip()),
            daemon=True,
        )
        self.worker_thread.start()

    def stop_testing(self) -> None:
        if not self.running:
            return
        self.stop_event.set()
        self._set_status("Stopping test run...")

    def _run_phase(self, phase: GuiPhase, duration_sec: int, interval_sec: float, cpu_workers: int, gpu_stress_cmd: str) -> dict:
        self._set_phase(f"{phase.name} ({'cooled' if phase.cooled else 'uncooled'}, {'stress' if phase.stress else 'idle'})")
        self._set_status(phase.pre_message)
        self._set_remaining_seconds(duration_sec)

        stressors = []
        if phase.stress:
            stressors = start_stressors(cpu_workers=cpu_workers, gpu_stress_cmd=(gpu_stress_cmd or None))
            self._set_status(f"{phase.name}: stressors running")
            time.sleep(2.0)

        started = time.time()
        next_sample = started
        samples = 0
        sensors_seen = set()

        try:
            while not self.stop_event.is_set():
                now = time.time()
                elapsed = now - started
                if elapsed >= duration_sec:
                    break

                if now < next_sample:
                    time.sleep(min(0.25, next_sample - now))
                    continue

                raw_readings = collect_temperatures()
                plot_readings = self._deduplicate_sensors(raw_readings)
                ts = utc_now_iso()
                for sensor, value in raw_readings.items():
                    sensors_seen.add(sensor)
                    if self.csv_writer is not None:
                        self.csv_writer.writerow(
                            [
                                ts,
                                phase.name,
                                "cooled" if phase.cooled else "uncooled",
                                "stress" if phase.stress else "idle",
                                sensor,
                                f"{value:.3f}",
                            ]
                        )

                samples += 1
                run_elapsed = (time.time() - self.run_started_epoch) if self.run_started_epoch else elapsed
                self.sensor_sample_ready.emit(plot_readings, run_elapsed)

                remaining = max(0, duration_sec - int(elapsed))
                self._set_remaining_seconds(remaining)
                self._set_status(f"Running {phase.name}: sample {samples}, sensors {len(raw_readings)}")
                next_sample += interval_sec
        finally:
            if stressors:
                stop_stressors(stressors)

        return {
            "phase": phase.name,
            "cooled": phase.cooled,
            "stress": phase.stress,
            "duration_sec": int(time.time() - started),
            "samples": samples,
            "sensors_seen": sorted(sensors_seen),
        }

    def _confirm_cooler_state(self, *, cooled: bool) -> bool:
        event = threading.Event()
        payload = {"result": False}
        self._pending_cooler_confirmation = (event, payload)
        self._set_status(f"Waiting for cooler {'ON' if cooled else 'OFF'} confirmation.")
        self.cooler_confirm_requested.emit(cooled)

        while not self.stop_event.is_set():
            if event.wait(timeout=0.2):
                break

        pending = self._pending_cooler_confirmation
        self._pending_cooler_confirmation = None
        if self.stop_event.is_set() and not event.is_set():
            return False
        if pending is None:
            return False
        return payload["result"] and not self.stop_event.is_set()

    def _show_cooler_confirmation(self, cooled: bool) -> None:
        pending = self._pending_cooler_confirmation
        if pending is None:
            return
        event, payload = pending
        state_text = "ON" if cooled else "OFF"
        phase_text = "cooled" if cooled else "uncooled"
        result = QMessageBox.question(
            self,
            "Confirm Cooler State",
            f"Confirm the external cooler is turned {state_text}.\n\nClick OK to begin the {phase_text} phase.",
            QMessageBox.Ok | QMessageBox.Cancel,
            QMessageBox.Ok,
        )
        payload["result"] = result == QMessageBox.Ok
        event.set()

    def _format_cooler_summary(self, assessment: dict) -> str:
        headline = "KEEP" if assessment.get("keep_recommended") else "RETURN"
        reason = assessment.get("reason", "no clear signal")
        best_drop = assessment.get("best_temp_drop_c", 0.0)
        freq_gain = assessment.get("best_cpu_freq_gain_pct", 0.0)
        lines = [
            f"Cooler verdict: {headline}",
            f"Result: {reason}.",
            f"Best temp drop: {best_drop:.1f} C | Best CPU clock gain: {freq_gain:.1f}%.",
            "Noise not scored (not measured).",
        ]
        tradeoff = assessment.get("clock_tradeoff_note")
        if tradeoff:
            lines.append(tradeoff)
        return "\n".join(lines)

    def _run_test_worker(self, duration_min: int, interval_sec: float, cpu_workers: int, gpu_stress_cmd: str) -> None:
        out_dir = Path("results")
        out_dir.mkdir(parents=True, exist_ok=True)

        experiment_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = out_dir / f"temperature_log_{experiment_id}.csv"
        metadata_path = out_dir / f"run_metadata_{experiment_id}.json"
        self.output_changed.emit(str(out_dir))

        phases = [
            GuiPhase("uncooled_idle", cooled=False, stress=False, pre_message="Starting uncooled idle phase."),
            GuiPhase("uncooled_stress", cooled=False, stress=True, pre_message="Starting uncooled stress phase."),
            GuiPhase("cooled_idle", cooled=True, stress=False, pre_message="Starting cooled idle phase."),
            GuiPhase("cooled_stress", cooled=True, stress=True, pre_message="Starting cooled stress phase."),
        ]

        duration_sec = duration_min * 60

        try:
            with csv_path.open("w", newline="") as handle:
                self.csv_writer = csv.writer(handle)
                self.csv_writer.writerow(["timestamp_utc", "phase", "cooling_state", "load_state", "sensor", "value"])

                previous_cooling_state: Optional[bool] = None
                for phase in phases:
                    if self.stop_event.is_set():
                        break
                    if previous_cooling_state is None or previous_cooling_state != phase.cooled:
                        if getattr(self, "_self_test_mode", False):
                            print(f"[self-test] requesting confirmation for cooled={phase.cooled}")
                        ok = self._confirm_cooler_state(cooled=phase.cooled)
                        if not ok:
                            self.stop_event.set()
                            break

                    summary = self._run_phase(
                        phase=phase,
                        duration_sec=duration_sec,
                        interval_sec=interval_sec,
                        cpu_workers=cpu_workers,
                        gpu_stress_cmd=gpu_stress_cmd,
                    )

                    phase_index = len(self.phase_summaries)
                    if phase_index < len(phases) - 1 and not self.stop_event.is_set():
                        summary["post_phase_steady_state"] = wait_until_temperatures_steady(
                            interval_sec=interval_sec,
                            status_cb=self._set_status,
                        )

                    self.phase_summaries.append(summary)
                    if getattr(self, "_self_test_mode", False):
                        print(f"[self-test] appended summary for phase {summary.get('phase')}")
                    previous_cooling_state = phase.cooled

            metadata = {
                "experiment_id": experiment_id,
                "created_utc": utc_now_iso(),
                "duration_per_phase_min": duration_min,
                "interval_sec": interval_sec,
                "cpu_workers": cpu_workers,
                "gpu_stress_cmd": gpu_stress_cmd,
                "gpu_vendor_guess": infer_gpu_vendor(),
                "phase_count_completed": len(self.phase_summaries),
                "phases": self.phase_summaries,
                "files": {
                    "csv": str(csv_path),
                    "metadata": str(metadata_path),
                },
            }

            cooler_assessment: Optional[dict] = None
            if not self.stop_event.is_set() and len(self.phase_summaries) == len(phases):
                cooler_assessment = summarize_cooler_effect_from_csv(csv_path)
                metadata["cooler_assessment"] = cooler_assessment

            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

            if self.stop_event.is_set():
                self._set_status("Test stopped by user.")
            else:
                self._set_status("All phases complete.")
                if cooler_assessment is not None:
                    summary_text = self._format_cooler_summary(cooler_assessment)
                    self._set_status(cooler_assessment.get("summary_short", "Cooler summary ready."))
                    self.dialog_info_requested.emit("Cooler Summary", summary_text)
            self._set_phase("Finished")
            self._set_remaining_seconds(0)
        except Exception as exc:
            self._set_status("Run failed")
            self._set_phase("Error")
            self.dialog_error_requested.emit("Run Error", str(exc))
        finally:
            self.csv_writer = None
            self.running = False
            self.stop_event.clear()
            self.buttons_changed.emit(True, False)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._poll_active = False
        self.live_poll_timer.stop()
        if self.running:
            result = QMessageBox.question(
                self,
                "Quit",
                "A test is running. Stop and exit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if result != QMessageBox.Yes:
                event.ignore()
                return
            self.stop_testing()
            if self.worker_thread and self.worker_thread.is_alive():
                self.worker_thread.join(timeout=3.0)
        event.accept()


def main() -> int:
    # Support a self-test mode for automated verification:
    # `python temp_compare_gui.py --self-test` will run a short auto-confirming
    # sequence that exercises uncooled -> cooled phases and verifies logging.
    if "--self-test" in sys.argv:
        return run_self_test()

    app = QApplication(sys.argv)
    window = TempCompareWindow()
    window.show()
    # Ensure the window is raised and activated after the event loop starts
    QTimer.singleShot(250, lambda: (window.raise_(), window.activateWindow()))
    return app.exec_()


def run_self_test() -> int:
    """Run a short automated test flow and print verification output.

    This auto-confirms cooler prompts, uses a synthetic temperature source,
    runs very short phase durations, and prints sample counts and phase
    summaries to stdout so CI / local runs can verify behavior.
    """
    print("Starting self-test mode: auto-confirming cooler prompts and using synthetic temps")
    # Create QApplication for widgets
    app = QApplication.instance() or QApplication([])

    # Use a fast synthetic sampler
    def _fake_collect():
        t = time.time()
        return {
            "coretemp:Package id 0:temp_c": 40.0 + (t % 5),
            "k10temp:cpu temp_c": 45.0 + ((t + 1) % 4),
        }

    # Patch the module-level name used by the GUI
    globals()["collect_temperatures"] = _fake_collect

    # Create the window and auto-confirm cooler prompts (with logging)
    win = TempCompareWindow()
    def _auto_confirm(*, cooled: bool) -> bool:
        print(f"AUTO_CONFIRM called for cooled={cooled}")
        return True

    win._confirm_cooler_state = _auto_confirm

    # Speed up live polling for the self-test
    try:
        win.live_poll_timer.setInterval(250)
    except Exception:
        pass

    # Enable internal self-test debug prints
    win._self_test_mode = True

    # Start the worker in background with moderate short durations so phases
    # can progress during the self-test.
    duration_min = 0.15
    interval_sec = 0.5
    thr = threading.Thread(
        target=lambda: win._run_test_worker(duration_min=duration_min, interval_sec=interval_sec, cpu_workers=1, gpu_stress_cmd=""),
        daemon=True,
    )
    thr.start()

    # Process Qt events while the worker runs and print periodic diagnostics
    start = time.time()
    try:
        while thr.is_alive() and (time.time() - start) < 60.0:
            app.processEvents()
            time.sleep(0.15)
            # Print a small diagnostic line showing collected sample counts
            counts = {k: len(v) for k, v in win.sensor_time_data.items()}
            print("sample_counts:", counts, "phases:", len(win.phase_summaries))
    except KeyboardInterrupt:
        pass

    thr.join(timeout=1.0)
    print("Self-test complete. Phase summaries:", win.phase_summaries)
    # List any CSV files created in results/
    try:
        out_files = sorted(Path("results").glob("temperature_log_*.csv"))
        print("CSV files:", [str(p) for p in out_files[-3:]])
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


if __name__ == "__main__":
    raise SystemExit(main())
