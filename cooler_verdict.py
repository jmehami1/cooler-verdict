#!/usr/bin/env python3
"""Cooler Verdict: benchmark external laptop/PC coolers with GUI auto-detection.

This single-file version combines the command-line runner and the PyQt5 GUI.
By default it starts the GUI when a display environment and PyQt5 are available;
otherwise it falls back to the CLI runner.

Usage:
  python cooler_verdict.py             # auto GUI/CLI
  python cooler_verdict.py --gui       # require GUI
  python cooler_verdict.py --cli       # force CLI
  python cooler_verdict.py --help      # CLI options
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import psutil


@dataclass
class Phase:
    name: str
    cooled: bool
    stress: bool


@dataclass
class RunningStressor:
    name: str
    proc: subprocess.Popen


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _is_valid_temperature_c(temp: float) -> bool:
    """Check if a temperature reading is physically reasonable (in Celsius).
    
    Rejects NaN, inf, and values outside -50°C to 150°C range.
    """
    if temp is None or math.isnan(temp) or math.isinf(temp):
        return False
    return -50.0 <= temp <= 150.0


def _is_valid_cpu_frequency_mhz(freq: float) -> bool:
    """Check if a CPU frequency reading is physically reasonable (in MHz).
    
    Rejects NaN, inf, and values outside 1 MHz to 10000 MHz range.
    """
    if freq is None or math.isnan(freq) or math.isinf(freq):
        return False
    return 1.0 <= freq <= 10000.0


def detect_nvidia_gpu() -> Optional[str]:
    """Detect NVIDIA GPU. Returns device name if found, None otherwise."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL
        )
        gpus = [line.strip() for line in output.strip().splitlines() if line.strip()]
        return gpus[0] if gpus else None
    except Exception:
        return None


def detect_amd_gpu() -> Optional[str]:
    """Detect AMD GPU (RDNA, NAVI, etc). Returns device name if found, None otherwise."""
    # Check for rocm-smi (AMD GPU driver)
    if shutil.which("rocm-smi"):
        try:
            output = subprocess.check_output(
                ["rocm-smi", "--showid"],
                text=True,
                stderr=subprocess.DEVNULL
            )
            if output and "GPU ID" in output:
                return "AMD Radeon (ROCm)"
        except Exception:
            pass
    
    # Check for AMD GPU via lspci
    if shutil.which("lspci"):
        try:
            output = subprocess.check_output(
                ["lspci"],
                text=True,
                stderr=subprocess.DEVNULL
            )
            for line in output.splitlines():
                if any(vendor in line.upper() for vendor in ["AMD", "RADEON", "NAVI", "RDNA"]):
                    if "VGA" in line or "3D" in line:
                        return line.split(": ", 1)[-1] if ": " in line else "AMD Radeon GPU"
        except Exception:
            pass
    
    return None


def detect_intel_gpu() -> Optional[str]:
    """Detect Intel GPU (Arc, Iris, UHD). Returns device name if found, None otherwise."""
    # Check for Intel GPU via lspci
    if shutil.which("lspci"):
        try:
            output = subprocess.check_output(
                ["lspci"],
                text=True,
                stderr=subprocess.DEVNULL
            )
            for line in output.splitlines():
                if "Intel" in line and any(gpu_type in line for gpu_type in ["Arc", "Iris", "UHD", "Xe"]):
                    if "VGA" in line or "3D" in line:
                        return line.split(": ", 1)[-1] if ": " in line else "Intel GPU"
        except Exception:
            pass
    
    # Check for Intel GPU driver (i915, xe)
    if Path("/sys/module/i915").exists():
        return "Intel GPU (i915)"
    if Path("/sys/module/xe").exists():
        return "Intel GPU (Xe)"
    
    return None


def get_gpu_info() -> dict:
    """Detect available GPUs and their capabilities.
    
    Returns:
        {
            "vendor": "nvidia" | "amd" | "intel" | "none",
            "device": device_name or None,
            "available": bool,
            "can_run_code": bool,
            "message": str
        }
    """
    nvidia = detect_nvidia_gpu()
    if nvidia:
        return {
            "vendor": "nvidia",
            "device": nvidia,
            "available": True,
            "can_run_code": check_gpu_can_run_code("nvidia"),
            "message": f"NVIDIA GPU detected: {nvidia}"
        }
    
    amd = detect_amd_gpu()
    if amd:
        return {
            "vendor": "amd",
            "device": amd,
            "available": True,
            "can_run_code": check_gpu_can_run_code("amd"),
            "message": f"AMD GPU detected: {amd}"
        }
    
    intel = detect_intel_gpu()
    if intel:
        return {
            "vendor": "intel",
            "device": intel,
            "available": True,
            "can_run_code": check_gpu_can_run_code("intel"),
            "message": f"Intel GPU detected: {intel}"
        }
    
    return {
        "vendor": "none",
        "device": None,
        "available": False,
        "can_run_code": False,
        "message": "No supported GPU detected"
    }


def check_gpu_can_run_code(vendor: str) -> bool:
    """Check if the detected GPU can actually run code.
    
    This tests the GPU capabilities for each vendor.
    """
    if vendor == "nvidia":
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "-L"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5
            )
            return len(output.strip()) > 0
        except Exception:
            return False
    
    elif vendor == "amd":
        if shutil.which("rocm-smi"):
            try:
                output = subprocess.check_output(
                    ["rocm-smi"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=5
                )
                return "GPU" in output
            except Exception:
                return False
        return False
    
    elif vendor == "intel":
        # Check if Intel GPU driver is loaded
        if Path("/sys/module/i915").exists() or Path("/sys/module/xe").exists():
            return True
        return False
    
    return False


def infer_gpu_vendor() -> str:
    """Legacy function for backward compatibility. Returns vendor name."""
    info = get_gpu_info()
    return info["vendor"]


def get_gpu_stress_cmd() -> Optional[str]:
    """Generate a mild GPU stress command appropriate for the detected GPU.
    
    Returns None if no GPU or stress tool is available.
    """
    gpu_info = get_gpu_info()
    vendor = gpu_info["vendor"]
    
    if vendor == "nvidia":
        # Use nvidia-smi or CUDA if available
        if shutil.which("nvidia-smi"):
            # Mild stress: query GPU info periodically
            return "while true; do nvidia-smi -q -d UTILIZATION > /dev/null 2>&1; sleep 0.5; done"
        if shutil.which("cuda-samples"):
            return "nbody -benchmark -numbodies=100000 2>/dev/null &"
    
    elif vendor == "amd":
        # Use rocm-smi or clpeak for mild testing
        if shutil.which("rocm-smi"):
            return "while true; do rocm-smi > /dev/null 2>&1; sleep 0.5; done"
        if shutil.which("clpeak"):
            return "clpeak --device gpu --single-precision 2>/dev/null &"
    
    elif vendor == "intel":
        # Intel GPU: use tools if available
        if shutil.which("intel_gpu_top"):
            return "intel_gpu_top -l 1 > /dev/null 2>&1 &"
        if shutil.which("clpeak"):
            return "clpeak --device gpu --single-precision 2>/dev/null &"
    
    return None


def detect_nvidia_temps() -> Dict[str, float]:
    if not shutil.which("nvidia-smi"):
        return {}
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,temperature.gpu,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        raw = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}

    out: Dict[str, float] = {}
    for line in raw.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 4:
            continue
        idx, name, temp_s, util_s = parts
        try:
            temp = float(temp_s)
            util = float(util_s)
        except ValueError:
            continue
        if _is_valid_temperature_c(temp):
            out[f"gpu{idx}:{name}:temp_c"] = temp
        if not math.isnan(util) and 0 <= util <= 100:
            out[f"gpu{idx}:{name}:util_percent"] = util
    return out


def detect_psutil_temps() -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        groups = psutil.sensors_temperatures(fahrenheit=False)
    except Exception:
        return out

    for group_name, entries in groups.items():
        for idx, entry in enumerate(entries):
            label = entry.label or f"sensor{idx}"
            key = f"{group_name}:{label}:temp_c"
            if entry.current is not None and _is_valid_temperature_c(entry.current):
                out[key] = float(entry.current)
    return out


def detect_linux_thermal_zone_temps() -> Dict[str, float]:
    root = Path("/sys/class/thermal")
    if not root.exists():
        return {}

    out: Dict[str, float] = {}
    for zone in root.glob("thermal_zone*"):
        tfile = zone / "temp"
        typefile = zone / "type"
        if not tfile.exists():
            continue
        try:
            raw_temp = tfile.read_text().strip()
            raw_type = typefile.read_text().strip() if typefile.exists() else zone.name
            val = float(raw_temp)
            if val > 200.0:
                val = val / 1000.0
            if _is_valid_temperature_c(val):
                out[f"linux_thermal:{raw_type}:temp_c"] = val
        except Exception:
            continue
    return out


def detect_psutil_core_freqs() -> Dict[str, float]:
    """Return current CPU frequency for each logical core in MHz."""
    out: Dict[str, float] = {}
    try:
        freqs = psutil.cpu_freq(percpu=True)
    except Exception:
        return out

    if not freqs:
        return out

    valid_vals: List[float] = []
    for idx, freq in enumerate(freqs):
        current = getattr(freq, "current", None)
        if current is None:
            continue
        try:
            cur_mhz = float(current)
        except (TypeError, ValueError):
            continue
        if _is_valid_cpu_frequency_mhz(cur_mhz):
            out[f"cpu_freq:core{idx}:mhz"] = cur_mhz
            valid_vals.append(cur_mhz)

    if valid_vals:
        out["cpu_freq:avg:mhz"] = sum(valid_vals) / len(valid_vals)
    return out



def detect_sensors_cmd_temps() -> Dict[str, float]:
    """Parse output of 'sensors' command for all temperature readings (including SSD/NVMe)."""
    import re
    out: Dict[str, float] = {}
    try:
        import shutil
        if not shutil.which("sensors"):
            return out
        import subprocess
        raw = subprocess.check_output(["sensors"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return out
    # Regex for lines like: 'Composite:    +38.9°C  (low  =  -0.1°C, high = +84.9°C)
    temp_re = re.compile(r"([\w\-/ ]+):\s*([+\-]?[0-9]+\.?[0-9]*)°C")
    for line in raw.splitlines():
        m = temp_re.search(line)
        if m:
            label = m.group(1).strip().replace(" ", "_")
            try:
                val = float(m.group(2))
                if _is_valid_temperature_c(val):
                    out[f"sensors:{label}:temp_c"] = val
            except Exception:
                continue
    return out

def collect_temperatures() -> Dict[str, float]:
    sensors: Dict[str, float] = {}
    sensors.update(detect_psutil_temps())
    sensors.update(detect_linux_thermal_zone_temps())
    sensors.update(detect_nvidia_temps())
    sensors.update(detect_sensors_cmd_temps())
    sensors.update(detect_psutil_core_freqs())
    return sensors


def is_temperature_series_steady(
    values: List[float],
    interval_sec: float,
    window_points: int = 12,
    min_points: int = 12,
    slope_threshold_c_per_min: float = 0.25,
    max_range_c: float = 1.5,
    max_stddev_c: float = 0.45,
) -> bool:
    """Heuristic steady-state check using slope, range, and noise in a recent window."""
    if interval_sec <= 0:
        return False
    if len(values) < max(min_points, 2):
        return False

    window = values[-window_points:] if window_points > 0 else values
    if len(window) < max(min_points, 2):
        return False

    if any(v is None or math.isnan(v) for v in window):
        return False

    n = len(window)
    xs = [(i * interval_sec) / 60.0 for i in range(n)]
    x_mean = sum(xs) / n
    y_mean = sum(window) / n

    var_x = sum((x - x_mean) ** 2 for x in xs)
    if var_x == 0:
        return False
    cov_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, window))
    slope_c_per_min = cov_xy / var_x

    value_range = max(window) - min(window)
    variance = sum((y - y_mean) ** 2 for y in window) / n
    stddev = math.sqrt(max(0.0, variance))

    return (
        abs(slope_c_per_min) <= slope_threshold_c_per_min
        and value_range <= max_range_c
        and stddev <= max_stddev_c
    )


def all_temperatures_steady(
    history_by_sensor: Dict[str, List[float]],
    interval_sec: float,
    window_points: int = 12,
    min_points: int = 12,
) -> bool:
    temp_items = [(k, v) for k, v in history_by_sensor.items() if k.endswith(":temp_c")]
    if not temp_items:
        return False
    for _sensor, values in temp_items:
        if not is_temperature_series_steady(
            values=values,
            interval_sec=interval_sec,
            window_points=window_points,
            min_points=min_points,
        ):
            return False
    return True


def wait_until_temperatures_steady(
    interval_sec: float,
    timeout_sec: int = 900,
    window_points: int = 12,
    min_points: int = 12,
    status_cb: Optional[Callable[[str], None]] = None,
    progress_cb: Optional[Callable[[int, int, int], None]] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> dict:
    """Poll temperatures until all detected temperature sensors are approximately steady."""
    if interval_sec <= 0:
        interval_sec = 1.0

    def emit(msg: str) -> None:
        if status_cb:
            status_cb(msg)
        else:
            print(msg)

    started = time.time()
    next_sample = started
    history: Dict[str, List[float]] = {}
    samples = 0

    emit("Waiting for temperatures to reach steady state...")
    while True:
        now = time.time()
        elapsed = now - started
        elapsed_i = max(0, int(elapsed))
        remaining_i = max(0, int(math.ceil(timeout_sec - elapsed)))
        if progress_cb:
            progress_cb(elapsed_i, remaining_i, timeout_sec)
        if cancel_cb and cancel_cb():
            emit("Steady-state check cancelled.")
            if progress_cb:
                progress_cb(elapsed_i, 0, timeout_sec)
            return {
                "reached_steady": False,
                "timed_out": False,
                "cancelled": True,
                "wait_duration_sec": int(elapsed),
                "samples": samples,
                "temp_sensors_tracked": len([k for k in history if k.endswith(":temp_c")]),
            }
        if elapsed >= timeout_sec:
            emit("Steady-state timeout reached; continuing to next phase.")
            if progress_cb:
                progress_cb(max(0, int(timeout_sec)), 0, timeout_sec)
            return {
                "reached_steady": False,
                "timed_out": True,
                "cancelled": False,
                "wait_duration_sec": int(elapsed),
                "samples": samples,
                "temp_sensors_tracked": len([k for k in history if k.endswith(":temp_c")]),
            }

        if now < next_sample:
            time.sleep(min(0.5, next_sample - now))
            continue

        readings = collect_temperatures()
        for sensor, value in readings.items():
            if not sensor.endswith(":temp_c"):
                continue
            if value is None or math.isnan(value):
                continue
            history.setdefault(sensor, []).append(float(value))

        samples += 1
        temp_sensor_count = len([k for k in history if k.endswith(":temp_c")])
        if samples % 6 == 0:
            emit(
                f"  steady-check: {int(elapsed)}s elapsed, samples={samples}, temp_sensors={temp_sensor_count}"
            )

        if all_temperatures_steady(
            history_by_sensor=history,
            interval_sec=interval_sec,
            window_points=window_points,
            min_points=min_points,
        ):
            emit("All temperatures are at approximate steady state.")
            if progress_cb:
                progress_cb(elapsed_i, 0, timeout_sec)
            return {
                "reached_steady": True,
                "timed_out": False,
                "cancelled": False,
                "wait_duration_sec": int(elapsed),
                "samples": samples,
                "temp_sensors_tracked": temp_sensor_count,
            }

        next_sample += interval_sec


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _rolling_mean(values: List[float], window_points: int) -> List[float]:
    """Return trailing rolling mean for each point in a series."""
    if not values:
        return []

    window = max(1, int(window_points))
    out: List[float] = []
    running = 0.0
    for idx, value in enumerate(values):
        running += value
        if idx >= window:
            running -= values[idx - window]
            out.append(running / window)
        else:
            out.append(running / (idx + 1))
    return out


def _build_rolling_overtime_profile(
    uncooled_series: List[float],
    cooled_series: List[float],
    rolling_window_points: int,
) -> Dict[str, Optional[float]]:
    """Compare uncooled vs cooled trajectories using rolling averages only."""
    compared_points = min(len(uncooled_series), len(cooled_series))
    if compared_points == 0:
        return {
            "samples_compared": 0,
            "mean_rolling_drop_c": None,
            "peak_rolling_drop_c": None,
            "worst_rolling_drop_c": None,
            "points_cooler_pct": None,
        }

    uncooled_smoothed = _rolling_mean(uncooled_series[:compared_points], rolling_window_points)
    cooled_smoothed = _rolling_mean(cooled_series[:compared_points], rolling_window_points)
    rolling_deltas = [u - c for u, c in zip(uncooled_smoothed, cooled_smoothed)]
    cooler_points = sum(1 for delta in rolling_deltas if delta > 0.0)

    return {
        "samples_compared": compared_points,
        "mean_rolling_drop_c": _mean(rolling_deltas),
        "peak_rolling_drop_c": max(rolling_deltas),
        "worst_rolling_drop_c": min(rolling_deltas),
        "points_cooler_pct": (cooler_points / compared_points) * 100.0,
    }


def _tail_window(values: List[float], fraction: float = 0.35, min_points: int = 12) -> List[float]:
    if not values:
        return []
    n = len(values)
    take = max(1, min(n, max(min_points, int(n * fraction))))
    return values[-take:]


def _is_cpu_temp_sensor(sensor: str) -> bool:
    s = sensor.lower()
    if not s.endswith(":temp_c"):
        return False
    return any(
        token in s
        for token in (
            "coretemp",
            "k10temp",
            "zenpower",
            "cpu",
            "core",
            "package",
            "tctl",
            "tdie",
            "x86_pkg_temp",
            "ccd",
        )
    )


def _is_gpu_temp_sensor(sensor: str) -> bool:
    s = sensor.lower()
    return s.endswith(":temp_c") and "gpu" in s


def _find_duplicate_sensors(
    phase_sensor_temps: Dict[str, Dict[str, List[float]]],
    tolerance: float = 0.01,
) -> Dict[str, str]:
    """Identify duplicate sensors with identical or near-identical readings.
    
    Returns a mapping of sensor names to keep. If duplicates are found, keeps
    the first occurrence and returns it for all duplicates.
    
    Args:
        phase_sensor_temps: Dict mapping phase names to sensor dicts
        tolerance: Maximum relative difference to consider readings identical
    
    Returns:
        Dict mapping all sensors to their canonical representative
    """
    # Collect all sensors and compute their fingerprint (mean value across all phases)
    sensor_fingerprints: Dict[str, Optional[float]] = {}
    sensor_first_occurrence: Dict[str, str] = {}  # Maps fingerprint to first sensor name
    
    for phase, sensor_dict in phase_sensor_temps.items():
        for sensor, values in sensor_dict.items():
            if not values:
                continue
            mean_val = sum(values) / len(values)
            if sensor not in sensor_fingerprints:
                sensor_fingerprints[sensor] = mean_val
    
    # Find duplicates: sensors with mean values within tolerance of each other
    sensors_sorted = sorted(sensor_fingerprints.keys())
    canonical_map: Dict[str, str] = {sensor: sensor for sensor in sensors_sorted}
    
    for i, sensor_a in enumerate(sensors_sorted):
        if canonical_map[sensor_a] != sensor_a:
            continue  # Already mapped to a duplicate
        
        mean_a = sensor_fingerprints[sensor_a]
        if mean_a is None:
            continue
        
        # Check against other sensors for duplicates
        for sensor_b in sensors_sorted[i + 1:]:
            if canonical_map[sensor_b] != sensor_b:
                continue  # Already mapped
            
            mean_b = sensor_fingerprints[sensor_b]
            if mean_b is None:
                continue
            
            # Check if readings are nearly identical
            max_val = max(abs(mean_a), abs(mean_b))
            if max_val == 0:
                relative_diff = 0.0
            else:
                relative_diff = abs(mean_a - mean_b) / max_val
            
            if relative_diff <= tolerance:
                # Mark sensor_b as a duplicate of sensor_a
                canonical_map[sensor_b] = sensor_a
    
    return canonical_map


def _deduplicate_phase_data(
    phase_sensor_temps: Dict[str, Dict[str, List[float]]],
    phase_component_sensors: Dict[str, Dict[str, set[str]]],
    canonical_map: Dict[str, str],
) -> None:
    """In-place deduplication: merge duplicate sensors into their canonical form.
    
    Modifies phase_sensor_temps and phase_component_sensors to consolidate duplicates.
    """
    for phase in phase_sensor_temps:
        # Consolidate values from duplicates into canonical sensors
        consolidated: Dict[str, List[float]] = {}
        for sensor, values in phase_sensor_temps[phase].items():
            canonical = canonical_map.get(sensor, sensor)
            if canonical not in consolidated:
                consolidated[canonical] = []
            consolidated[canonical].extend(values)
        
        phase_sensor_temps[phase] = consolidated
        
        # Update component tracking
        for component in phase_component_sensors[phase]:
            old_sensors = phase_component_sensors[phase][component]
            new_sensors = set()
            for sensor in old_sensors:
                canonical = canonical_map.get(sensor, sensor)
                new_sensors.add(canonical)
            phase_component_sensors[phase][component] = new_sensors


def summarize_cooler_effect_from_csv(csv_path: Path) -> dict:
    """Summarize cooler effectiveness from phase CSV data.

    Uses final-window averages per phase and various heuristics to determine if the cooler provides a meaningful improvement.
    """
    phase_names = ["uncooled_idle", "cooled_idle", "uncooled_stress", "cooled_stress"]
    phase_metrics: Dict[str, Dict[str, List[float]]] = {
        phase: {"cpu_temp": [], "gpu_temp": [], "storage_temp": [], "cpu_freq": []}
        for phase in phase_names
    }
    phase_sensor_temps: Dict[str, Dict[str, List[float]]] = {phase: {} for phase in phase_names}
    phase_sample_timestamps: Dict[str, set[str]] = {phase: set() for phase in phase_names}
    phase_component_sensors: Dict[str, Dict[str, set[str]]] = {
        phase: {"cpu": set(), "gpu": set(), "storage": set()} for phase in phase_names
    }
    phase_component_by_timestamp: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        phase: {"cpu": {}, "gpu": {}, "storage": {}} for phase in phase_names
    }

    rolling_window_points = 5

    if not csv_path.exists():
        return {
            "verdict": "inconclusive",
            "keep_recommended": False,
            "summary_short": "Inconclusive: no CSV data.",
        }

    def sensor_component(sensor: str) -> Optional[str]:
        if _is_cpu_temp_sensor(sensor):
            return "cpu"
        if _is_gpu_temp_sensor(sensor):
            return "gpu"
        lowered = sensor.lower()
        if sensor.endswith(":temp_c") and any(token in lowered for token in ("nvme", "ssd", "storage", "disk", "hdd", "sata")):
            return "storage"
        return None

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            phase = (row.get("phase") or "").strip()
            sensor = (row.get("sensor") or "").strip()
            value_raw = (row.get("value") or "").strip()
            ts = (row.get("timestamp_utc") or "").strip()
            if phase not in phase_metrics:
                continue
            try:
                value = float(value_raw)
            except ValueError:
                continue

            # Skip invalid temperature and frequency readings
            if sensor.endswith(":temp_c") and not _is_valid_temperature_c(value):
                continue
            if sensor.endswith(":mhz") and not _is_valid_cpu_frequency_mhz(value):
                continue

            if ts:
                phase_sample_timestamps[phase].add(ts)

            if sensor == "cpu_freq:avg:mhz":
                phase_metrics[phase]["cpu_freq"].append(value)
            component = sensor_component(sensor)
            if component == "cpu":
                phase_metrics[phase]["cpu_temp"].append(value)
            elif component == "gpu":
                phase_metrics[phase]["gpu_temp"].append(value)
            elif component == "storage":
                phase_metrics[phase]["storage_temp"].append(value)
            if sensor.endswith(":temp_c"):
                phase_sensor_temps[phase].setdefault(sensor, []).append(value)
                if component:
                    phase_component_sensors[phase][component].add(sensor)
                    if ts:
                        phase_component_by_timestamp[phase][component].setdefault(ts, []).append(value)

    phase_component_series: Dict[str, Dict[str, List[float]]] = {
        phase: {"cpu": [], "gpu": [], "storage": []} for phase in phase_names
    }
    for phase in phase_names:
        for component in ("cpu", "gpu", "storage"):
            ts_map = phase_component_by_timestamp[phase][component]
            for ts in sorted(ts_map.keys()):
                point_avg = _mean(ts_map[ts])
                if point_avg is not None:
                    phase_component_series[phase][component].append(point_avg)

    rolling_overtime_comparisons: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {
        "idle": {},
        "stress": {},
    }
    for comparison_name, uncooled_phase, cooled_phase in (
        ("idle", "uncooled_idle", "cooled_idle"),
        ("stress", "uncooled_stress", "cooled_stress"),
    ):
        for component in ("cpu", "gpu", "storage"):
            rolling_overtime_comparisons[comparison_name][component] = _build_rolling_overtime_profile(
                uncooled_series=phase_component_series[uncooled_phase][component],
                cooled_series=phase_component_series[cooled_phase][component],
                rolling_window_points=rolling_window_points,
            )

    # Deduplicate identical sensors before verdict calculation
    canonical_map = _find_duplicate_sensors(phase_sensor_temps, tolerance=0.01)
    _deduplicate_phase_data(phase_sensor_temps, phase_component_sensors, canonical_map)

    phase_stats: Dict[str, Dict[str, Optional[float]]] = {}
    for phase, metrics in phase_metrics.items():
        phase_stats[phase] = {
            "cpu_temp_avg": _mean(_tail_window(metrics["cpu_temp"])),
            "gpu_temp_avg": _mean(_tail_window(metrics["gpu_temp"])),
            "storage_temp_avg": _mean(_tail_window(metrics["storage_temp"])),
            "cpu_freq_avg_mhz": _mean(_tail_window(metrics["cpu_freq"])),
        }

    phase_sensor_temp_avg: Dict[str, Dict[str, Optional[float]]] = {
        phase: {
            sensor: _mean(_tail_window(values))
            for sensor, values in sensor_map.items()
        }
        for phase, sensor_map in phase_sensor_temps.items()
    }

    def pct_gain(new_val: Optional[float], old_val: Optional[float]) -> Optional[float]:
        if new_val is None or old_val is None or old_val == 0:
            return None
        return ((new_val - old_val) / old_val) * 100.0

    def temp_drop(uncooled_val: Optional[float], cooled_val: Optional[float]) -> Optional[float]:
        if uncooled_val is None or cooled_val is None:
            return None
        return uncooled_val - cooled_val

    comparisons = {
        "idle": {
            "cpu_temp_drop_c": temp_drop(
                phase_stats["uncooled_idle"]["cpu_temp_avg"],
                phase_stats["cooled_idle"]["cpu_temp_avg"],
            ),
            "gpu_temp_drop_c": temp_drop(
                phase_stats["uncooled_idle"]["gpu_temp_avg"],
                phase_stats["cooled_idle"]["gpu_temp_avg"],
            ),
            "storage_temp_drop_c": temp_drop(
                phase_stats["uncooled_idle"]["storage_temp_avg"],
                phase_stats["cooled_idle"]["storage_temp_avg"],
            ),
            "cpu_freq_gain_pct": pct_gain(
                phase_stats["cooled_idle"]["cpu_freq_avg_mhz"],
                phase_stats["uncooled_idle"]["cpu_freq_avg_mhz"],
            ),
        },
        "stress": {
            "cpu_temp_drop_c": temp_drop(
                phase_stats["uncooled_stress"]["cpu_temp_avg"],
                phase_stats["cooled_stress"]["cpu_temp_avg"],
            ),
            "gpu_temp_drop_c": temp_drop(
                phase_stats["uncooled_stress"]["gpu_temp_avg"],
                phase_stats["cooled_stress"]["gpu_temp_avg"],
            ),
            "storage_temp_drop_c": temp_drop(
                phase_stats["uncooled_stress"]["storage_temp_avg"],
                phase_stats["cooled_stress"]["storage_temp_avg"],
            ),
            "cpu_freq_gain_pct": pct_gain(
                phase_stats["cooled_stress"]["cpu_freq_avg_mhz"],
                phase_stats["uncooled_stress"]["cpu_freq_avg_mhz"],
            ),
        },
    }

    sensor_temp_drop_summary: List[Dict[str, object]] = []
    all_sensors = sorted(
        set(phase_sensor_temp_avg["uncooled_idle"]) | set(phase_sensor_temp_avg["uncooled_stress"])
    )
    for sensor in all_sensors:
        uncooled_idle = phase_sensor_temp_avg["uncooled_idle"].get(sensor)
        cooled_idle = phase_sensor_temp_avg["cooled_idle"].get(sensor)
        uncooled_stress = phase_sensor_temp_avg["uncooled_stress"].get(sensor)
        cooled_stress = phase_sensor_temp_avg["cooled_stress"].get(sensor)

        idle_drop_c = None
        idle_drop_pct = None
        if (
            uncooled_idle is not None
            and cooled_idle is not None
            and uncooled_idle > 0
        ):
            idle_drop_c = uncooled_idle - cooled_idle
            idle_drop_pct = (idle_drop_c / uncooled_idle) * 100.0

        stress_drop_c = None
        stress_drop_pct = None
        if (
            uncooled_stress is not None
            and cooled_stress is not None
            and uncooled_stress > 0
        ):
            stress_drop_c = uncooled_stress - cooled_stress
            stress_drop_pct = (stress_drop_c / uncooled_stress) * 100.0

        if idle_drop_pct is None and stress_drop_pct is None:
            continue

        sensor_temp_drop_summary.append(
            {
                "sensor": sensor,
                "idle_drop_c": round(idle_drop_c, 3) if idle_drop_c is not None else None,
                "idle_drop_pct": round(idle_drop_pct, 3) if idle_drop_pct is not None else None,
                "stress_drop_c": round(stress_drop_c, 3) if stress_drop_c is not None else None,
                "stress_drop_pct": round(stress_drop_pct, 3) if stress_drop_pct is not None else None,
            }
        )

    all_temp_drops = [
        v
        for profile in comparisons.values()
        for key, v in profile.items()
        if key.endswith("_drop_c") and v is not None
    ]
    all_freq_gains = [
        v
        for profile in comparisons.values()
        for key, v in profile.items()
        if key.endswith("_gain_pct") and v is not None
    ]

    best_temp_drop = max(all_temp_drops) if all_temp_drops else 0.0
    worst_temp_drop = min(all_temp_drops) if all_temp_drops else 0.0
    best_freq_gain = max(all_freq_gains) if all_freq_gains else 0.0

    idle_cpu_drop = comparisons["idle"]["cpu_temp_drop_c"]
    idle_gpu_drop = comparisons["idle"]["gpu_temp_drop_c"]
    stress_cpu_drop = comparisons["stress"]["cpu_temp_drop_c"]
    stress_gpu_drop = comparisons["stress"]["gpu_temp_drop_c"]
    stress_freq_gain_raw = comparisons["stress"]["cpu_freq_gain_pct"]
    stress_freq_gain = float(stress_freq_gain_raw) if stress_freq_gain_raw is not None else 0.0
    has_stress_clock_data = stress_freq_gain_raw is not None

    def max_or_none(values: List[Optional[float]]) -> Optional[float]:
        valid = [float(v) for v in values if v is not None]
        return max(valid) if valid else None

    def min_or_none(values: List[Optional[float]]) -> Optional[float]:
        valid = [float(v) for v in values if v is not None]
        return min(valid) if valid else None

    idle_best_cpu_gpu_drop = max_or_none([idle_cpu_drop, idle_gpu_drop])
    idle_worst_cpu_gpu_drop = min_or_none([idle_cpu_drop, idle_gpu_drop])
    stress_best_cpu_gpu_drop = max_or_none([stress_cpu_drop, stress_gpu_drop])
    stress_worst_cpu_gpu_drop = min_or_none([stress_cpu_drop, stress_gpu_drop])

    phase_complete = {phase: len(phase_sample_timestamps[phase]) > 0 for phase in phase_names}
    all_phases_complete = all(phase_complete.values())

    uncooled_cpu_gpu = set().union(
        *(phase_component_sensors[p]["cpu"] | phase_component_sensors[p]["gpu"] for p in ("uncooled_idle", "uncooled_stress"))
    )
    cooled_cpu_gpu = set().union(
        *(phase_component_sensors[p]["cpu"] | phase_component_sensors[p]["gpu"] for p in ("cooled_idle", "cooled_stress"))
    )
    comparable_cpu_gpu_sensors = sorted(uncooled_cpu_gpu.intersection(cooled_cpu_gpu))
    has_comparable_cpu_gpu = len(comparable_cpu_gpu_sensors) > 0

    sample_counts = {phase: len(phase_sample_timestamps[phase]) for phase in phase_names}
    min_samples = min(sample_counts.values()) if sample_counts else 0

    criteria_evaluation: List[Dict[str, object]] = []

    def add_criterion(
        name: str,
        status: str,
        level: str,
        threshold: str,
        observed: str,
        impact: str,
    ) -> None:
        criteria_evaluation.append(
            {
                "name": name,
                "status": status,
                "level": level,
                "threshold": threshold,
                "observed": observed,
                "impact": impact,
            }
        )

    add_criterion(
        name="data_complete",
        status="pass" if all_phases_complete else "fail",
        level="normal" if all_phases_complete else "none",
        threshold="All 4 phases completed",
        observed=(
            "completed phases: " + ", ".join([phase for phase, ok in phase_complete.items() if ok])
            if any(phase_complete.values())
            else "completed phases: none"
        ),
        impact="inconclusive_on_fail",
    )

    add_criterion(
        name="comparable_cpu_or_gpu_sensors_available",
        status="pass" if has_comparable_cpu_gpu else "fail",
        level="normal" if has_comparable_cpu_gpu else "none",
        threshold="At least one comparable CPU or GPU temperature sensor across cooled and uncooled phases",
        observed=(
            "comparable sensors: " + ", ".join(comparable_cpu_gpu_sensors)
            if comparable_cpu_gpu_sensors
            else "comparable sensors: none"
        ),
        impact="inconclusive_on_fail",
    )

    if min_samples < 5:
        minimum_samples_status = "fail"
        minimum_samples_level = "none"
    elif min_samples < 10:
        minimum_samples_status = "warning"
        minimum_samples_level = "weak"
    else:
        minimum_samples_status = "pass"
        minimum_samples_level = "normal"
    add_criterion(
        name="minimum_samples_met",
        status=minimum_samples_status,
        level=minimum_samples_level,
        threshold="Each phase has >=10 samples (warning 5-9, fail <5)",
        observed="sample counts: " + ", ".join(f"{phase}={count}" for phase, count in sample_counts.items()),
        impact="inconclusive_on_fail",
    )

    idle_improved = idle_best_cpu_gpu_drop is not None and idle_best_cpu_gpu_drop > 0.0
    idle_strong = idle_best_cpu_gpu_drop is not None and idle_best_cpu_gpu_drop >= 3.0
    add_criterion(
        name="idle_temperature_improved",
        status="pass" if idle_improved else "fail",
        level="strong" if idle_strong else ("normal" if idle_improved else "none"),
        threshold="Idle CPU/GPU temperature decreases with cooler (strong >=3 C drop)",
        observed=(
            f"idle CPU drop={idle_cpu_drop}, idle GPU drop={idle_gpu_drop}"
        ),
        impact="mandatory_keep_gate",
    )

    idle_penalty_fail = idle_worst_cpu_gpu_drop is not None and idle_worst_cpu_gpu_drop < -2.0
    add_criterion(
        name="idle_no_heat_penalty",
        status="fail" if idle_penalty_fail else "pass",
        level="none" if idle_penalty_fail else "normal",
        threshold="No idle CPU/GPU temperature more than 2 C hotter",
        observed=(
            f"worst idle CPU/GPU delta={idle_worst_cpu_gpu_drop:.1f} C"
            if idle_worst_cpu_gpu_drop is not None
            else "no idle CPU/GPU data"
        ),
        impact="return_on_fail",
    )

    if stress_best_cpu_gpu_drop is None:
        stress_temp_status = "fail"
        stress_temp_level = "none"
    elif stress_best_cpu_gpu_drop >= 8.0:
        stress_temp_status = "pass"
        stress_temp_level = "very_strong"
    elif stress_best_cpu_gpu_drop >= 5.0:
        stress_temp_status = "pass"
        stress_temp_level = "strong"
    elif stress_best_cpu_gpu_drop > 0.0:
        stress_temp_status = "pass"
        stress_temp_level = "normal"
    else:
        stress_temp_status = "fail"
        stress_temp_level = "none"
    add_criterion(
        name="stress_temperature_improved",
        status=stress_temp_status,
        level=stress_temp_level,
        threshold="Stress CPU/GPU temperature drops (strong >=5 C, very strong >=8 C)",
        observed=(
            f"stress CPU drop={stress_cpu_drop}, stress GPU drop={stress_gpu_drop}"
        ),
        impact="main_stress_condition",
    )

    if not has_stress_clock_data:
        stress_clock_status = "warning"
        stress_clock_level = "weak"
    elif stress_freq_gain >= 5.0:
        stress_clock_status = "pass"
        stress_clock_level = "strong"
    elif stress_freq_gain >= 3.0:
        stress_clock_status = "pass"
        stress_clock_level = "normal"
    else:
        stress_clock_status = "fail"
        stress_clock_level = "none"
    add_criterion(
        name="stress_clock_speed_improved",
        status=stress_clock_status,
        level=stress_clock_level,
        threshold="Stress clock gain >=3% (strong >=5%)",
        observed=(
            f"stress CPU clock gain={stress_freq_gain:.1f}%"
            if has_stress_clock_data
            else "stress clock data unavailable"
        ),
        impact="used_for_tradeoff",
    )

    stress_tradeoff_pass = (
        stress_worst_cpu_gpu_drop is not None
        and stress_worst_cpu_gpu_drop >= -3.0
        and stress_best_cpu_gpu_drop is not None
        and stress_best_cpu_gpu_drop <= 0.0
        and stress_freq_gain >= 3.0
    )
    stress_tradeoff_fail = (
        stress_best_cpu_gpu_drop is not None
        and stress_best_cpu_gpu_drop <= 0.0
        and stress_freq_gain < 3.0
    )
    if stress_tradeoff_pass:
        stress_tradeoff_status = "pass"
        stress_tradeoff_level = "strong" if stress_freq_gain >= 5.0 else "normal"
    elif stress_tradeoff_fail:
        stress_tradeoff_status = "fail"
        stress_tradeoff_level = "none"
    else:
        stress_tradeoff_status = "warning"
        stress_tradeoff_level = "weak"
    add_criterion(
        name="stress_performance_positive_tradeoff",
        status=stress_tradeoff_status,
        level=stress_tradeoff_level,
        threshold="Stress temperature same to +3 C hotter with >=3% stress clock gain",
        observed=(
            f"stress worst CPU/GPU delta={stress_worst_cpu_gpu_drop}, stress clock gain={stress_freq_gain:.1f}%"
        ),
        impact="alternative_stress_keep_condition",
    )

    stress_penalty_fail = (
        stress_worst_cpu_gpu_drop is not None
        and stress_worst_cpu_gpu_drop < -3.0
        and stress_freq_gain < 3.0
    )
    add_criterion(
        name="stress_no_unjustified_heat_penalty",
        status="fail" if stress_penalty_fail else "pass",
        level="none" if stress_penalty_fail else "normal",
        threshold="No stress CPU/GPU more than 3 C hotter without >=3% clock gain",
        observed=(
            f"stress worst CPU/GPU delta={stress_worst_cpu_gpu_drop}, stress clock gain={stress_freq_gain:.1f}%"
        ),
        impact="return_on_fail",
    )

    keep_requirements_met = idle_improved and (
        stress_temp_status == "pass" or stress_tradeoff_status == "pass"
    ) and not stress_penalty_fail and not idle_penalty_fail
    add_criterion(
        name="keep_requirements_met",
        status="pass" if keep_requirements_met else "fail",
        level="normal" if keep_requirements_met else "none",
        threshold="Idle improved AND (stress temp improved OR stress tradeoff passed)",
        observed=(
            f"idle_improved={idle_improved}, stress_temp_pass={stress_temp_status == 'pass'}, "
            f"stress_tradeoff_pass={stress_tradeoff_status == 'pass'}, penalties={idle_penalty_fail or stress_penalty_fail}"
        ),
        impact="enables_keep",
    )

    definitely_keep_requirements_met = (
        keep_requirements_met
        and (stress_best_cpu_gpu_drop is not None and stress_best_cpu_gpu_drop >= 5.0)
        and not stress_penalty_fail
    )
    definitely_keep_level = "strong"
    if stress_best_cpu_gpu_drop is not None and stress_best_cpu_gpu_drop >= 8.0:
        definitely_keep_level = "very_strong"
    add_criterion(
        name="definitely_keep_requirements_met",
        status="pass" if definitely_keep_requirements_met else "fail",
        level=definitely_keep_level if definitely_keep_requirements_met else "none",
        threshold="Keep requirements plus strong stress drop (>=5 C, very strong >=8 C)",
        observed=(
            f"stress_best_drop={stress_best_cpu_gpu_drop}, stress_clock_gain={stress_freq_gain:.1f}%"
        ),
        impact="enables_definitely_keep",
    )

    data_quality_failed = (
        not all_phases_complete
        or not has_comparable_cpu_gpu
        or minimum_samples_status == "fail"
    )
    tradeoff_needs_clock_but_missing = (
        stress_best_cpu_gpu_drop is not None
        and stress_best_cpu_gpu_drop <= 0.0
        and not has_stress_clock_data
    )
    stress_failed = (
        (stress_temp_status != "pass") and (stress_tradeoff_status != "pass")
    )
    marginal_result = (
        (idle_best_cpu_gpu_drop is None or idle_best_cpu_gpu_drop < 3.0)
        and (stress_best_cpu_gpu_drop is None or stress_best_cpu_gpu_drop < 3.0)
        and stress_freq_gain < 3.0
    )

    if data_quality_failed:
        verdict = "inconclusive"
        reason = "missing phase, sensor comparability, or minimum sample requirements"
        applied_rule_id = "inconclusive_data_quality_failure"
    elif tradeoff_needs_clock_but_missing:
        verdict = "inconclusive"
        reason = "stress tradeoff cannot be evaluated because stress clock data is unavailable"
        applied_rule_id = "inconclusive_missing_clock_data"
    elif idle_penalty_fail:
        verdict = "return"
        reason = "idle temperatures increased by more than 2 C"
        applied_rule_id = "return_idle_heat_penalty"
    elif not idle_improved:
        if stress_temp_status == "pass" or stress_tradeoff_status == "pass":
            verdict = "return"
            reason = "stress looked useful but idle did not improve"
            applied_rule_id = "return_idle_gate_failed"
        else:
            verdict = "return"
            reason = "idle temperatures did not improve and stress showed no useful benefit"
            applied_rule_id = "return_idle_not_improved"
    elif stress_penalty_fail:
        verdict = "return"
        reason = "stress temperatures were more than 3 C hotter without meaningful clock gain"
        applied_rule_id = "return_stress_heat_penalty"
    elif stress_failed:
        verdict = "return"
        reason = "stress temperatures did not improve and stress clocks did not improve"
        applied_rule_id = "return_stress_failed"
    elif marginal_result:
        verdict = "probably_return"
        reason = "idle and stress improvements were both marginal"
        applied_rule_id = "return_marginal_result"
    elif definitely_keep_requirements_met:
        verdict = "definitely_keep"
        reason = "idle improved and stress showed strong benefit without major penalty"
        applied_rule_id = "definitely_keep_requirements_met"
    elif keep_requirements_met:
        verdict = "keep"
        reason = "idle improved and stress showed practical cooling/performance benefit"
        applied_rule_id = "keep_requirements_met"
    else:
        verdict = "return"
        reason = "mixed results"
        applied_rule_id = "return_mixed_results"

    keep = verdict in ("keep", "definitely_keep")

    clock_tradeoff_note = None
    if any((v is not None and v < 0.0) for v in all_temp_drops) and best_freq_gain >= 3.0:
        clock_tradeoff_note = "Some phases ran hotter while CPU clocks increased."

    verdict_short = "KEEP" if verdict == "keep" else (
        "DEFINITELY KEEP" if verdict == "definitely_keep" else (
            "INCONCLUSIVE" if verdict == "inconclusive" else "RETURN"
        )
    )
    summary_short = (
        f"Verdict: {verdict_short}. "
        f"Best temp drop {best_temp_drop:.1f}C, best CPU clock gain {best_freq_gain:.1f}%."
    )

    return {
        "verdict": verdict,
        "keep_recommended": keep,
        "reason": reason,
        "applied_rule_id": applied_rule_id,
        "summary_short": summary_short,
        "clock_tradeoff_note": clock_tradeoff_note,
        "best_temp_drop_c": round(best_temp_drop, 3),
        "worst_temp_drop_c": round(worst_temp_drop, 3),
        "best_cpu_freq_gain_pct": round(best_freq_gain, 3),
        "phase_stats": phase_stats,
        "comparisons": comparisons,
        "rolling_window_points": rolling_window_points,
        "rolling_overtime_comparisons": rolling_overtime_comparisons,
        "sensor_temp_drop_summary": sensor_temp_drop_summary,
        "criteria_evaluation": criteria_evaluation,
    }


def build_process_allowlist() -> set[str]:
    return {
        "systemd",
        "dbus-daemon",
        "pipewire",
        "pulseaudio",
        "wireplumber",
        "Xorg",
        "Xwayland",
        "gnome-shell",
        "plasmashell",
        "kwin_wayland",
        "kwin_x11",
        "ksmserver",
        "sddm-helper",
        "lightdm",
        "gdm",
        "ssh-agent",
        "tmux",
        "bash",
        "zsh",
        "fish",
        "python",
        "python3",
        "code",
        "Code",
    }


def find_busy_user_processes(cpu_threshold: float = 3.0) -> List[Tuple[int, str, float]]:
    """Return non-whitelisted user processes consuming notable CPU."""
    uid = os.getuid()
    allow = build_process_allowlist()

    # Prime cpu_percent values.
    for proc in psutil.process_iter(["pid", "name", "uids"]):
        try:
            uids = proc.info.get("uids")
            if uids and uids.real == uid:
                proc.cpu_percent(None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    time.sleep(1.0)

    offenders: List[Tuple[int, str, float]] = []
    for proc in psutil.process_iter(["pid", "name", "uids"]):
        try:
            uids = proc.info.get("uids")
            if not uids or uids.real != uid:
                continue
            pid = int(proc.info.get("pid", -1))
            name = (proc.info.get("name") or "unknown").strip()
            cpu = float(proc.cpu_percent(None))
            if pid == os.getpid():
                continue
            if name in allow:
                continue
            if cpu >= cpu_threshold:
                offenders.append((pid, name, cpu))
        except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
            continue

    offenders.sort(key=lambda row: row[2], reverse=True)
    return offenders


def run_subprocess(command: List[str], name: str) -> RunningStressor:
    proc = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )
    return RunningStressor(name=name, proc=proc)


def start_python_cpu_stressor(worker_count: int) -> RunningStressor:
    code = (
        "import math, time\n"
        "x=0.0001\n"
        "while True:\n"
        "  x = math.sin(x) * math.cos(x) + 1.000001\n"
        "  if x > 2:\n"
        "    x = 0.0001\n"
    )
    # One process with many workers would be cleaner, but separate processes map cleanly to cores.
    cmd = [
        sys.executable,
        "-c",
        (
            "import subprocess,sys,os,time;"
            f"workers={worker_count};"
            "procs=[];"
            f"code={code!r};"
            "\nfor _ in range(workers):"
            " procs.append(subprocess.Popen([sys.executable,'-c',code]))"
            "\ntry:"
            "\n while True: time.sleep(1)"
            "\nexcept KeyboardInterrupt: pass"
        ),
    ]
    return run_subprocess(cmd, f"python_cpu_stress_{worker_count}w")


def start_stressors(cpu_workers: int, gpu_stress_cmd: Optional[str]) -> List[RunningStressor]:
    stressors: List[RunningStressor] = []
    if shutil.which("stress-ng"):
        cmd = ["stress-ng", "--cpu", str(cpu_workers), "--cpu-method", "matrixprod", "--metrics-brief"]
        stressors.append(run_subprocess(cmd, "stress-ng-cpu"))
    else:
        stressors.append(start_python_cpu_stressor(cpu_workers))

    # Use provided GPU stress command, or auto-generate one if GPU is available
    actual_gpu_cmd = gpu_stress_cmd
    if not actual_gpu_cmd:
        actual_gpu_cmd = get_gpu_stress_cmd()
    
    if actual_gpu_cmd:
        stressors.append(run_subprocess(["bash", "-lc", actual_gpu_cmd], "gpu-stress-auto"))
    
    return stressors


def stop_stressors(stressors: Iterable[RunningStressor]) -> None:
    for stressor in stressors:
        proc = stressor.proc
        if proc.poll() is not None:
            continue
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            continue
        except Exception:
            proc.terminate()

    deadline = time.time() + 5.0
    for stressor in stressors:
        proc = stressor.proc
        while proc.poll() is None and time.time() < deadline:
            time.sleep(0.1)
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                proc.kill()


def ask_user_checkpoint(prompt: str) -> None:
    print("\n" + "=" * 80)
    print(prompt)
    print("Press Enter to continue...")
    input()


def write_metadata(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def log_phase(
    csv_writer: csv.writer,
    phase: Phase,
    duration_sec: int,
    interval_sec: float,
    cpu_workers: int,
    gpu_stress_cmd: Optional[str],
) -> dict:
    print(f"\nStarting phase: {phase.name}")
    print(f"  cooled={phase.cooled} stress={phase.stress} duration={duration_sec}s")

    stressors: List[RunningStressor] = []
    if phase.stress:
        stressors = start_stressors(cpu_workers=cpu_workers, gpu_stress_cmd=gpu_stress_cmd)
        time.sleep(2.0)

    started = time.time()
    next_sample = started
    samples = 0
    sensors_seen: set[str] = set()

    try:
        while True:
            now = time.time()
            if now >= started + duration_sec:
                break
            if now < next_sample:
                time.sleep(min(0.5, next_sample - now))
                continue

            ts = utc_now_iso()
            readings = collect_temperatures()
            for sensor, value in readings.items():
                sensors_seen.add(sensor)
                csv_writer.writerow(
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
            next_sample += interval_sec

            if samples % 6 == 0:
                elapsed = int(now - started)
                print(f"  {phase.name}: {elapsed}s elapsed, samples={samples}, sensors={len(readings)}")
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


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CPU/GPU temperature comparison phases.")
    parser.add_argument("--output-dir", default="results", help="Directory for CSV/JSON output.")
    parser.add_argument("--duration-min", type=int, default=20, help="Minutes per phase (default: 20).")
    parser.add_argument("--interval-sec", type=float, default=1.0, help="Sampling interval in seconds.")
    parser.add_argument(
        "--cpu-workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Number of CPU workers used during stress phases.",
    )
    parser.add_argument(
        "--gpu-stress-cmd",
        default="",
        help="Optional shell command to run GPU stress during stress phases.",
    )
    parser.add_argument(
        "--enforce-clean",
        action="store_true",
        help="Abort if high-CPU user processes are detected before the run.",
    )
    parser.add_argument(
        "--cpu-threshold",
        type=float,
        default=3.0,
        help="CPU percent threshold for reporting potentially interfering processes.",
    )
    parser.add_argument(
        "--steady-timeout-sec",
        type=int,
        default=900,
        help="Maximum seconds to wait for steady-state after uncooled stress (test 2) (default: 900).",
    )
    return parser.parse_args()


def cli_main() -> int:
    args = parse_cli_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    experiment_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"temperature_log_{experiment_id}.csv"
    metadata_path = out_dir / f"run_metadata_{experiment_id}.json"

    # Detect and report GPU status
    print("=" * 80)
    print("GPU Detection:")
    gpu_info = get_gpu_info()
    print(f"  Status: {gpu_info['message']}")
    if gpu_info["available"]:
        if gpu_info["can_run_code"]:
            print(f"  Device: {gpu_info['device']}")
            print(f"  Capability: ✓ GPU can run code - mild stress will be included")
        else:
            print(f"  WARNING: GPU detected but cannot execute code")
            print(f"  GPU stress testing will be skipped")
    else:
        print("  No GPU will be stressed during test phases")
    print("=" * 80)

    offenders = find_busy_user_processes(cpu_threshold=args.cpu_threshold)
    if offenders:
        print("Potentially interfering processes detected (CPU-active):")
        for pid, name, cpu in offenders[:20]:
            print(f"  PID={pid:<7} CPU={cpu:>5.1f}%  {name}")
        if args.enforce_clean:
            print("Aborting due to --enforce-clean.")
            return 2
        print("Continuing anyway. Close them manually if you want cleaner results.")
    else:
        print("No high-CPU user processes detected.")

    phases = [
        Phase("uncooled_idle", cooled=False, stress=False),
        Phase("uncooled_stress", cooled=False, stress=True),
        Phase("cooled_idle", cooled=True, stress=False),
        Phase("cooled_stress", cooled=True, stress=True),
    ]

    ask_user_checkpoint("Set the system to UNCOOLED mode before phase 1.")

    phase_summaries: List[dict] = []
    duration_sec = args.duration_min * 60

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_utc", "phase", "cooling_state", "load_state", "sensor", "value"])

        for idx, phase in enumerate(phases):
            if idx == 2:
                ask_user_checkpoint("Attach external cooler now (COOLED phases).")

            summary = log_phase(
                csv_writer=writer,
                phase=phase,
                duration_sec=duration_sec,
                interval_sec=args.interval_sec,
                cpu_workers=args.cpu_workers,
                gpu_stress_cmd=(args.gpu_stress_cmd.strip() or None),
            )

            if idx == 1:
                summary["post_phase_steady_state"] = wait_until_temperatures_steady(
                    interval_sec=args.interval_sec,
                    timeout_sec=max(1, int(args.steady_timeout_sec)),
                )

            phase_summaries.append(summary)

    metadata = {
        "experiment_id": experiment_id,
        "started_utc": utc_now_iso(),
        "duration_per_phase_min": args.duration_min,
        "interval_sec": args.interval_sec,
        "steady_timeout_sec": max(1, int(args.steady_timeout_sec)),
        "cpu_workers": args.cpu_workers,
        "gpu_stress_cmd": args.gpu_stress_cmd,
        "gpu_info": gpu_info,
        "host": {
            "platform": sys.platform,
            "hostname": os.uname().nodename if hasattr(os, "uname") else "unknown",
            "cpu_count": os.cpu_count(),
        },
        "phases": phase_summaries,
        "files": {
            "csv": str(csv_path),
            "metadata": str(metadata_path),
        },
    }
    write_metadata(metadata_path, metadata)

    print("\nRun complete.")
    print(f"Temperature log: {csv_path}")
    print(f"Metadata: {metadata_path}")
    return 0



# ----------------------------- GUI runner -----------------------------

GUI_DEPS_AVAILABLE = False
GUI_IMPORT_ERROR: Optional[BaseException] = None

try:
    import threading

    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.colors import to_hex
    from matplotlib.figure import Figure
    from PyQt5.QtCore import QEvent, Qt, QTimer, pyqtSignal
    from PyQt5.QtWidgets import (
        QApplication,
        QCheckBox,
        QDoubleSpinBox,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QStackedWidget,
        QSplitter,
        QSpinBox,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )

    GUI_DEPS_AVAILABLE = True
except Exception as exc:
    GUI_IMPORT_ERROR = exc


if GUI_DEPS_AVAILABLE:
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
        cooler_assessment_ready = pyqtSignal(object)

        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("Cooler Verdict - Interactive Dashboard")
            self.setMinimumSize(1180, 740)
            self.resize(1520, 920)

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
            self.sensor_live_rows: Dict[str, QFrame] = {}
            self.sensor_live_checkboxes: Dict[str, QCheckBox] = {}
            self.sensor_live_swatches: Dict[str, QLabel] = {}
            self.sensor_value_chips: Dict[str, QLabel] = {}
            self.sensor_color_hex: Dict[str, str] = {}
            self.phase_cards: Dict[str, QFrame] = {}
            self.phase_card_labels: Dict[str, QLabel] = {}
            self.plot_mode = "raw"
            self.plot_rolling_window_points = 5
            self._hovered_sensor: Optional[str] = None
            self.current_cooler_assessment: Optional[dict] = None

            self.figure = Figure(figsize=(9.5, 4.8), dpi=100)
            self.ax = self.figure.add_subplot(111)
            self.ax.set_title("Live Temperature Sensors")
            self.ax.set_xlabel("Elapsed time (min)")
            self.ax.set_ylabel("Temperature (C)")
            self.ax.grid(True, alpha=0.35)
            self.canvas = FigureCanvas(self.figure)
            self.canvas.mpl_connect("pick_event", self._on_plot_pick)
            self.canvas.mpl_connect("motion_notify_event", self._on_plot_hover)
            self.hover_annotation = None

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
            self.status_changed.connect(self.header_status_label.setText)
            self.phase_changed.connect(self.phase_value_label.setText)
            self.phase_changed.connect(self.header_phase_label.setText)
            self.phase_changed.connect(self._update_phase_cards)
            self.remaining_changed.connect(self._apply_remaining)
            self.warning_changed.connect(self.warning_value_label.setText)
            self.output_changed.connect(self.output_value_label.setText)
            self.buttons_changed.connect(self._apply_buttons)
            self.sensor_sample_ready.connect(self._consume_sensor_sample)
            self.dialog_info_requested.connect(self._show_info_dialog)
            self.dialog_error_requested.connect(self._show_error_dialog)
            self.cooler_confirm_requested.connect(self._show_cooler_confirmation)
            self.cooler_assessment_ready.connect(self._apply_cooler_assessment)

        def _build_layout(self) -> None:
            """Build a three-pane benchmark dashboard.

            The GUI is organised around the task flow rather than the widget
            types: configure the run on the left, inspect the plot in the
            centre, and control/inspect individual sensors on the right.
            """
            self.setStyleSheet(
                """
                QMainWindow { background: #eef2f7; }
                QFrame#TopBar, QFrame#Card, QGroupBox {
                    background: #ffffff;
                    border: 1px solid #d7dde8;
                    border-radius: 12px;
                }
                QGroupBox {
                    margin-top: 12px;
                    font-weight: 600;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 12px;
                    padding: 0 6px;
                    color: #1f2937;
                }
                QLabel#HeroTitle {
                    font-size: 22px;
                    font-weight: 800;
                    color: #111827;
                }
                QLabel#Muted { color: #64748b; }
                QLabel#SmallMuted { color: #64748b; font-size: 11px; }
                QLabel#MetricLabel { color: #64748b; font-size: 11px; font-weight: 600; }
                QLabel#StatusPill {
                    background: #e0f2fe;
                    color: #075985;
                    border: 1px solid #bae6fd;
                    border-radius: 12px;
                    padding: 5px 10px;
                    font-weight: 700;
                }
                QLabel#TimePill {
                    background: #ecfdf5;
                    color: #047857;
                    border: 1px solid #bbf7d0;
                    border-radius: 12px;
                    padding: 5px 10px;
                    font-weight: 800;
                    font-size: 16px;
                }
                QPushButton {
                    border: 1px solid #cbd5e1;
                    border-radius: 8px;
                    padding: 8px 12px;
                    background: #ffffff;
                    color: #111827;
                    font-weight: 600;
                }
                QPushButton:hover:!disabled { background: #f8fafc; border-color: #94a3b8; }
                QPushButton:pressed:!disabled { background: #e2e8f0; }
                QPushButton:disabled {
                    color: #94a3b8;
                    background: #e5e7eb;
                    border-color: #cbd5e1;
                }
                QPushButton#PrimaryButton {
                    background: #2563eb;
                    color: #ffffff;
                    border-color: #1d4ed8;
                    font-weight: 800;
                    padding: 10px 16px;
                }
                QPushButton#PrimaryButton:hover:!disabled {
                    background: #1d4ed8;
                    border-color: #1e40af;
                }
                QPushButton#PrimaryButton:pressed:!disabled { background: #1e3a8a; }
                QPushButton#PrimaryButton:disabled {
                    background: #e5e7eb;
                    color: #94a3b8;
                    border-color: #cbd5e1;
                }
                QPushButton#DangerButton {
                    background: #dc2626;
                    color: #ffffff;
                    border-color: #b91c1c;
                    font-weight: 800;
                }
                QPushButton#DangerButton:hover:!disabled {
                    background: #b91c1c;
                    border-color: #991b1b;
                }
                QPushButton#DangerButton:pressed:!disabled { background: #7f1d1d; }
                QPushButton#DangerButton:disabled {
                    background: #e5e7eb;
                    color: #94a3b8;
                    border-color: #cbd5e1;
                }
                QLineEdit, QSpinBox, QDoubleSpinBox {
                    padding: 5px;
                    border: 1px solid #cbd5e1;
                    border-radius: 6px;
                    background: #ffffff;
                }
                QSplitter::handle { background: #cbd5e1; }
                QSplitter::handle:horizontal { width: 5px; }
                QScrollArea { background: transparent; }
                QFrame#PhaseCard {
                    background: #f8fafc;
                    border: 1px solid #e2e8f0;
                    border-radius: 10px;
                }
                QFrame#SensorRow {
                    background: #ffffff;
                    border: 1px solid #e2e8f0;
                    border-radius: 8px;
                }
                QFrame#SummaryOverlay {
                    background: rgba(255, 255, 255, 238);
                    border: 1px solid #bfdbfe;
                    border-radius: 10px;
                }
                """
            )

            root = QWidget()
            self.setCentralWidget(root)
            root_layout = QVBoxLayout(root)
            root_layout.setContentsMargins(12, 12, 12, 12)
            root_layout.setSpacing(10)

            # Top application bar: current state is always visible.
            top_bar = QFrame()
            top_bar.setObjectName("TopBar")
            top_layout = QHBoxLayout(top_bar)
            top_layout.setContentsMargins(16, 12, 16, 12)
            top_layout.setSpacing(12)

            title_col = QVBoxLayout()
            title_col.setSpacing(2)
            title = QLabel("Cooler Verdict")
            title.setObjectName("HeroTitle")
            subtitle = QLabel("Interactive dashboard: configure the test, monitor live sensor plots, hover lines for sensor identity, and get a keep/return verdict.")
            subtitle.setObjectName("Muted")
            subtitle.setWordWrap(True)
            title_col.addWidget(title)
            title_col.addWidget(subtitle)
            top_layout.addLayout(title_col, stretch=1)

            status_col = QVBoxLayout()
            status_col.setSpacing(4)
            status_label = QLabel("State")
            status_label.setObjectName("MetricLabel")
            self.header_status_label = QLabel("Ready")
            self.header_status_label.setObjectName("StatusPill")
            status_col.addWidget(status_label)
            status_col.addWidget(self.header_status_label)
            top_layout.addLayout(status_col)

            phase_col = QVBoxLayout()
            phase_col.setSpacing(4)
            phase_label = QLabel("Phase")
            phase_label.setObjectName("MetricLabel")
            self.header_phase_label = QLabel("No test running")
            self.header_phase_label.setObjectName("StatusPill")
            phase_col.addWidget(phase_label)
            phase_col.addWidget(self.header_phase_label)
            top_layout.addLayout(phase_col)

            time_col = QVBoxLayout()
            time_col.setSpacing(4)
            time_label = QLabel("Remaining")
            time_label.setObjectName("MetricLabel")
            self.header_time_label = QLabel("00:00")
            self.header_time_label.setObjectName("TimePill")
            time_col.addWidget(time_label)
            time_col.addWidget(self.header_time_label)
            top_layout.addLayout(time_col)

            root_layout.addWidget(top_bar)

            main_splitter = QSplitter(Qt.Horizontal)
            main_splitter.setChildrenCollapsible(False)
            root_layout.addWidget(main_splitter, stretch=1)

            # Left pane: setup + workflow + run status.
            left_scroll = QScrollArea()
            left_scroll.setWidgetResizable(True)
            left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            left_scroll.setFrameShape(QFrame.NoFrame)
            left_scroll.setMinimumWidth(300)
            left_scroll.setMaximumWidth(430)
            left = QWidget()
            left_scroll.setWidget(left)
            left_layout = QVBoxLayout(left)
            left_layout.setContentsMargins(2, 2, 8, 2)
            left_layout.setSpacing(10)

            actions_box = QGroupBox("Run control")
            actions_layout = QVBoxLayout(actions_box)
            actions_layout.setContentsMargins(12, 20, 12, 12)
            actions_layout.setSpacing(10)
            actions_hint = QLabel("Start with the cooler OFF. The app will prompt you when it is time to turn it ON.")
            actions_hint.setObjectName("Muted")
            actions_hint.setWordWrap(True)
            actions_layout.addWidget(actions_hint)
            button_row = QHBoxLayout()
            button_row.setSpacing(8)
            self.start_button = QPushButton("Start benchmark")
            self.start_button.setObjectName("PrimaryButton")
            self.stop_button = QPushButton("Stop")
            self.stop_button.setObjectName("DangerButton")
            self.check_button = QPushButton("Check processes")
            for button in (self.start_button, self.stop_button, self.check_button):
                button.setCursor(Qt.PointingHandCursor)
                button.setMinimumHeight(40)
            self.stop_button.setEnabled(False)
            button_row.addWidget(self.start_button, stretch=2)
            button_row.addWidget(self.stop_button, stretch=1)
            actions_layout.addLayout(button_row)
            actions_layout.addWidget(self.check_button)
            left_layout.addWidget(actions_box)

            workflow_box = QGroupBox("Workflow")
            workflow_layout = QVBoxLayout(workflow_box)
            workflow_layout.setContentsMargins(12, 20, 12, 12)
            workflow_layout.setSpacing(8)
            self.phase_cards = {}
            self.phase_card_labels = {}
            for phase_name, title_text, detail in [
                ("uncooled_idle", "1  Uncooled idle", "Baseline temperatures with no external cooler."),
                ("uncooled_stress", "2  Uncooled stress", "Baseline under CPU/GPU load."),
                ("cooled_idle", "3  Cooled idle", "Cooler ON, same idle conditions."),
                ("cooled_stress", "4  Cooled stress", "Cooler ON, same load conditions."),
            ]:
                card = QFrame()
                card.setObjectName("PhaseCard")
                card_layout = QVBoxLayout(card)
                card_layout.setContentsMargins(10, 8, 10, 8)
                card_layout.setSpacing(2)
                label = QLabel(title_text)
                label.setStyleSheet("font-weight: 800; color: #1f2937;")
                desc = QLabel(detail)
                desc.setObjectName("SmallMuted")
                desc.setWordWrap(True)
                card_layout.addWidget(label)
                card_layout.addWidget(desc)
                workflow_layout.addWidget(card)
                self.phase_cards[phase_name] = card
                self.phase_card_labels[phase_name] = label

                if phase_name == "uncooled_stress":
                    steady_card = QFrame()
                    steady_card.setObjectName("PhaseCard")
                    steady_layout = QVBoxLayout(steady_card)
                    steady_layout.setContentsMargins(10, 8, 10, 8)
                    steady_layout.setSpacing(2)
                    steady_label = QLabel("Steady-state check (after test 2)")
                    steady_label.setStyleSheet("font-weight: 700; color: #1f2937;")
                    steady_desc = QLabel("Wait until temperatures stabilize before starting cooled phases.")
                    steady_desc.setObjectName("SmallMuted")
                    steady_desc.setWordWrap(True)
                    steady_layout.addWidget(steady_label)
                    steady_layout.addWidget(steady_desc)
                    workflow_layout.addWidget(steady_card)
            left_layout.addWidget(workflow_box)

            setup_box = QGroupBox("Test settings")
            setup_form = QFormLayout(setup_box)
            setup_form.setContentsMargins(12, 20, 12, 12)
            setup_form.setSpacing(8)
            setup_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

            self.duration_spin = QSpinBox()
            self.duration_spin.setRange(1, 1440)
            self.duration_spin.setValue(20)
            self.duration_spin.setSuffix(" min")

            self.interval_spin = QDoubleSpinBox()
            self.interval_spin.setRange(1.0, 3600.0)
            self.interval_spin.setDecimals(1)
            self.interval_spin.setSingleStep(0.5)
            self.interval_spin.setValue(1.0)
            self.interval_spin.setSuffix(" sec")

            self.cpu_workers_spin = QSpinBox()
            self.cpu_workers_spin.setRange(1, max(1, os.cpu_count() or 1))
            self.cpu_workers_spin.setValue(max(1, (os.cpu_count() or 2) - 1))

            self.cpu_threshold_spin = QDoubleSpinBox()
            self.cpu_threshold_spin.setRange(0.1, 100.0)
            self.cpu_threshold_spin.setDecimals(1)
            self.cpu_threshold_spin.setSingleStep(0.5)
            self.cpu_threshold_spin.setValue(3.0)
            self.cpu_threshold_spin.setSuffix(" %")

            self.steady_timeout_spin = QSpinBox()
            self.steady_timeout_spin.setRange(10, 36000)
            self.steady_timeout_spin.setValue(900)
            self.steady_timeout_spin.setSuffix(" sec")

            self.gpu_stress_cmd_edit = QLineEdit()
            self.gpu_stress_cmd_edit.setPlaceholderText("Optional shell command")
            self.enforce_clean_checkbox = QCheckBox("Abort if busy user processes are active")

            setup_form.addRow("Phase duration", self.duration_spin)
            setup_form.addRow("Sample interval", self.interval_spin)
            setup_form.addRow("CPU workers", self.cpu_workers_spin)
            setup_form.addRow("CPU threshold", self.cpu_threshold_spin)
            setup_form.addRow("Steady timeout", self.steady_timeout_spin)
            setup_form.addRow("GPU stress", self.gpu_stress_cmd_edit)
            setup_form.addRow("Clean guard", self.enforce_clean_checkbox)
            left_layout.addWidget(setup_box)

            status_box = QGroupBox("Run details")
            status_form = QFormLayout(status_box)
            status_form.setContentsMargins(12, 20, 12, 12)
            status_form.setSpacing(8)
            self.status_value_label = QLabel("Ready")
            self.phase_value_label = QLabel("No test running")
            self.remaining_value_label = QLabel("00:00")
            self.output_value_label = QLabel("results/")
            self.warning_value_label = QLabel("")
            self.warning_value_label.setStyleSheet("color: #b91c1c; font-weight: 700;")
            for label in (self.status_value_label, self.phase_value_label, self.remaining_value_label, self.output_value_label, self.warning_value_label):
                label.setWordWrap(True)
                label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            status_form.addRow("State", self.status_value_label)
            status_form.addRow("Phase", self.phase_value_label)
            status_form.addRow("Remaining", self.remaining_value_label)
            status_form.addRow("Output", self.output_value_label)
            status_form.addRow("Warnings", self.warning_value_label)
            left_layout.addWidget(status_box)

            hardware_box = QGroupBox("Hardware")
            hardware_form = QFormLayout(hardware_box)
            hardware_form.setContentsMargins(12, 20, 12, 12)
            hardware_form.setSpacing(8)
            self.cpu_spec_label = QLabel("Detecting CPU...")
            self.gpu_spec_label = QLabel("Detecting GPU...")
            self.storage_spec_label = QLabel("Detecting storage...")
            for label in (self.cpu_spec_label, self.gpu_spec_label, self.storage_spec_label):
                label.setWordWrap(True)
                label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            hardware_form.addRow("CPU", self.cpu_spec_label)
            hardware_form.addRow("GPU", self.gpu_spec_label)
            hardware_form.addRow("Storage", self.storage_spec_label)
            left_layout.addWidget(hardware_box)
            left_layout.addStretch(1)
            main_splitter.addWidget(left_scroll)

            # Centre pane: plot-first workspace.
            centre = QWidget()
            centre_layout = QVBoxLayout(centre)
            centre_layout.setContentsMargins(10, 0, 10, 0)
            centre_layout.setSpacing(10)

            plot_card = QFrame()
            plot_card.setObjectName("Card")
            plot_layout = QVBoxLayout(plot_card)
            plot_layout.setContentsMargins(12, 12, 12, 12)
            plot_layout.setSpacing(8)
            plot_header = QHBoxLayout()
            plot_title_col = QVBoxLayout()
            plot_title_col.setSpacing(1)
            plot_title = QLabel("Interactive temperature timeline")
            plot_title.setStyleSheet("font-size: 16px; font-weight: 800; color: #111827;")
            plot_subtitle = QLabel("Hover a line to identify the sensor. Use the right panel to show or hide sensors.")
            plot_subtitle.setObjectName("Muted")
            plot_title_col.addWidget(plot_title)
            plot_title_col.addWidget(plot_subtitle)
            plot_header.addLayout(plot_title_col, stretch=1)
            self.plot_mode_button = QPushButton("Mode: Raw")
            self.plot_mode_button.setCheckable(True)
            self.plot_mode_button.setToolTip("Switch between raw and rolling-average temperature plotting.")
            self.summary_toggle_button = QPushButton("Show summary")
            self.summary_toggle_button.setCheckable(True)
            self.summary_toggle_button.setEnabled(False)
            self.show_all_button = QPushButton("Show all")
            self.hide_all_button = QPushButton("Hide all")
            plot_header.addWidget(self.plot_mode_button)
            plot_header.addWidget(self.summary_toggle_button)
            plot_header.addWidget(self.show_all_button)
            plot_header.addWidget(self.hide_all_button)
            plot_layout.addLayout(plot_header)

            self.center_stack = QStackedWidget()

            timeline_page = QWidget()
            timeline_layout = QVBoxLayout(timeline_page)
            timeline_layout.setContentsMargins(0, 0, 0, 0)
            timeline_layout.setSpacing(8)
            self.toolbar = NavigationToolbar(self.canvas, self)
            timeline_layout.addWidget(self.toolbar)
            self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.canvas.setMinimumHeight(520)
            timeline_layout.addWidget(self.canvas, stretch=1)
            self.hover_value_label = QLabel("Hover over a plotted line to see the sensor name and value.")
            self.hover_value_label.setObjectName("Muted")
            self.hover_value_label.setWordWrap(True)
            timeline_layout.addWidget(self.hover_value_label)

            summary_page = QWidget()
            summary_page_layout = QVBoxLayout(summary_page)
            summary_page_layout.setContentsMargins(0, 0, 0, 0)
            summary_page_layout.setSpacing(8)
            summary_title = QLabel("Results summary")
            summary_title.setStyleSheet("font-size: 16px; font-weight: 800; color: #111827;")
            summary_page_layout.addWidget(summary_title)

            summary_scroll = QScrollArea()
            summary_scroll.setWidgetResizable(True)
            summary_scroll.setFrameShape(QFrame.NoFrame)
            summary_content = QWidget()
            summary_scroll.setWidget(summary_content)
            summary_content_layout = QVBoxLayout(summary_content)
            summary_content_layout.setContentsMargins(0, 0, 0, 0)
            summary_content_layout.setSpacing(8)

            self.summary_verdict_label = QLabel("Verdict pending")
            self.summary_verdict_label.setStyleSheet(
                "font-weight: 800; padding: 3px 8px; border-radius: 8px; "
                "background: #e5e7eb; color: #334155;"
            )
            self.summary_reason_label = QLabel("Run all phases to generate a verdict.")
            self.summary_reason_label.setWordWrap(True)
            self.summary_decision_label = QLabel("")
            self.summary_decision_label.setWordWrap(True)
            self.summary_metrics_label = QLabel("")
            self.summary_metrics_label.setWordWrap(True)
            self.summary_criteria_label = QLabel("")
            self.summary_criteria_label.setWordWrap(True)
            self.summary_criteria_label.setTextFormat(Qt.RichText)
            self.summary_criteria_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

            summary_content_layout.addWidget(self.summary_verdict_label)
            summary_content_layout.addWidget(self.summary_reason_label)
            summary_content_layout.addWidget(self.summary_decision_label)
            summary_content_layout.addWidget(self.summary_metrics_label)
            summary_content_layout.addWidget(self.summary_criteria_label)
            summary_content_layout.addStretch(1)
            summary_page_layout.addWidget(summary_scroll, stretch=1)

            self.center_stack.addWidget(timeline_page)
            self.center_stack.addWidget(summary_page)
            self.center_stack.setCurrentIndex(0)

            plot_layout.addWidget(self.center_stack, stretch=1)
            centre_layout.addWidget(plot_card, stretch=1)
            main_splitter.addWidget(centre)

            # Right pane: live values double as plot visibility controls.
            right_scroll = QScrollArea()
            right_scroll.setWidgetResizable(True)
            right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            right_scroll.setFrameShape(QFrame.NoFrame)
            right_scroll.setMinimumWidth(300)
            right_scroll.setMaximumWidth(460)
            right = QWidget()
            right_scroll.setWidget(right)
            right_layout = QVBoxLayout(right)
            right_layout.setContentsMargins(8, 2, 2, 2)
            right_layout.setSpacing(10)

            live_box = QGroupBox("Live values + plot visibility")
            live_layout = QVBoxLayout(live_box)
            live_layout.setContentsMargins(12, 20, 12, 12)
            live_layout.setSpacing(8)
            live_hint = QLabel("Each row shows the latest reading. Tick or untick a sensor to show or hide its plot line.")
            live_hint.setObjectName("Muted")
            live_hint.setWordWrap(True)
            live_layout.addWidget(live_hint)
            self.sensor_columns_widget = QWidget()
            self.sensor_columns_layout = QGridLayout(self.sensor_columns_widget)
            self.sensor_columns_layout.setContentsMargins(0, 0, 0, 0)
            self.sensor_columns_layout.setSpacing(8)
            live_layout.addWidget(self.sensor_columns_widget)
            right_layout.addWidget(live_box, stretch=1)
            right_layout.addStretch(1)
            main_splitter.addWidget(right_scroll)

            main_splitter.setSizes([340, 880, 340])
            main_splitter.setStretchFactor(0, 0)
            main_splitter.setStretchFactor(1, 1)
            main_splitter.setStretchFactor(2, 0)

            self.start_button.clicked.connect(self.start_testing)
            self.stop_button.clicked.connect(self.stop_testing)
            self.check_button.clicked.connect(self.check_running_processes)
            self.plot_mode_button.toggled.connect(self._toggle_plot_mode)
            self.summary_toggle_button.toggled.connect(self._toggle_summary_overlay)
            self.show_all_button.clicked.connect(lambda: self._set_all_plot_visibility(True))
            self.hide_all_button.clicked.connect(lambda: self._set_all_plot_visibility(False))

        def _start_live_polling(self) -> None:
            self._poll_active = True
            self._poll_live_temps()
            self.live_poll_timer.start()

        def _toggle_summary_overlay(self, checked: bool) -> None:
            self._set_summary_overlay_visible(checked)

        def _toggle_plot_mode(self, rolling_enabled: bool) -> None:
            self.plot_mode = "rolling" if rolling_enabled else "raw"
            self.plot_mode_button.setText("Mode: Rolling avg" if rolling_enabled else "Mode: Raw")
            self._refresh_plot()

        def _set_summary_overlay_visible(self, visible: bool) -> None:
            can_show = self.current_cooler_assessment is not None
            show = bool(visible and can_show)
            self.center_stack.setCurrentIndex(1 if show else 0)
            self.summary_toggle_button.blockSignals(True)
            self.summary_toggle_button.setChecked(show)
            self.summary_toggle_button.setText("Show timeline" if show else "Show summary")
            self.summary_toggle_button.blockSignals(False)
            self.plot_mode_button.setEnabled(not show)
            self.show_all_button.setEnabled(not show)
            self.hide_all_button.setEnabled(not show)

        def _reset_summary_panel(self) -> None:
            self.current_cooler_assessment = None
            self.summary_verdict_label.setText("Verdict pending")
            self.summary_verdict_label.setStyleSheet(
                "font-weight: 800; padding: 3px 8px; border-radius: 8px; "
                "background: #e5e7eb; color: #334155;"
            )
            self.summary_reason_label.setText("Run all phases to generate a verdict.")
            self.summary_decision_label.setText("")
            self.summary_metrics_label.setText("")
            self.summary_criteria_label.setText("")
            self.summary_toggle_button.setEnabled(False)
            self._set_summary_overlay_visible(False)

        def _apply_cooler_assessment(self, assessment_obj: object) -> None:
            if not isinstance(assessment_obj, dict):
                return

            self.current_cooler_assessment = assessment_obj
            keep = bool(assessment_obj.get("keep_recommended"))
            verdict_text = "KEEP" if keep else "RETURN"
            verdict_bg = "#dcfce7" if keep else "#fee2e2"
            verdict_fg = "#166534" if keep else "#991b1b"
            self.summary_verdict_label.setText(f"Verdict: {verdict_text}")
            self.summary_verdict_label.setStyleSheet(
                "font-weight: 800; padding: 3px 8px; border-radius: 8px; "
                f"background: {verdict_bg}; color: {verdict_fg};"
            )

            reason = str(assessment_obj.get("reason") or "No reason provided")
            best_drop = float(assessment_obj.get("best_temp_drop_c") or 0.0)
            worst_drop = float(assessment_obj.get("worst_temp_drop_c") or 0.0)
            best_freq = float(assessment_obj.get("best_cpu_freq_gain_pct") or 0.0)
            self.summary_reason_label.setText(
                "Decision criteria use CPU/GPU temperature behavior plus CPU speed. "
                "The sensor list below includes all detected temperature sensors."
            )
            self.summary_decision_label.setText(f"Decision: {reason}.")
            self.summary_metrics_label.setText(
                f"Cooling change: up to {best_drop:.1f} C cooler. "
                f"Worst downside: up to {abs(min(0.0, worst_drop)):.1f} C hotter. "
                f"CPU speed change: up to {best_freq:.1f}% higher."
            )

            criteria = assessment_obj.get("criteria_evaluation") or []
            criteria_map: Dict[str, str] = {}
            for item in criteria:
                if not isinstance(item, dict):
                    continue
                item_name = str(item.get("name") or "")
                criteria_map[item_name] = str(item.get("status") or "")

            reject_guard_hit = criteria_map.get("stress_no_unjustified_heat_penalty") == "fail"
            idle_improved_check = criteria_map.get("idle_temperature_improved") == "pass"

            checks = [
                ("Test completed", criteria_map.get("data_complete") == "pass"),
                (
                    "Enough sensor data",
                    criteria_map.get("comparable_cpu_or_gpu_sensors_available") == "pass",
                ),
                ("Idle temperatures improved", idle_improved_check),
                ("No severe stress heat penalty", not reject_guard_hit),
                ("Stress cooling improved", criteria_map.get("stress_temperature_improved") == "pass"),
                ("Stress clock improved", criteria_map.get("stress_clock_speed_improved") == "pass"),
                (
                    "Overall improvement meaningful",
                    best_drop >= 5.0 or best_freq >= 5.0,
                ),
                ("Keep requirements met", criteria_map.get("keep_requirements_met") == "pass"),
            ]

            passed_checks = [label for label, passed in checks if passed]
            failed_checks = [label for label, passed in checks if not passed]

            passed_html = "".join(
                f"<div style='margin: 3px 0; color:#166534;'>Pass: {label}</div>" for label in passed_checks
            )
            failed_html = "".join(
                f"<div style='margin: 3px 0; color:#991b1b;'>Fail: {label}</div>" for label in failed_checks
            )

            if not passed_html:
                passed_html = "<div style='margin: 3px 0; color:#64748b;'>No passing checks.</div>"
            if not failed_html:
                failed_html = "<div style='margin: 3px 0; color:#64748b;'>No failed checks.</div>"

            reject_reason_text = "None" if keep else reason

            sensor_rows = assessment_obj.get("sensor_temp_drop_summary") or []
            sensor_table_rows: List[str] = []
            for row in sensor_rows:
                if not isinstance(row, dict):
                    continue
                sensor_key = str(row.get("sensor") or "")
                if not sensor_key:
                    continue
                idle_pct = row.get("idle_drop_pct")
                stress_pct = row.get("stress_drop_pct")

                def _fmt_pct(value: object) -> str:
                    if value is None:
                        return "n/a"
                    try:
                        return f"{float(value):+.1f}%"
                    except (TypeError, ValueError):
                        return "n/a"

                sensor_name = self._format_sensor_label(sensor_key)
                sensor_table_rows.append(
                    "<tr>"
                    f"<td style='padding:2px 6px; color:#334155;'>{sensor_name}</td>"
                    f"<td style='padding:2px 6px; color:#334155;'>{_fmt_pct(idle_pct)}</td>"
                    f"<td style='padding:2px 6px; color:#334155;'>{_fmt_pct(stress_pct)}</td>"
                    "</tr>"
                )

            if sensor_table_rows:
                sensor_section_html = (
                    "<table style='border-collapse:collapse; width:100%;'>"
                    "<thead><tr>"
                    "<th style='text-align:left; padding:3px 6px; color:#0f172a;'>Sensor</th>"
                    "<th style='text-align:left; padding:3px 6px; color:#0f172a;'>Idle</th>"
                    "<th style='text-align:left; padding:3px 6px; color:#0f172a;'>Stress</th>"
                    "</tr></thead>"
                    "<tbody>"
                    + "".join(sensor_table_rows)
                    + "</tbody></table>"
                )
            else:
                sensor_section_html = "<div style='margin: 2px 0; color:#64748b;'>No sensor drop data available.</div>"

            criteria_table_rows: List[str] = []
            for item in criteria:
                if not isinstance(item, dict):
                    continue
                name_text = html.escape(str(item.get("name") or ""))
                status_text = html.escape(str(item.get("status") or ""))
                level_text = html.escape(str(item.get("level") or ""))
                threshold_text = html.escape(str(item.get("threshold") or ""))
                observed_text = html.escape(str(item.get("observed") or ""))
                impact_text = html.escape(str(item.get("impact") or ""))
                if not name_text:
                    continue
                criteria_table_rows.append(
                    "<tr>"
                    f"<td style='padding:2px 6px; color:#334155;'>{name_text}</td>"
                    f"<td style='padding:2px 6px; color:#334155;'>{status_text}</td>"
                    f"<td style='padding:2px 6px; color:#334155;'>{level_text}</td>"
                    f"<td style='padding:2px 6px; color:#334155;'>{threshold_text}</td>"
                    f"<td style='padding:2px 6px; color:#334155;'>{observed_text}</td>"
                    f"<td style='padding:2px 6px; color:#334155;'>{impact_text}</td>"
                    "</tr>"
                )

            if criteria_table_rows:
                criteria_table_html = (
                    "<table style='border-collapse:collapse; width:100%;'>"
                    "<thead><tr>"
                    "<th style='text-align:left; padding:3px 6px; color:#0f172a;'>Name</th>"
                    "<th style='text-align:left; padding:3px 6px; color:#0f172a;'>Status</th>"
                    "<th style='text-align:left; padding:3px 6px; color:#0f172a;'>Level</th>"
                    "<th style='text-align:left; padding:3px 6px; color:#0f172a;'>Threshold</th>"
                    "<th style='text-align:left; padding:3px 6px; color:#0f172a;'>Observed</th>"
                    "<th style='text-align:left; padding:3px 6px; color:#0f172a;'>Impact</th>"
                    "</tr></thead>"
                    "<tbody>"
                    + "".join(criteria_table_rows)
                    + "</tbody></table>"
                )
            else:
                criteria_table_html = "<div style='margin: 2px 0; color:#64748b;'>No criteria metadata available.</div>"

            self.summary_criteria_label.setText(
                "<b>What passed</b><br/>"
                + passed_html
                + "<br/><b>What failed</b><br/>"
                + failed_html
                + "<br/><b>Reject reason</b><br/>"
                + f"<div style='margin: 3px 0; color:#334155;'>{reject_reason_text}.</div>"
                + "<br/><b>Temperature drop by sensor (%)</b><br/>"
                + sensor_section_html
                + "<br/><b>Criteria table</b><br/>"
                + criteria_table_html
            )

            self.summary_toggle_button.setEnabled(True)
            self._set_summary_overlay_visible(False)

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

        def _on_steady_progress(self, elapsed_sec: int, remaining_sec: int, timeout_sec: int) -> None:
            self._set_phase(f"steady_state_check ({elapsed_sec}s/{timeout_sec}s)")
            self._set_remaining_seconds(remaining_sec)

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
            text = f"{minutes:02d}:{remainder:02d}"
            self.remaining_value_label.setText(text)
            self.header_time_label.setText(text)

        def _apply_buttons(self, start_enabled: bool, stop_enabled: bool) -> None:
            self.start_button.setEnabled(start_enabled)
            self.stop_button.setEnabled(stop_enabled)
            # Disabled buttons keep a normal arrow cursor so they do not appear clickable.
            self.start_button.setCursor(Qt.PointingHandCursor if start_enabled else Qt.ArrowCursor)
            self.stop_button.setCursor(Qt.PointingHandCursor if stop_enabled else Qt.ArrowCursor)

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
            self.ax.set_facecolor("#fbfdff")
            self.ax.set_title("")
            self.ax.set_xlabel("Elapsed time (min)")
            self.ax.set_ylabel("Temperature (°C)")
            self.ax.grid(True, alpha=0.28, linestyle="--", linewidth=0.8)
            self.ax.spines["top"].set_visible(False)
            self.ax.spines["right"].set_visible(False)
            self.plot_lines = {}
            self.legend_artist_to_sensor = {}

            plotted: List[str] = []
            plotted_labels: List[str] = []
            visible_values: List[float] = []
            for sensor in sorted(self.sensor_time_data):
                xs = [value / 60.0 for value in self.sensor_time_data.get(sensor, [])]
                ys = self.sensor_data.get(sensor, [])
                if not xs or not ys:
                    continue

                if self.plot_mode == "rolling":
                    ys_plot = _rolling_mean([float(v) for v in ys], self.plot_rolling_window_points)
                else:
                    ys_plot = [float(v) for v in ys]

                label = self.sensor_display_names.get(sensor, self._format_sensor_label(sensor))
                color = self.sensor_color_hex.get(sensor)
                if color:
                    (line,) = self.ax.plot(xs, ys_plot, label=label, linewidth=2.0, color=color, solid_capstyle="round")
                else:
                    (line,) = self.ax.plot(xs, ys_plot, label=label, linewidth=2.0, solid_capstyle="round")
                    try:
                        self.sensor_color_hex[sensor] = to_hex(line.get_color())
                    except Exception:
                        self.sensor_color_hex[sensor] = "#2563eb"
                line.set_picker(7)
                visible = self.plot_visibility.get(sensor, True)
                line.set_visible(visible)
                if visible:
                    visible_values.extend([float(v) for v in ys_plot if v is not None])
                self.plot_lines[sensor] = line
                self.legend_artist_to_sensor[line] = sensor
                plotted.append(sensor)
                plotted_labels.append(label)

            if visible_values:
                ymin = min(visible_values)
                ymax = max(visible_values)
                pad = max(2.0, (ymax - ymin) * 0.12)
                self.ax.set_ylim(ymin - pad, ymax + pad)

            # Recreate hover annotation after clearing the axes.
            self.hover_annotation = self.ax.annotate(
                "",
                xy=(0, 0),
                xytext=(14, 14),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.35", fc="#111827", ec="#111827", alpha=0.92),
                color="white",
                fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#111827", alpha=0.8),
            )
            self.hover_annotation.set_visible(False)

            try:
                for old_leg in list(self.figure.legends):
                    old_leg.remove()
            except Exception:
                pass

            self._sync_live_value_rows_with_plot()
            self.figure.tight_layout(pad=1.6)
            self.canvas.draw_idle()

        def _style_live_sensor_row(self, sensor: str) -> None:
            row = self.sensor_live_rows.get(sensor)
            swatch = self.sensor_live_swatches.get(sensor)
            checkbox = self.sensor_live_checkboxes.get(sensor)
            if row is None:
                return
            color = self.sensor_color_hex.get(sensor, "#94a3b8")
            visible = self.plot_visibility.get(sensor, True)
            row.setStyleSheet(
                f"QFrame#SensorValueRow {{ background: {'#ffffff' if visible else '#f1f5f9'}; "
                f"border: 1px solid {'#cbd5e1' if visible else '#e2e8f0'}; "
                f"border-left: 7px solid {color}; border-radius: 8px; }}"
            )
            if swatch is not None:
                swatch.setStyleSheet(f"background: {color}; border-radius: 5px;")
            if checkbox is not None and checkbox.isChecked() != visible:
                checkbox.blockSignals(True)
                checkbox.setChecked(visible)
                checkbox.blockSignals(False)

        def _sync_live_value_rows_with_plot(self) -> None:
            for sensor in self.sensor_live_rows:
                self._style_live_sensor_row(sensor)
                value_label = self.sensor_live_labels.get(sensor)
                if value_label is not None:
                    value_label.setText(self._latest_sensor_value_text(sensor))

        def _latest_sensor_value_text(self, sensor: str) -> str:
            values = self.sensor_data.get(sensor) or []
            if not values:
                return "—"
            try:
                return f"{float(values[-1]):.1f} °C"
            except Exception:
                return "—"

        def _set_all_plot_visibility(self, visible: bool) -> None:
            for sensor in list(self.plot_visibility):
                self.plot_visibility[sensor] = visible
            self._refresh_plot()

        def _on_plot_hover(self, event) -> None:
            if event.inaxes != self.ax or event.x is None or event.y is None:
                self._clear_hover_annotation()
                return

            best_sensor = None
            best_xy = None
            best_data_xy = None
            best_dist = 12.0  # pixels
            for sensor, line in self.plot_lines.items():
                if not line.get_visible():
                    continue
                try:
                    xy = line.get_xydata()
                    if xy is None or len(xy) == 0:
                        continue
                    display_xy = self.ax.transData.transform(xy)
                    distances = ((display_xy[:, 0] - event.x) ** 2 + (display_xy[:, 1] - event.y) ** 2) ** 0.5
                    idx = int(distances.argmin())
                    dist = float(distances[idx])
                except Exception:
                    continue
                if dist < best_dist:
                    best_dist = dist
                    best_sensor = sensor
                    best_xy = xy[idx]
                    best_data_xy = (float(xy[idx][0]), float(xy[idx][1]))

            if best_sensor is None or best_xy is None or best_data_xy is None:
                self._clear_hover_annotation()
                return

            label = self.sensor_display_names.get(best_sensor, self._format_sensor_label(best_sensor))
            minutes, temp_c = best_data_xy
            self.hover_annotation.xy = (minutes, temp_c)
            self.hover_annotation.set_text(f"{label}\n{temp_c:.1f} °C at {minutes:.1f} min")
            self.hover_annotation.set_visible(True)
            self.hover_value_label.setText(f"Hover: {label} — {temp_c:.1f} °C at {minutes:.1f} min")

            if self._hovered_sensor != best_sensor:
                self._hovered_sensor = best_sensor
                for sensor, line in self.plot_lines.items():
                    if not line.get_visible():
                        continue
                    if sensor == best_sensor:
                        line.set_linewidth(3.6)
                        line.set_alpha(1.0)
                    else:
                        line.set_linewidth(1.4)
                        line.set_alpha(0.35)
            self.canvas.draw_idle()

        def _clear_hover_annotation(self) -> None:
            changed = False
            if self.hover_annotation is not None and self.hover_annotation.get_visible():
                self.hover_annotation.set_visible(False)
                changed = True
            if self._hovered_sensor is not None:
                self._hovered_sensor = None
                for line in self.plot_lines.values():
                    line.set_linewidth(2.0)
                    line.set_alpha(1.0)
                self.hover_value_label.setText("Hover over a plotted line to see the sensor name and value.")
                changed = True
            if changed:
                self.canvas.draw_idle()

        def _on_plot_pick(self, event) -> None:
            sensor = self.legend_artist_to_sensor.get(event.artist)
            if sensor is None:
                return
            self.plot_visibility[sensor] = not self.plot_visibility.get(sensor, True)
            self._refresh_plot()

        def _update_phase_cards(self, phase_text: str) -> None:
            active = None
            for phase_name in self.phase_cards:
                if phase_name in phase_text:
                    active = phase_name
                    break
            completed = {str(item.get("phase", "")) for item in self.phase_summaries}
            for phase_name, card in self.phase_cards.items():
                label = self.phase_card_labels.get(phase_name)
                if phase_name == active:
                    card.setStyleSheet(
                        "QFrame#PhaseCard { background: #eff6ff; border: 2px solid #2563eb; border-radius: 10px; }"
                    )
                    if label is not None:
                        label.setText(label.text().replace("✓ ", "").replace("▶ ", ""))
                        label.setText("▶ " + label.text())
                elif phase_name in completed:
                    card.setStyleSheet(
                        "QFrame#PhaseCard { background: #ecfdf5; border: 1px solid #86efac; border-radius: 10px; }"
                    )
                    if label is not None and not label.text().startswith("✓ "):
                        label.setText(label.text().replace("▶ ", ""))
                        label.setText("✓ " + label.text())
                else:
                    card.setStyleSheet(
                        "QFrame#PhaseCard { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; }"
                    )
                    if label is not None:
                        label.setText(label.text().replace("✓ ", "").replace("▶ ", ""))

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
            self.sensor_live_rows = {}
            self.sensor_live_checkboxes = {}
            self.sensor_live_swatches = {}
            self.sensor_value_chips = {}

            groups: Dict[str, List[str]] = {}
            for key in sensor_keys:
                if key.endswith(":temp_c"):
                    groups.setdefault(self._categorize_sensor(key), []).append(key)

            ordered_groups = [
                ("CPU", sorted(groups.get("CPU", []))),
                ("GPU", sorted(groups.get("GPU", []))),
                ("Storage", sorted(groups.get("NVMe / M.2", []) + groups.get("SSD", []) + groups.get("SATA", []) + groups.get("HDD", []))),
                ("System / Other", sorted(groups.get("Chipset", []) + groups.get("System (ACPI)", []) + groups.get("Other", []))),
            ]
            visible_groups = [(header, sensors) for header, sensors in ordered_groups if sensors]
            if not visible_groups:
                visible_groups = [("Sensors", [])]

            for row_index, (header, sensors) in enumerate(visible_groups):
                frame = QFrame()
                frame.setObjectName("SensorGroup")
                frame.setStyleSheet(
                    "QFrame#SensorGroup { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; }"
                )
                frame_layout = QVBoxLayout(frame)
                frame_layout.setContentsMargins(8, 8, 8, 8)
                frame_layout.setSpacing(6)

                title = QLabel(header)
                title.setStyleSheet("font-weight: 800; color: #334155;")
                frame_layout.addWidget(title)

                if not sensors:
                    empty = QLabel("No temperature sensors detected yet.")
                    empty.setObjectName("SmallMuted")
                    empty.setWordWrap(True)
                    frame_layout.addWidget(empty)
                else:
                    for key in sensors:
                        row_frame = QFrame()
                        row_frame.setObjectName("SensorValueRow")
                        row = QHBoxLayout(row_frame)
                        row.setContentsMargins(8, 6, 8, 6)
                        row.setSpacing(7)

                        color = self.sensor_color_hex.get(key, "#94a3b8")
                        swatch = QLabel()
                        swatch.setFixedSize(10, 10)
                        swatch.setStyleSheet(f"background: {color}; border-radius: 5px;")

                        checkbox = QCheckBox(self.sensor_display_names.get(key, self._format_sensor_label(key)))
                        checkbox.setChecked(self.plot_visibility.get(key, True))
                        checkbox.setToolTip(f"Show/hide {key} on the plot")
                        checkbox.setStyleSheet("QCheckBox { font-weight: 700; color: #334155; }")

                        value_label = QLabel(self._latest_sensor_value_text(key))
                        value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                        value_label.setMinimumWidth(68)
                        value_label.setStyleSheet(
                            "background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; "
                            "padding: 3px 7px; color: #0f172a; font-weight: 800;"
                        )

                        def _make_toggled(sensor_name: str):
                            def _toggled(state: int) -> None:
                                self.plot_visibility[sensor_name] = bool(state)
                                line = self.plot_lines.get(sensor_name)
                                if line is not None:
                                    line.set_visible(bool(state))
                                self._style_live_sensor_row(sensor_name)
                                self._refresh_plot()

                            return _toggled

                        checkbox.stateChanged.connect(_make_toggled(key))

                        row.addWidget(swatch)
                        row.addWidget(checkbox, stretch=1)
                        row.addWidget(value_label)
                        frame_layout.addWidget(row_frame)

                        self.sensor_live_rows[key] = row_frame
                        self.sensor_live_checkboxes[key] = checkbox
                        self.sensor_live_swatches[key] = swatch
                        self.sensor_live_labels[key] = value_label
                        self.sensor_value_chips[key] = value_label
                        self._style_live_sensor_row(key)
                self.sensor_columns_layout.addWidget(frame, row_index, 0)

        def _update_live_temps(self, readings: Dict[str, float]) -> None:
            for key, label in self.sensor_live_labels.items():
                value = readings.get(key)
                label.setText(f"{value:.1f} °C" if value is not None else "—")
            for key in readings:
                self._style_live_sensor_row(key)

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
            steady_timeout_sec = int(self.steady_timeout_spin.value())

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
            self._reset_summary_panel()

            known_sensors = sorted(self.sensor_display_names) or sorted(self.sensor_data)
            self.sensor_data = {sensor: [] for sensor in known_sensors}
            self.sensor_time_data = {sensor: [] for sensor in known_sensors}
            self._refresh_sensor_metadata()
            self._refresh_plot()

            self.buttons_changed.emit(False, True)

            self.worker_thread = threading.Thread(
                target=self._run_test_worker,
                args=(
                    duration_min,
                    interval_sec,
                    cpu_workers,
                    self.gpu_stress_cmd_edit.text().strip(),
                    steady_timeout_sec,
                ),
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
            ]
            tradeoff = assessment.get("clock_tradeoff_note")
            if tradeoff:
                lines.append(tradeoff)
            return "\n".join(lines)

        def _run_test_worker(
            self,
            duration_min: int,
            interval_sec: float,
            cpu_workers: int,
            gpu_stress_cmd: str,
            steady_timeout_sec: int,
        ) -> None:
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
                        if phase_index == 1 and not self.stop_event.is_set():
                            self._set_status("Waiting for temperatures to reach steady state...")
                            summary["post_phase_steady_state"] = wait_until_temperatures_steady(
                                interval_sec=interval_sec,
                                timeout_sec=max(1, int(steady_timeout_sec)),
                                status_cb=self._set_status,
                                progress_cb=self._on_steady_progress,
                                cancel_cb=self.stop_event.is_set,
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
                    "steady_timeout_sec": max(1, int(steady_timeout_sec)),
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
                    self.cooler_assessment_ready.emit(cooler_assessment)

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


    def gui_main() -> int:
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
            target=lambda: win._run_test_worker(
                duration_min=duration_min,
                interval_sec=interval_sec,
                cpu_workers=1,
                gpu_stress_cmd="",
                steady_timeout_sec=120,
            ),
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


def _display_environment_present() -> bool:
    """Return True when a graphical display appears available."""
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if sys.platform == "darwin":
        return True
    if sys.platform.startswith("win"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _strip_dispatch_flags(argv: List[str]) -> Tuple[List[str], Optional[str]]:
    """Remove --gui/--cli dispatch flags before handing argv to the chosen runner."""
    mode: Optional[str] = None
    cleaned = [argv[0]]
    for arg in argv[1:]:
        if arg == "--gui":
            mode = "gui"
        elif arg == "--cli":
            mode = "cli"
        else:
            cleaned.append(arg)
    return cleaned, mode


def _gui_unavailable_reason() -> str:
    if not _display_environment_present():
        return "no DISPLAY or WAYLAND_DISPLAY environment was detected"
    if not GUI_DEPS_AVAILABLE:
        return str(GUI_IMPORT_ERROR) if GUI_IMPORT_ERROR else "GUI dependencies are unavailable"
    return "unknown reason"


def dispatch_main() -> int:
    original_argv = sys.argv[:]
    cleaned_argv, mode = _strip_dispatch_flags(original_argv)
    sys.argv = cleaned_argv

    if mode == "cli" or any(arg in ("-h", "--help") for arg in cleaned_argv[1:]):
        return cli_main()

    gui_possible = _display_environment_present() and GUI_DEPS_AVAILABLE

    if mode == "gui" and not gui_possible:
        print(f"GUI requested, but unavailable: {_gui_unavailable_reason()}.", file=sys.stderr)
        return 2

    if gui_possible:
        try:
            return gui_main()
        except Exception as exc:
            if mode == "gui":
                print(f"GUI failed to start: {exc}", file=sys.stderr)
                return 2
            print(f"GUI unavailable ({exc}); falling back to CLI mode.", file=sys.stderr)

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(dispatch_main())
