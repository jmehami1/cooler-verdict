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


def infer_gpu_vendor() -> str:
    if shutil.which("nvidia-smi"):
        return "nvidia"
    if Path("/sys/class/drm").exists():
        return "unknown"
    return "none"


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
        out[f"gpu{idx}:{name}:temp_c"] = temp
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
            if entry.current is not None and not math.isnan(entry.current):
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
        if math.isnan(cur_mhz):
            continue
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
        if elapsed >= timeout_sec:
            emit("Steady-state timeout reached; continuing to next phase.")
            return {
                "reached_steady": False,
                "timed_out": True,
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
            return {
                "reached_steady": True,
                "timed_out": False,
                "wait_duration_sec": int(elapsed),
                "samples": samples,
                "temp_sensors_tracked": temp_sensor_count,
            }

        next_sample += interval_sec


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


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


def summarize_cooler_effect_from_csv(csv_path: Path) -> dict:
    """Summarize cooler effectiveness from phase CSV data.

    Uses final-window averages per phase and ignores external noise because
    this runner does not measure acoustic data.
    """
    phase_metrics: Dict[str, Dict[str, List[float]]] = {
        "uncooled_idle": {"cpu_temp": [], "gpu_temp": [], "cpu_freq": []},
        "cooled_idle": {"cpu_temp": [], "gpu_temp": [], "cpu_freq": []},
        "uncooled_stress": {"cpu_temp": [], "gpu_temp": [], "cpu_freq": []},
        "cooled_stress": {"cpu_temp": [], "gpu_temp": [], "cpu_freq": []},
    }

    if not csv_path.exists():
        return {
            "verdict": "inconclusive",
            "keep_recommended": False,
            "summary_short": "Inconclusive: no CSV data.",
            "noise_considered": False,
        }

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            phase = (row.get("phase") or "").strip()
            sensor = (row.get("sensor") or "").strip()
            value_raw = (row.get("value") or "").strip()
            if phase not in phase_metrics:
                continue
            try:
                value = float(value_raw)
            except ValueError:
                continue

            if sensor == "cpu_freq:avg:mhz":
                phase_metrics[phase]["cpu_freq"].append(value)
            if _is_cpu_temp_sensor(sensor):
                phase_metrics[phase]["cpu_temp"].append(value)
            if _is_gpu_temp_sensor(sensor):
                phase_metrics[phase]["gpu_temp"].append(value)

    phase_stats: Dict[str, Dict[str, Optional[float]]] = {}
    for phase, metrics in phase_metrics.items():
        phase_stats[phase] = {
            "cpu_temp_avg": _mean(_tail_window(metrics["cpu_temp"])),
            "gpu_temp_avg": _mean(_tail_window(metrics["gpu_temp"])),
            "cpu_freq_avg_mhz": _mean(_tail_window(metrics["cpu_freq"])),
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
            "cpu_freq_gain_pct": pct_gain(
                phase_stats["cooled_stress"]["cpu_freq_avg_mhz"],
                phase_stats["uncooled_stress"]["cpu_freq_avg_mhz"],
            ),
        },
    }

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

    performance_positive_tradeoff = False
    for profile in comparisons.values():
        profile_best_drop = max(
            [x for x in (profile["cpu_temp_drop_c"], profile["gpu_temp_drop_c"]) if x is not None],
            default=None,
        )
        freq_gain = profile["cpu_freq_gain_pct"]
        if profile_best_drop is None or freq_gain is None:
            continue
        # Same/slightly hotter (within 3C) with stronger clocks is considered acceptable.
        if profile_best_drop <= 0.0 and profile_best_drop >= -3.0 and freq_gain >= 5.0:
            performance_positive_tradeoff = True

    if worst_temp_drop < -3.0 and best_freq_gain < 3.0:
        verdict = "reject"
        keep = False
        reason = "hotter without meaningful clock gain"
    elif best_temp_drop >= 8.0 and best_freq_gain >= 5.0:
        verdict = "definitely_keep"
        keep = True
        reason = "large thermal drop and higher clocks"
    elif best_temp_drop >= 5.0 and best_freq_gain >= 3.0:
        verdict = "keep"
        keep = True
        reason = "good thermal drop with clock uplift"
    elif performance_positive_tradeoff:
        verdict = "keep"
        keep = True
        reason = "performance-positive thermal tradeoff"
    elif best_temp_drop >= 5.0 or best_freq_gain >= 5.0:
        verdict = "keep"
        keep = True
        reason = "meets practical keep threshold"
    elif best_temp_drop < 3.0 and best_freq_gain < 3.0:
        verdict = "probably_return"
        keep = False
        reason = "marginal improvement"
    else:
        verdict = "inconclusive"
        keep = False
        reason = "mixed or limited data"

    clock_tradeoff_note = None
    if any((v is not None and v < 0.0) for v in all_temp_drops) and best_freq_gain >= 3.0:
        clock_tradeoff_note = "Some phases ran hotter while CPU clocks increased."

    summary_short = (
        f"Verdict: {'KEEP' if keep else 'RETURN'}. "
        f"Best temp drop {best_temp_drop:.1f}C, best CPU clock gain {best_freq_gain:.1f}%."
    )

    return {
        "verdict": verdict,
        "keep_recommended": keep,
        "reason": reason,
        "summary_short": summary_short,
        "noise_considered": False,
        "clock_tradeoff_note": clock_tradeoff_note,
        "best_temp_drop_c": round(best_temp_drop, 3),
        "worst_temp_drop_c": round(worst_temp_drop, 3),
        "best_cpu_freq_gain_pct": round(best_freq_gain, 3),
        "phase_stats": phase_stats,
        "comparisons": comparisons,
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

    if gpu_stress_cmd:
        stressors.append(run_subprocess(["bash", "-lc", gpu_stress_cmd], "gpu-stress-custom"))
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
    return parser.parse_args()


def cli_main() -> int:
    args = parse_cli_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    experiment_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"temperature_log_{experiment_id}.csv"
    metadata_path = out_dir / f"run_metadata_{experiment_id}.json"

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

            if idx < len(phases) - 1:
                summary["post_phase_steady_state"] = wait_until_temperatures_steady(
                    interval_sec=args.interval_sec,
                )

            phase_summaries.append(summary)

    metadata = {
        "experiment_id": experiment_id,
        "started_utc": utc_now_iso(),
        "duration_per_phase_min": args.duration_min,
        "interval_sec": args.interval_sec,
        "cpu_workers": args.cpu_workers,
        "gpu_stress_cmd": args.gpu_stress_cmd,
        "gpu_vendor_guess": infer_gpu_vendor(),
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
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal
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

        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("Cooler Verdict - Interactive Dashboard v3")
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
            self._hovered_sensor: Optional[str] = None

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
            subtitle = QLabel("Interactive dashboard v3: configure the test, monitor live sensor plots, hover lines for sensor identity, and get a keep/return verdict.")
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

            self.gpu_stress_cmd_edit = QLineEdit()
            self.gpu_stress_cmd_edit.setPlaceholderText("Optional shell command")
            self.enforce_clean_checkbox = QCheckBox("Abort if busy user processes are active")

            setup_form.addRow("Phase duration", self.duration_spin)
            setup_form.addRow("Sample interval", self.interval_spin)
            setup_form.addRow("CPU workers", self.cpu_workers_spin)
            setup_form.addRow("CPU threshold", self.cpu_threshold_spin)
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
            self.show_all_button = QPushButton("Show all")
            self.hide_all_button = QPushButton("Hide all")
            plot_header.addWidget(self.show_all_button)
            plot_header.addWidget(self.hide_all_button)
            plot_layout.addLayout(plot_header)

            self.toolbar = NavigationToolbar(self.canvas, self)
            plot_layout.addWidget(self.toolbar)
            self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.canvas.setMinimumHeight(520)
            plot_layout.addWidget(self.canvas, stretch=1)
            self.hover_value_label = QLabel("Hover over a plotted line to see the sensor name and value.")
            self.hover_value_label.setObjectName("Muted")
            self.hover_value_label.setWordWrap(True)
            plot_layout.addWidget(self.hover_value_label)
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
            self.show_all_button.clicked.connect(lambda: self._set_all_plot_visibility(True))
            self.hide_all_button.clicked.connect(lambda: self._set_all_plot_visibility(False))

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
                label = self.sensor_display_names.get(sensor, self._format_sensor_label(sensor))
                color = self.sensor_color_hex.get(sensor)
                if color:
                    (line,) = self.ax.plot(xs, ys, label=label, linewidth=2.0, color=color, solid_capstyle="round")
                else:
                    (line,) = self.ax.plot(xs, ys, label=label, linewidth=2.0, solid_capstyle="round")
                    try:
                        self.sensor_color_hex[sensor] = to_hex(line.get_color())
                    except Exception:
                        self.sensor_color_hex[sensor] = "#2563eb"
                line.set_picker(7)
                visible = self.plot_visibility.get(sensor, True)
                line.set_visible(visible)
                if visible:
                    visible_values.extend([float(v) for v in ys if v is not None])
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
