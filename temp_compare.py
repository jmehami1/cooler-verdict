#!/usr/bin/env python3
"""CPU/GPU temperature comparison runner.

This script records temperatures across four experiment phases:
1) Uncooled, idle
2) Cooled, idle
3) Uncooled, stressed
4) Cooled, stressed

Each phase defaults to 20 minutes and logs all discovered sensors to CSV.
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


def parse_args() -> argparse.Namespace:
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


def main() -> int:
    args = parse_args()
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


if __name__ == "__main__":
    raise SystemExit(main())
