
# Pytest version of sensor detection tests
import math
import random
import csv
from types import SimpleNamespace
import pytest
from unittest.mock import patch
import cooler_verdict as temp_compare


def _generate_noisy_steady_series(seed: int, steps: int = 100):
    rng = random.Random(seed)
    start = rng.uniform(25.0, 85.0)
    target = rng.uniform(30.0, 80.0)
    # Force a non-trivial direction so the test covers both rise and fall behavior.
    if abs(target - start) < 5.0:
        if target > 55.0:
            start = target + 8.0
        else:
            start = target - 8.0

    value = start
    out = []
    for _ in range(steps):
        # Exponential approach toward target with low-amplitude sensor-like noise.
        value += (target - value) * rng.uniform(0.10, 0.22)
        value += rng.uniform(-0.08, 0.08)
        out.append(value)
    return out

def test_detect_psutil_temps_detects_cpu_values():
    fake_groups = {
        "coretemp": [
            SimpleNamespace(label="Package id 0", current=58.5),
            SimpleNamespace(label="", current=54.0),
            SimpleNamespace(label="nan_sensor", current=float("nan")),
            SimpleNamespace(label="none_sensor", current=None),
        ]
    }
    with patch.object(temp_compare.psutil, "sensors_temperatures", return_value=fake_groups):
        out = temp_compare.detect_psutil_temps()
    assert "coretemp:Package id 0:temp_c" in out
    assert math.isclose(out["coretemp:Package id 0:temp_c"], 58.5)
    assert "coretemp:sensor1:temp_c" in out
    assert math.isclose(out["coretemp:sensor1:temp_c"], 54.0)
    assert "coretemp:nan_sensor:temp_c" not in out
    assert "coretemp:none_sensor:temp_c" not in out

def test_detect_nvidia_temps_detects_gpu_values():
    fake_output = "0, RTX 3080, 71, 93\n1, RTX 3090, 69, 88\n"
    with patch.object(temp_compare.shutil, "which", return_value="/usr/bin/nvidia-smi"):
        with patch.object(temp_compare.subprocess, "check_output", return_value=fake_output):
            out = temp_compare.detect_nvidia_temps()
    assert out["gpu0:RTX 3080:temp_c"] == 71.0
    assert out["gpu0:RTX 3080:util_percent"] == 93.0
    assert out["gpu1:RTX 3090:temp_c"] == 69.0
    assert out["gpu1:RTX 3090:util_percent"] == 88.0

def test_detect_nvidia_temps_returns_empty_when_nvidia_smi_missing():
    with patch.object(temp_compare.shutil, "which", return_value=None):
        out = temp_compare.detect_nvidia_temps()
    assert out == {}


def test_detect_psutil_core_freqs_detects_per_core_and_average():
    fake_freqs = [
        SimpleNamespace(current=3800.0),
        SimpleNamespace(current=3650.5),
        SimpleNamespace(current=None),
        SimpleNamespace(current=float("nan")),
    ]
    with patch.object(temp_compare.psutil, "cpu_freq", return_value=fake_freqs):
        out = temp_compare.detect_psutil_core_freqs()

    assert out["cpu_freq:core0:mhz"] == 3800.0
    assert out["cpu_freq:core1:mhz"] == 3650.5
    assert "cpu_freq:core2:mhz" not in out
    assert "cpu_freq:core3:mhz" not in out
    assert math.isclose(out["cpu_freq:avg:mhz"], (3800.0 + 3650.5) / 2.0)


def test_all_temperatures_steady_with_noisy_generated_profiles():
    history = {
        "cpu:core0:temp_c": _generate_noisy_steady_series(seed=1),
        "cpu:core1:temp_c": _generate_noisy_steady_series(seed=2),
        "gpu0:temp_c": _generate_noisy_steady_series(seed=3),
    }

    assert temp_compare.all_temperatures_steady(
        history_by_sensor=history,
        interval_sec=5.0,
        window_points=18,
        min_points=12,
    )


def test_all_temperatures_steady_rejects_continuous_drift():
    rng = random.Random(99)
    drifting = [40.0 + (0.18 * i) + rng.uniform(-0.04, 0.04) for i in range(80)]
    history = {
        "cpu:core0:temp_c": _generate_noisy_steady_series(seed=4),
        "cpu:core1:temp_c": drifting,
    }

    assert not temp_compare.all_temperatures_steady(
        history_by_sensor=history,
        interval_sec=5.0,
        window_points=18,
        min_points=12,
    )

def test_collect_temperatures_combines_cpu_gpu_and_thermal_sources():
    with patch.object(temp_compare, "detect_psutil_temps", return_value={"cpu:pkg:temp_c": 61.2}):
        with patch.object(temp_compare, "detect_linux_thermal_zone_temps", return_value={"linux_thermal:cpu:temp_c": 60.9}):
            with patch.object(temp_compare, "detect_nvidia_temps", return_value={"gpu0:demo:temp_c": 70.0}):
                with patch.object(temp_compare, "detect_sensors_cmd_temps", return_value={}):
                    with patch.object(temp_compare, "detect_psutil_core_freqs", return_value={"cpu_freq:core0:mhz": 4100.0}):
                        out = temp_compare.collect_temperatures()
    assert out["cpu:pkg:temp_c"] == 61.2
    assert out["linux_thermal:cpu:temp_c"] == 60.9
    assert out["gpu0:demo:temp_c"] == 70.0
    assert out["cpu_freq:core0:mhz"] == 4100.0


def _write_phase_rows(csv_path, phase, cpu_temp, gpu_temp, cpu_freq, count=30):
    with csv_path.open("a", newline="") as f:
        w = csv.writer(f)
        for i in range(count):
            ts = f"2026-01-01T00:00:{i:02d}+00:00"
            load_state = "stress" if "stress" in phase else "idle"
            cooling_state = "cooled" if phase.startswith("cooled") else "uncooled"
            w.writerow([ts, phase, cooling_state, load_state, "cpu:pkg:temp_c", f"{cpu_temp:.3f}"])
            w.writerow([ts, phase, cooling_state, load_state, "gpu0:demo:temp_c", f"{gpu_temp:.3f}"])
            w.writerow([ts, phase, cooling_state, load_state, "cpu_freq:avg:mhz", f"{cpu_freq:.3f}"])


def test_summarize_cooler_effect_from_csv_recommends_keep(tmp_path):
    csv_path = tmp_path / "run.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_utc", "phase", "cooling_state", "load_state", "sensor", "value"])

    _write_phase_rows(csv_path, "uncooled_idle", cpu_temp=55.0, gpu_temp=52.0, cpu_freq=3000.0)
    _write_phase_rows(csv_path, "cooled_idle", cpu_temp=50.0, gpu_temp=49.0, cpu_freq=3150.0)
    _write_phase_rows(csv_path, "uncooled_stress", cpu_temp=88.0, gpu_temp=80.0, cpu_freq=3650.0)
    _write_phase_rows(csv_path, "cooled_stress", cpu_temp=79.0, gpu_temp=74.0, cpu_freq=3950.0)

    out = temp_compare.summarize_cooler_effect_from_csv(csv_path)
    assert out["keep_recommended"]
    assert out["best_temp_drop_c"] >= 5.0
    assert out["best_cpu_freq_gain_pct"] >= 3.0
    assert out["noise_considered"] is False


def test_summarize_cooler_effect_from_csv_flags_temp_clock_tradeoff(tmp_path):
    csv_path = tmp_path / "tradeoff.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_utc", "phase", "cooling_state", "load_state", "sensor", "value"])

    _write_phase_rows(csv_path, "uncooled_idle", cpu_temp=53.0, gpu_temp=50.0, cpu_freq=2800.0)
    _write_phase_rows(csv_path, "cooled_idle", cpu_temp=54.5, gpu_temp=50.5, cpu_freq=3000.0)
    _write_phase_rows(csv_path, "uncooled_stress", cpu_temp=85.0, gpu_temp=78.0, cpu_freq=3550.0)
    _write_phase_rows(csv_path, "cooled_stress", cpu_temp=87.0, gpu_temp=79.5, cpu_freq=3780.0)

    out = temp_compare.summarize_cooler_effect_from_csv(csv_path)
    assert out["clock_tradeoff_note"] is not None
    assert out["best_cpu_freq_gain_pct"] >= 3.0
