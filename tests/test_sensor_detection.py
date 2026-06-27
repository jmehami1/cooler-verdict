
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
    assert out["verdict"] in ("keep", "definitely_keep")
    assert out["best_temp_drop_c"] >= 5.0
    assert out["best_cpu_freq_gain_pct"] >= 3.0
    criteria_by_name = {item["name"]: item for item in out["criteria_evaluation"]}
    assert criteria_by_name["data_complete"]["status"] == "pass"
    assert criteria_by_name["comparable_cpu_or_gpu_sensors_available"]["status"] == "pass"
    assert criteria_by_name["minimum_samples_met"]["status"] == "pass"
    assert criteria_by_name["idle_temperature_improved"]["status"] == "pass"
    assert criteria_by_name["keep_requirements_met"]["status"] == "pass"
    assert out["rolling_window_points"] == 5
    assert out["rolling_overtime_comparisons"]["idle"]["cpu"]["mean_rolling_drop_c"] is not None
    assert out["rolling_overtime_comparisons"]["stress"]["cpu"]["mean_rolling_drop_c"] is not None
    sensor_map = {item["sensor"]: item for item in out["sensor_temp_drop_summary"]}
    assert "cpu:pkg:temp_c" in sensor_map
    assert sensor_map["cpu:pkg:temp_c"]["idle_drop_pct"] is not None
    assert sensor_map["cpu:pkg:temp_c"]["stress_drop_pct"] is not None


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
    criteria_by_name = {item["name"]: item for item in out["criteria_evaluation"]}
    assert criteria_by_name["stress_performance_positive_tradeoff"]["status"] in ("pass", "warning", "fail")
    assert len(out["sensor_temp_drop_summary"]) > 0


def test_wait_until_temperatures_steady_honors_cancel_callback():
    out = temp_compare.wait_until_temperatures_steady(
        interval_sec=1.0,
        timeout_sec=60,
        cancel_cb=lambda: True,
    )

    assert out["cancelled"] is True
    assert out["timed_out"] is False
    assert out["reached_steady"] is False


def test_rolling_overtime_comparison_suppresses_single_spike(tmp_path):
    csv_path = tmp_path / "rolling_spike.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_utc", "phase", "cooling_state", "load_state", "sensor", "value"])

        for i in range(24):
            ts = f"2026-01-01T00:10:{i:02d}+00:00"
            uncooled_temp = 88.0
            cooled_temp = 82.0
            if i == 12:
                cooled_temp = 100.0  # synthetic spike

            w.writerow([ts, "uncooled_stress", "uncooled", "stress", "cpu:pkg:temp_c", f"{uncooled_temp:.3f}"])
            w.writerow([ts, "cooled_stress", "cooled", "stress", "cpu:pkg:temp_c", f"{cooled_temp:.3f}"])
            w.writerow([ts, "uncooled_stress", "uncooled", "stress", "cpu_freq:avg:mhz", "3600.000"])
            w.writerow([ts, "cooled_stress", "cooled", "stress", "cpu_freq:avg:mhz", "3700.000"])

        for i in range(24):
            ts = f"2026-01-01T00:20:{i:02d}+00:00"
            w.writerow([ts, "uncooled_idle", "uncooled", "idle", "cpu:pkg:temp_c", "55.000"])
            w.writerow([ts, "cooled_idle", "cooled", "idle", "cpu:pkg:temp_c", "51.000"])
            w.writerow([ts, "uncooled_idle", "uncooled", "idle", "cpu_freq:avg:mhz", "3000.000"])
            w.writerow([ts, "cooled_idle", "cooled", "idle", "cpu_freq:avg:mhz", "3100.000"])

    out = temp_compare.summarize_cooler_effect_from_csv(csv_path)
    stress_cpu_profile = out["rolling_overtime_comparisons"]["stress"]["cpu"]

    assert stress_cpu_profile["samples_compared"] == 24
    assert stress_cpu_profile["mean_rolling_drop_c"] is not None
    assert stress_cpu_profile["mean_rolling_drop_c"] > 0.0
    assert stress_cpu_profile["worst_rolling_drop_c"] is not None
    assert stress_cpu_profile["worst_rolling_drop_c"] > -15.0
    assert stress_cpu_profile["points_cooler_pct"] is not None
    assert stress_cpu_profile["points_cooler_pct"] > 70.0


# ============ Tests for validation functions ============

def test_is_valid_temperature_c_accepts_reasonable_values():
    """Test that physically reasonable temperatures pass validation."""
    assert temp_compare._is_valid_temperature_c(0.0)
    assert temp_compare._is_valid_temperature_c(20.5)
    assert temp_compare._is_valid_temperature_c(50.0)
    assert temp_compare._is_valid_temperature_c(100.0)
    assert temp_compare._is_valid_temperature_c(-10.0)
    assert temp_compare._is_valid_temperature_c(-50.0)  # Boundary: min
    assert temp_compare._is_valid_temperature_c(150.0)  # Boundary: max


def test_is_valid_temperature_c_rejects_invalid_values():
    """Test that invalid temperatures are rejected."""
    assert not temp_compare._is_valid_temperature_c(float('nan'))
    assert not temp_compare._is_valid_temperature_c(float('inf'))
    assert not temp_compare._is_valid_temperature_c(float('-inf'))
    assert not temp_compare._is_valid_temperature_c(None)
    assert not temp_compare._is_valid_temperature_c(-51.0)  # Below min
    assert not temp_compare._is_valid_temperature_c(151.0)  # Above max
    assert not temp_compare._is_valid_temperature_c(-100.0)
    assert not temp_compare._is_valid_temperature_c(200.0)


def test_is_valid_cpu_frequency_mhz_accepts_reasonable_values():
    """Test that physically reasonable CPU frequencies pass validation."""
    assert temp_compare._is_valid_cpu_frequency_mhz(1.0)  # Boundary: min
    assert temp_compare._is_valid_cpu_frequency_mhz(1000.0)
    assert temp_compare._is_valid_cpu_frequency_mhz(3500.0)
    assert temp_compare._is_valid_cpu_frequency_mhz(5000.0)
    assert temp_compare._is_valid_cpu_frequency_mhz(10000.0)  # Boundary: max


def test_is_valid_cpu_frequency_mhz_rejects_invalid_values():
    """Test that invalid CPU frequencies are rejected."""
    assert not temp_compare._is_valid_cpu_frequency_mhz(float('nan'))
    assert not temp_compare._is_valid_cpu_frequency_mhz(float('inf'))
    assert not temp_compare._is_valid_cpu_frequency_mhz(float('-inf'))
    assert not temp_compare._is_valid_cpu_frequency_mhz(None)
    assert not temp_compare._is_valid_cpu_frequency_mhz(0.5)  # Below min
    assert not temp_compare._is_valid_cpu_frequency_mhz(0.0)
    assert not temp_compare._is_valid_cpu_frequency_mhz(-100.0)
    assert not temp_compare._is_valid_cpu_frequency_mhz(10001.0)  # Above max


# ============ Tests for deduplication functions ============

def test_find_duplicate_sensors_identifies_identical_sensors():
    """Test that identical sensor readings are correctly identified as duplicates."""
    phase_sensor_temps = {
        "uncooled_idle": {
            "cpu:core0:temp_c": [40.0, 40.1, 40.2],
            "cpu:core1:temp_c": [40.0, 40.1, 40.2],  # Identical to core0
            "gpu0:temp_c": [70.0, 70.1, 70.2],
        },
        "cooled_idle": {
            "cpu:core0:temp_c": [35.0, 35.1, 35.2],
            "cpu:core1:temp_c": [35.0, 35.1, 35.2],  # Still identical
            "gpu0:temp_c": [65.0, 65.1, 65.2],
        },
        "uncooled_stress": {},
        "cooled_stress": {},
    }
    
    canonical_map = temp_compare._find_duplicate_sensors(phase_sensor_temps, tolerance=0.01)
    
    # cpu:core1 should map to cpu:core0 (first occurrence)
    assert canonical_map["cpu:core1:temp_c"] == "cpu:core0:temp_c"
    # gpu0 should stay as itself (not identical to CPU temps)
    assert canonical_map["gpu0:temp_c"] == "gpu0:temp_c"


def test_find_duplicate_sensors_handles_no_duplicates():
    """Test that distinct sensors are not marked as duplicates."""
    phase_sensor_temps = {
        "uncooled_idle": {
            "cpu:core0:temp_c": [40.0, 40.1, 40.2],
            "gpu0:temp_c": [70.0, 70.1, 70.2],
            "storage:ssd:temp_c": [45.0, 45.1, 45.2],
        },
        "cooled_idle": {
            "cpu:core0:temp_c": [35.0, 35.1, 35.2],
            "gpu0:temp_c": [65.0, 65.1, 65.2],
            "storage:ssd:temp_c": [40.0, 40.1, 40.2],
        },
        "uncooled_stress": {},
        "cooled_stress": {},
    }
    
    canonical_map = temp_compare._find_duplicate_sensors(phase_sensor_temps, tolerance=0.01)
    
    # All should map to themselves
    assert canonical_map["cpu:core0:temp_c"] == "cpu:core0:temp_c"
    assert canonical_map["gpu0:temp_c"] == "gpu0:temp_c"
    assert canonical_map["storage:ssd:temp_c"] == "storage:ssd:temp_c"


def test_find_duplicate_sensors_near_identical_within_tolerance():
    """Test that nearly identical sensors (within tolerance) are marked as duplicates."""
    phase_sensor_temps = {
        "uncooled_idle": {
            "sensor_a:temp_c": [50.0, 50.1, 50.2],
            "sensor_b:temp_c": [50.3, 50.4, 50.5],  # ~0.5% difference, within tolerance
        },
        "cooled_idle": {
            "sensor_a:temp_c": [45.0, 45.1, 45.2],
            "sensor_b:temp_c": [45.3, 45.4, 45.5],
        },
        "uncooled_stress": {},
        "cooled_stress": {},
    }
    
    canonical_map = temp_compare._find_duplicate_sensors(phase_sensor_temps, tolerance=0.02)
    
    # sensor_b should map to sensor_a (very close readings)
    assert canonical_map["sensor_b:temp_c"] == "sensor_a:temp_c"


def test_find_duplicate_sensors_preserves_first_occurrence():
    """Test that deduplication keeps the first sensor name (alphabetically)."""
    phase_sensor_temps = {
        "uncooled_idle": {
            "z_sensor:temp_c": [50.0, 50.1, 50.2],
            "a_sensor:temp_c": [50.0, 50.1, 50.2],  # Identical to z_sensor
        },
        "cooled_idle": {},
        "uncooled_stress": {},
        "cooled_stress": {},
    }
    
    canonical_map = temp_compare._find_duplicate_sensors(phase_sensor_temps, tolerance=0.01)
    
    # a_sensor comes first alphabetically
    assert canonical_map["z_sensor:temp_c"] == "a_sensor:temp_c"


def test_deduplicate_phase_data_consolidates_values():
    """Test that deduplication consolidates sensor values correctly."""
    phase_sensor_temps = {
        "uncooled_idle": {
            "sensor_a:temp_c": [40.0, 41.0],
            "sensor_b:temp_c": [40.0, 41.0],  # Duplicate
        },
        "cooled_idle": {
            "sensor_a:temp_c": [35.0, 36.0],
            "sensor_b:temp_c": [35.0, 36.0],
        },
    }
    
    phase_component_sensors = {
        "uncooled_idle": {"cpu": {"sensor_a:temp_c", "sensor_b:temp_c"}, "gpu": set(), "storage": set()},
        "cooled_idle": {"cpu": {"sensor_a:temp_c", "sensor_b:temp_c"}, "gpu": set(), "storage": set()},
    }
    
    canonical_map = {
        "sensor_a:temp_c": "sensor_a:temp_c",
        "sensor_b:temp_c": "sensor_a:temp_c",
    }
    
    temp_compare._deduplicate_phase_data(phase_sensor_temps, phase_component_sensors, canonical_map)
    
    # After deduplication, only sensor_a should exist
    assert "sensor_a:temp_c" in phase_sensor_temps["uncooled_idle"]
    assert "sensor_b:temp_c" not in phase_sensor_temps["uncooled_idle"]
    
    # Values should be consolidated (merged)
    assert len(phase_sensor_temps["uncooled_idle"]["sensor_a:temp_c"]) == 4  # 2 + 2 values
    assert phase_sensor_temps["uncooled_idle"]["sensor_a:temp_c"] == [40.0, 41.0, 40.0, 41.0]
    
    # Component tracking should be updated
    assert "sensor_a:temp_c" in phase_component_sensors["uncooled_idle"]["cpu"]
    assert "sensor_b:temp_c" not in phase_component_sensors["uncooled_idle"]["cpu"]


def test_deduplicate_phase_data_handles_empty_phases():
    """Test that deduplication handles empty phases gracefully."""
    phase_sensor_temps = {
        "uncooled_idle": {"sensor_a:temp_c": [40.0, 41.0]},
        "cooled_idle": {},
        "uncooled_stress": {},
        "cooled_stress": {},
    }
    
    phase_component_sensors = {
        "uncooled_idle": {"cpu": {"sensor_a:temp_c"}, "gpu": set(), "storage": set()},
        "cooled_idle": {"cpu": set(), "gpu": set(), "storage": set()},
        "uncooled_stress": {"cpu": set(), "gpu": set(), "storage": set()},
        "cooled_stress": {"cpu": set(), "gpu": set(), "storage": set()},
    }
    
    canonical_map = {"sensor_a:temp_c": "sensor_a:temp_c"}
    
    # Should not raise any errors
    temp_compare._deduplicate_phase_data(phase_sensor_temps, phase_component_sensors, canonical_map)
    
    assert phase_sensor_temps["uncooled_idle"]["sensor_a:temp_c"] == [40.0, 41.0]
    assert phase_sensor_temps["cooled_idle"] == {}


def test_invalid_readings_filtered_in_detect_functions():
    """Test that invalid readings are properly filtered in detection functions."""
    fake_groups = {
        "coretemp": [
            SimpleNamespace(label="valid", current=50.0),
            SimpleNamespace(label="too_cold", current=-51.0),
            SimpleNamespace(label="too_hot", current=151.0),
            SimpleNamespace(label="nan_val", current=float("nan")),
        ]
    }
    with patch.object(temp_compare.psutil, "sensors_temperatures", return_value=fake_groups):
        out = temp_compare.detect_psutil_temps()
    
    # Only valid reading should be included
    assert "coretemp:valid:temp_c" in out
    assert out["coretemp:valid:temp_c"] == 50.0
    assert "coretemp:too_cold:temp_c" not in out
    assert "coretemp:too_hot:temp_c" not in out
    assert "coretemp:nan_val:temp_c" not in out


def test_invalid_cpu_frequencies_filtered():
    """Test that invalid CPU frequency readings are filtered."""
    fake_freqs = [
        SimpleNamespace(current=3500.0),  # Valid
        SimpleNamespace(current=0.5),      # Too low
        SimpleNamespace(current=15000.0),  # Too high
        SimpleNamespace(current=float("nan")),  # NaN
    ]
    with patch.object(temp_compare.psutil, "cpu_freq", return_value=fake_freqs):
        out = temp_compare.detect_psutil_core_freqs()
    
    # Only valid frequency should be included in output
    assert "cpu_freq:core0:mhz" in out
    assert out["cpu_freq:core0:mhz"] == 3500.0
    assert "cpu_freq:core1:mhz" not in out
    assert "cpu_freq:core2:mhz" not in out
    assert "cpu_freq:core3:mhz" not in out
    assert math.isclose(out["cpu_freq:avg:mhz"], 3500.0)
