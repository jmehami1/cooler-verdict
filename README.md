# CoolVerdict

CoolVerdict compares idle and stress temperatures before and after attaching an external cooler, then gives a simple keep/return verdict.

Linux-only: this tool is intended for Linux systems (tested on Ubuntu 24.04).

![Cooling Compare GUI](docs/GUI.png)

## What It Does

Runs a four-phase cooling comparison test: idle and stress, before and after the external cooler is attached. It logs temperature sensors, CPU frequency, and GPU telemetry, and it can check for background processes before starting.

- Single-file runner: `python cooler_verdict.py`
- Live chart, sensor list, and end-of-run KEEP/RETURN verdict
- Optional clean-start enforcement for interference checks

## Setup

Ubuntu sensor packages:

```bash
sudo apt install lm-sensors
sudo apt install nvidia-utils-<your-driver-version>  # only if you want NVIDIA GPU temps via nvidia-smi
sudo apt install nvme-cli  # only if you want NVMe temps via nvme smart-log
```

`lm-sensors` provides the `sensors` command used for broader CPU/SSD/NVMe sensor coverage. NVMe temperatures are picked up when exposed by `sensors` or Linux thermal zones, so no extra Python package is needed.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Output

Results are written to `results/` as CSV logs and run metadata JSON.

## Tests

```bash
pytest tests
```

## Verdict Criteria (Summary)

| Criterion | Pass Rule | Effect |
| --- | --- | --- |
| Data complete | All 4 phases present; enough comparable CPU/GPU samples | Required, else inconclusive |
| Idle improved | Idle best CPU/GPU drop >= 3 C | Required for keep |
| Idle penalty check | No idle CPU/GPU rise worse than -2 C | Return on fail |
| Stress temp improved | Stress best CPU/GPU drop >= 3 C | Supports keep |
| Stress tradeoff positive | Stress can be up to +3 C hotter only with >= 3% stress clock gain | Alternative keep path |
| Stress penalty check | No stress rise worse than -3 C without >= 3% clock gain | Return on fail |
| Definitely keep upgrade | Keep conditions plus strong stress drop (>= 5 C, very strong >= 8 C) | Upgrades to definitely_keep |

## License

Free to use under the MIT License. See LICENSE.