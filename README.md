# CoolVerdict

I really hate overheating laptops, but I also hate external cooling solutions that don't work. This project is a simple tool to help me (and hopefully others) determine if an external cooler is actually effective at reducing temperatures under load. 

It compares idle and stress temperatures before and after attaching the cooler, and it gives a clear verdict on whether the cooler is worth keeping or should be returned.

![Cooling Compare GUI](docs/GUI.png)

## What It Does

Runs a four-phase cooling comparison test: idle and stress, before and after the external cooler is attached. It logs temperature sensors, CPU frequency, and GPU telemetry, and it can check for background processes before starting.

- CLI runner: `python temp_compare.py`
- GUI runner: `python temp_compare_gui.py`
- Live chart, sensor list, and end-of-run KEEP/RETURN verdict
- Optional clean-start enforcement for interference checks

## Setup

Tested on Ubuntu 24.04. 

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Ubuntu sensor packages:

```bash
sudo apt install lm-sensors
sudo apt install nvidia-utils-<your-driver-version>  # only if you want NVIDIA GPU temps via nvidia-smi
sudo apt install nvme-cli  # only if you want NVMe temps via nvme smart-log
```

`lm-sensors` provides the `sensors` command used for broader CPU/SSD/NVMe sensor coverage. NVMe temperatures are picked up when exposed by `sensors` or the Linux thermal zones, so no extra Python package is needed.

## Output

Results are written to `results/` as CSV logs and run metadata JSON.

## Tests

```bash
pytest tests
```
