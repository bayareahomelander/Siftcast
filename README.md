# Siftcast

Siftcast trains a tiny CPU GRU to jointly forecast the **next reconstructed sensor reading** and the **next packet-trust bit** from a Gilbert–Elliott serial capture. A C++ plant wraps a small public temperature series into CRC-framed packets and corrupts them on the wire; a Rust streaming decoder CRC-checks, de-duplicates, and reorders that capture into a trust-tagged timeline; Python then updates real weights and infers from the checkpoint. The reconstruct/trust path is the product: without it this is ordinary univariate forecasting of a dirty arrival-order dump.

## Setup

Windows, PowerShell or cmd. This repo was built and run on a 16 GB RAM / 6 GB VRAM box.

- Python 3.12+ with `numpy` (`pip install numpy` if needed)
- Rust toolchain (`cargo` / `rustc`)
- MSVC Build Tools (the `vcvars64.bat` that ships with Visual Studio Build Tools 2022)

No CUDA, no pretrained weights, no extra Python packages.

Confirm:

```text
python --version
cargo --version
```

## Run

From the repo root, one command builds the natives, obtains the small CSV (download, or the vendored copy if the network is down), simulates a capture, reconstructs it, trains, and infers:

```text
python siftcast.py
```

Useful knobs (all optional):

```text
python siftcast.py --n 4096 --steps 500 --seed 1 --horizon 16 --error-scale 1.0
```

A second run reuses the compiled binaries and the cached CSV under `artifacts/`.

## Tests

```text
python tests/test_siftcast.py
```

That drives the shipped GRU `train_step` / checkpoint / infer path, the C++ simulator (`--self-test` plus a clean capture), and the Rust decoder (`cargo test` plus reconstruct of the clean and noisy captures).

## What you should see

- `sim:` stderr from the C++ encoder (packet counts, drops, bit flips)
- `reconstruct:` stderr from the Rust decoder (trusted vs held ticks, CRC fails)
- `train step k/N loss=...` with the last loss below the first
- `checkpoint wrote artifacts/checkpoint.npz`
- `siftcast: wrote artifacts/forecast.json points=16`
- `budget peak_ws_mb=...` well under 16 GB; training is CPU-only so VRAM stays at the desktop baseline

## Outputs

| path | what |
| --- | --- |
| `artifacts/daily-min-temperatures.csv` | obtained series (same schema as `vendor/`) |
| `artifacts/capture.bin` | Gilbert–Elliott wire dump from C++ |
| `artifacts/series.jsonl` | reconstructed timeline from Rust |
| `artifacts/checkpoint.npz` | trained GRU weights + norm stats |
| `artifacts/forecast.json` | 16-step forecast: `x_c` (°C) and `trust` per step |
| `artifacts/metrics.json` | train losses, holdout vs persistence MSE |
| `artifacts/budget.txt` / `artifacts/ram.txt` | peak working set, file sizes, `nvidia-smi` |

## Project structure

```text
siftcast.py          obtain + build + train + infer (the one command)
sim.cpp              C++ plant, CRC-8 frame codec, Gilbert–Elliott channel
reconstruct/         Rust streaming CRC / reorder / hold reconstruct
vendor/              offline copy of the ~68 KB daily-min-temperatures CSV
tests/test_siftcast.py
artifacts/           runtime outputs (created on run)
build/sim.exe        produced by MSVC
```

## Pipeline

1. **Obtain** Melbourne daily-min-temperatures (~68 KB). If the GitHub raw URL fails, `vendor/daily-min-temperatures.csv` is used.
2. **C++ `sim`** interpolates that series through a forced thermal plant, packs each tick as `A5 5A | seq | t_ms | x | y | crc8`, then applies a two-state Gilbert–Elliott channel (bit flips, drops, duplicates, reorder delay, junk bytes).
3. **Rust `reconstruct`** hunts sync bytes, verifies CRC-8/SMBUS, reorders by sequence, and emits one JSONL row per seq with `trust=1` on committed packets and `trust=0` on holds for gaps.
4. **Python** trains a 16-hidden GRU on windows of `(x, y, trust)` with trust-gated value MSE plus trust BCE, writes `checkpoint.npz`, then rolls out a 16-step forecast.

C++, Rust, and Python each sit on that train/infer path. Drop C++ and there is no framed capture; drop Rust and there are no trust labels; drop Python and no weights update.

## Frame

13 bytes, little-endian:

```text
A5 5A | seq u16 | t_ms u32 | x i16 (cK) | y i16 (cK) | crc8
```

CRC-8/SMBUS (poly `0x07`, init `0`, check vector `"123456789"` → `0xF4`) covers `seq` through `y`, not the sync bytes.
