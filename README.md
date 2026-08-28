# Siftcast

Siftcast trains a tiny CPU GRU to jointly forecast **16 future reconstructed sensor readings** (mean and scale) and **16 packet-trust bits** from a Gilbert–Elliott serial capture. A C++ plant wraps a small public temperature series into CRC-framed packets and corrupts them on the wire; a Rust streaming decoder CRC-checks, de-duplicates, and reorders that capture into a trust-tagged timeline; Python then updates real weights, fits split-conformal-style interval quantiles, and infers from the checkpoint in **one direct pass**. The reconstruct/trust path is the product: without it this is ordinary univariate forecasting of a dirty arrival-order dump.

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

From the repo root, one command builds the natives, obtains the small CSV (download, or the vendored copy if the network is down), simulates a capture, reconstructs it, trains, calibrates, and infers:

```text
python siftcast.py
```

Useful knobs (all optional):

```text
python siftcast.py --n 4096 --steps 500 --seed 1 --horizon 16 --error-scale 1.0 --coverage 0.90
```

`--horizon` is the trained output width (one GRU pass emits all `H` steps). `--coverage` is the nominal level of the split-conformal-style intervals (default `0.90`). A second run reuses the compiled binaries and the cached CSV under `artifacts/`.

Old `checkpoint.npz` files from the one-step recursive head are not migrated: `load_checkpoint` fails with a rerun-training message.

## Tests

```text
python tests/test_siftcast.py
```

That drives the shipped GRU `train_step` / calibration / checkpoint / direct infer path (including split-boundary, gradient, schema, and forecast-record checks), the C++ simulator (`--self-test` plus a clean capture), and the Rust decoder (`cargo test` plus reconstruct of the clean and noisy captures).

## What you should see

- `sim:` stderr from the C++ encoder (packet counts, drops, bit flips)
- `reconstruct:` stderr from the Rust decoder (trusted vs held ticks, CRC fails)
- `train step k/N loss=...` with the last loss below the first
- `siftcast: calibrate` then `checkpoint wrote artifacts/checkpoint.npz`
- `siftcast: wrote artifacts/forecast.json points=16`
- `budget peak_ws_mb=...` well under 16 GB; training is CPU-only so VRAM stays at the desktop baseline

## Outputs

| path | what |
| --- | --- |
| `artifacts/daily-min-temperatures.csv` | obtained series (same schema as `vendor/`) |
| `artifacts/capture.bin` | Gilbert–Elliott wire dump from C++ |
| `artifacts/truth.jsonl` | clean plant `(seq, x, y)` for oracle eval only |
| `artifacts/series.jsonl` | reconstructed timeline from Rust |
| `artifacts/checkpoint.npz` | trained GRU weights, norm stats, per-horizon `q_h` |
| `artifacts/forecast.json` | one test origin: `x_mean_c`, `x_sigma_c`, `x_lo_c`, `x_hi_c`, `trust` per step |
| `artifacts/metrics.json` | rolling test-block observed vs oracle metrics |
| `artifacts/budget.txt` / `artifacts/ram.txt` | peak working set, file sizes, `nvidia-smi` |

Each `forecast.json` step is a calibrated interval in °C plus a trust probability in `[0, 1]`. When the run is a simulated holdout, steps also include `trust_true`, `x_observed_c`, and `x_oracle_c`. Top-level metadata includes `model_type`, `schema_version`, `nominal_coverage`, `origin_seq`, and `horizon`.

Intervals are **split-conformal-style**, not a coverage guarantee: serial dependence and drift violate exchangeability. See `REPORT.md`.

## Project structure

```text
siftcast.py          obtain + build + train + calibrate + infer (the one command)
sim.cpp              C++ plant, CRC-8 frame codec, Gilbert–Elliott channel
reconstruct/         Rust streaming CRC / reorder / hold reconstruct
vendor/              offline copy of the ~68 KB daily-min-temperatures CSV
tests/test_siftcast.py
artifacts/           runtime outputs (created on run)
build/sim.exe        produced by MSVC
```

## Pipeline

1. **Obtain** Melbourne daily-min-temperatures (~68 KB). If the GitHub raw URL fails, `vendor/daily-min-temperatures.csv` is used.
2. **C++ `sim`** interpolates that series through a forced thermal plant, packs each tick as `A5 5A | seq | t_ms | x | y | crc8`, then applies a two-state Gilbert–Elliott channel (bit flips, drops, duplicates, reorder delay, junk bytes). Writes `truth.jsonl` (clean plant) and `capture.bin` (the wire).
3. **Rust `reconstruct`** hunts sync bytes, verifies CRC-8/SMBUS, reorders by sequence, and emits one JSONL row per seq with `trust=1` on committed packets and `trust=0` on holds for gaps.
4. **Python** trains a 16-hidden GRU on windows of `(x, y, trust)` with a direct `H×3` head (mean, scale, trust logit). Loss is trust-gated Gaussian NLL plus trust BCE, averaged over horizons. After training it fits per-horizon split-conformal-style quantiles on the calibration block, writes `checkpoint.npz`, then emits all `H` steps in one forward pass.

C++, Rust, and Python each sit on that train/infer path. Drop C++ and there is no framed capture; drop Rust and there are no trust labels; drop Python and no weights update.

## Frame

13 bytes, little-endian:

```text
A5 5A | seq u16 | t_ms u32 | x i16 (cK) | y i16 (cK) | crc8
```

CRC-8/SMBUS (poly `0x07`, init `0`, check vector `"123456789"` → `0xF4`) covers `seq` through `y`, not the sync bytes.
