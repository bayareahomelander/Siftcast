# Siftcast: trust-gated forecast of a reconstructed serial capture

## Thesis

A tiny sequence model jointly forecasts the next reconstructed sensor reading and the next packet-trust bit from a CRC-validated, sequence-reordered Gilbert–Elliott serial capture; stripping reconstruction or the trust-gated head collapses the product to ordinary univariate forecasting of a dirty arrival-order dump.

That sentence is the product. A capture-only dump, a CRC library, or a GRU on a clean CSV each leaves a hole the other two languages are there to fill.

## Why this exists

Industrial telemetry does not arrive as a tidy CSV. It arrives as bytes: sync words, sequence numbers, clocks that wander, CRC that sometimes fails, bursts of corruption, packets that show up twice or late. Most tiny forecasting stacks ignore that layer. They either assume a clean series or treat dropouts as generic missing-data imputation.

Siftcast makes the wire the dataset. The labels the trainer is allowed to trust are exactly the ticks a streaming decoder would commit. The model has two jobs: predict the next physical reading, and predict whether the next tick will even be trustworthy. The value loss is multiplied by that trust, so a hold-filled gap does not get to pretend it was a measurement.

## Method

**Plant (C++).** Melbourne daily minimum temperatures (3,650 days) are interpolated to 4,096 ticks and drive a leaky thermal state `y`, plus a 48-tick forced sinusoid so the series is not a flat climatology. Each tick is a 13-byte frame:

```text
A5 5A | seq u16 LE | t_ms u32 LE | x i16 (cK) | y i16 (cK) | crc8
```

CRC-8/SMBUS (poly `0x07`, init `0`; check `"123456789"` → `0xF4`) covers `seq` through `y`. A two-state Gilbert–Elliott channel then mutates the wire: bit flips, drops, duplicates, delay-reorder, and junk bytes, with a random-walk clock. `--error-scale 0` is a clean codec; the default is the bursty link.

**Reconstruct (Rust).** A byte-at-a-time hunter finds `A5 5A`, verifies CRC, drops stale duplicates, and commits in sequence order with a window of 16. Missing seqs become holds: last committed `(x, y)`, `trust = 0`. That JSONL timeline is the only series the trainer sees.

**Learn (Python, CPU).** A 16-hidden GRU (994 parameters) reads windows of `(x_norm, y_norm, trust)` and emits `(x_next, trust_logit)`. Loss is trust-gated value MSE plus BCE on the next trust bit. SGD with grad clip writes `artifacts/checkpoint.npz`. Inference rolls 16 steps and reports holdout MSE against a persistence baseline.

## Negative catalog (not a clone)

| Nearby work | What it is | Why Siftcast is not it |
| --- | --- | --- |
| Chronos, TinyTimeMixer, PatchTST, DLinear | Foundation / clean-CSV forecasters, often with large checkpoints | No packet layer, no CRC/reorder, weights far above this envelope |
| sktime / darts / tsai tutorials | Univariate or covariate TS on prepared tables | Arrival-order dump, not a reconstructed wire |
| Wireshark, Scapy, cantools | Protocol decode | No learning, no trust-gated forecast |
| CRC-8/SMBUS libraries | Checksum primitives | Codec without a plant, channel, or model |
| Gilbert & Kaliaperumal 2018 TAR/TSTM | Node-*reputation* trust in WSNs, AR with exogenous trust, compressed sensing | Compromised-node reputation, not CRC/reorder/GE bit errors, not a joint integrity head on a framed capture |
| Remote estimation over Gilbert–Elliott channels (Kalman / team-theory papers) | Optimal linear estimators under a Markov drop channel | No CRC framing, no learned GRU, no reconstruct-then-train product |
| Multi-rate fusion over GE channels (DSP 2025) | Covariance-bounded fusion estimators | Filter theory, not a trained two-head sequence model on a decoded timeline |
| TRUST-RNN (IEEE) | Bayesian weight uncertainty through an RNN | Parameter posterior, not packet-integrity labels from a decoder |
| TFLite Micro / ONNX “tiny sensor” demos | Classify clean features or wrap a famous checkpoint | No from-scratch train on a reconstructed capture |

The twist on a known form is ordinary short-horizon forecasting **after** a real encoder/channel/decoder, with the decoder’s trust bit as both an input channel and a supervised head.

## C++ / Rust / Python split

Each language does a job the thesis needs. Subprocess + files is the bridge; there is no FFI framework.

| Language | Critical-path job | What breaks if you remove it |
| --- | --- | --- |
| **C++** | Plant, 13-byte frame codec, Gilbert–Elliott wire physics | No capture.bin with CRC, bursts, reorder, or clock drift. Python never sees a real channel. |
| **Rust** | Streaming sync hunt, CRC-8, reorder window, hold-fill trust timeline | No `series.jsonl` with committed vs held ticks. The trainer would have to parse the dirty dump itself and would not know which values were measurements. |
| **Python** | Trust-gated GRU SGD, checkpoint, 16-step infer | No weight updates, no forecast. |

On the recorded run the three jobs actually executed, not merely existed as source:

```text
siftcast: sim (C++)
sim: n=4096 temps=3650 seed=1 error_scale=1.000 bytes_out=50734 dropped=272 flipped=324 dups=58 junk_bursts=92 ticks_bad=605
siftcast: reconstruct (Rust)
reconstruct: bytes=50734 rows=4096 trusted=3500 held=596 crc_fail=279 dups_old=12 sync_hits=3824
siftcast: train (Python GRU)
train step 1/500 loss=1.18240
...
train step 500/500 loss=0.47774
checkpoint wrote ...\artifacts\checkpoint.npz bytes=11860 first=1.18240 last=0.47774
```

Clean-path tests (`--error-scale 0`, 64 frames) reconstruct to 64 rows, all `trust=1`. A noisy 256-packet capture produces holds and CRC fails (211 trusted / 45 held / 23 CRC fails in the in-repo test).

## Envelope: 16 GB RAM / 6 GB VRAM

Designed to stay far under both ceilings. Training is CPU numpy; CUDA is not used even though this box has a 6 GB RTX 4050. The only download is the 68 KB CSV (vendored at `vendor/daily-min-temperatures.csv` for offline reruns). No foundation-model weights.

Evidence from the actual one-command run on this machine (`python siftcast.py --n 4096 --steps 500 --seed 1 --horizon 16`), also written to `artifacts/budget.txt` and `artifacts/ram.txt`:

| Quantity | Run 1 | Ceiling |
| --- | --- | --- |
| Python peak working set | **43.418 MB** (`run-1.log`) | 16 GB |
| Python working set at budget | 42.191 MB (`run-1.log`) | 16 GB |
| OS visible RAM / free after run | 15,987 MB / 7,510 MB | 16 GB class |
| GPU memory used / total (`nvidia-smi`) | **744 MiB / 6,141 MiB** | 6 GB VRAM |
| CSV download / vendor | 67,921 B | — |
| Capture | 50,734 B | — |
| Checkpoint | 11,860 B (994 params) | — |
| Forecast JSON | 2,705 B | — |

The 744 MiB VRAM figure is the desktop baseline (same before and after train). The process did not allocate GPU memory. Peak working set is ~0.27% of 16 GB. The heaviest files are a 203 KB `sim.exe` and a 187 KB `reconstruct.exe`. A second identical command succeeded with peak working set **43.246 MB** / 45,346,816 bytes (`artifacts/ram.txt`, `artifacts/budget.txt`) and the same checkpoint size, losses, and forecast metrics (seed is fixed).

## What the run learned

500 SGD steps on 4,096 reconstructed ticks, window 12, batch 32, lr 0.05:

- Train loss 1.18240 → 0.47774
- 16-step holdout MSE **6.021** vs persistence **9.420**
- Trust BCE 0.267; mean predicted trust 0.860 vs mean true trust 0.938 on that holdout window

Forecast records are task-shaped: each step has `x_c` (deg C) and `trust` in `artifacts/forecast.json`. The GRU is small and mean-reverting at long horizon; beating persistence on this holdout is the check that parameters actually moved on this data, not a SOTA claim.

## Reproduce

```text
python siftcast.py
python tests/test_siftcast.py
```

Setup, structure, and flags: `README.md`.
