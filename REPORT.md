# Siftcast: trust-gated forecast of a reconstructed serial capture

## Thesis

A tiny sequence model jointly forecasts the next reconstructed sensor readings and packet-trust bits from a CRC-validated, sequence-reordered Gilbert–Elliott serial capture; stripping reconstruction or the trust-gated head collapses the product to ordinary univariate forecasting of a dirty arrival-order dump.

That sentence is the product. A capture-only dump, a CRC library, or a GRU on a clean CSV each leaves a hole the other two languages are there to fill.

## Why this exists

Industrial telemetry does not arrive as a tidy CSV. It arrives as bytes: sync words, sequence numbers, clocks that wander, CRC that sometimes fails, bursts of corruption, packets that show up twice or late. Most tiny forecasting stacks ignore that layer. They either assume a clean series or treat dropouts as generic missing-data imputation.

Siftcast makes the wire the dataset. The labels the trainer is allowed to trust are exactly the ticks a streaming decoder would commit. The model has two jobs: predict the physical reading (with uncertainty), and predict whether each future tick will even be trustworthy. Held decoder values train the trust head only; they are not treated as temperature measurements.

## Method

**Plant (C++).** Melbourne daily minimum temperatures (3,650 days) are interpolated to 4,096 ticks and drive a leaky thermal state `y`, plus a 48-tick forced sinusoid so the series is not a flat climatology. Each tick is a 13-byte frame:

```text
A5 5A | seq u16 LE | t_ms u32 LE | x i16 (cK) | y i16 (cK) | crc8
```

CRC-8/SMBUS (poly `0x07`, init `0`; check `"123456789"` → `0xF4`) covers `seq` through `y`. A two-state Gilbert–Elliott channel then mutates the wire: bit flips, drops, duplicates, delay-reorder, and junk bytes, with a random-walk clock. `--error-scale 0` is a clean codec; the default is the bursty link. The plant also writes `artifacts/truth.jsonl` (`seq`, `x`, `y`) for oracle evaluation only.

**Reconstruct (Rust).** A byte-at-a-time hunter finds `A5 5A`, verifies CRC, drops stale duplicates, and commits in sequence order with a window of 16. Missing seqs become holds: last committed `(x, y)`, `trust = 0`. That JSONL timeline is the only series the trainer sees.

**Learn (Python, CPU).** A 16-hidden GRU reads windows of `(x_norm, y_norm, trust)` and emits all `H=16` horizons in one pass. Each horizon is `(mu, raw_scale, trust_logit)` with `sigma = 1e-3 + softplus(raw_scale)` and `p = sigmoid(logit)`. Time is split **70% train | 15% calibration | 15% test**. Mean/std and weights come from the training block only. Context may look backward across a boundary; targets never cross forward.

Per-horizon loss (averaged over `H` and the batch):

```text
L_h = BCE(o, p) + o * [log(sigma) + 0.5 * ((x - mu) / sigma)^2]
```

Held ticks (`o = 0`) train trust only. SGD with grad clip writes `artifacts/checkpoint.npz` (`schema_version=2`). After training, non-overlapping calibration origins (stride `H`) yield per-horizon scores `abs(x-mu)/sigma` on trusted reconstructed targets. The split-conformal-style order statistic is

```text
k = min(n_h, ceil((n_h + 1) * c))
q_h = sorted(scores_h)[k - 1]
interval = mu ± q_h * sigma
```

with nominal `c = 0.90`. Inference is one GRU pass per origin: no recursive append, no frozen-`y` rollout, no feeding soft trust back as input.

These are **split-conformal-style calibrated intervals**, not a coverage guarantee. Serial dependence and distribution drift violate the usual exchangeability assumption, so finite-sample marginal coverage need not equal `c`.

## Ground truth

Training and calibration labels are reconstructed-stream values with the trust mask. Clean simulator truth is never an input, a normalization source, a training label for held ticks, or a calibration label for held ticks.

- **`observed_*`**: trusted reconstructed test targets — the labels a deployment would have.
- **`oracle_*`**: clean plant temperatures aligned by `seq` across trusted and held ticks — available only for this synthetic plant. A missing seq is an error; evaluation never compares by array index.

Rolling metrics use non-overlapping test origins with stride `H`. `forecast.json` holds one representative origin; aggregates live in `metrics.json`.

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

The twist on a known form is short-horizon probabilistic forecasting **after** a real encoder/channel/decoder, with the decoder’s trust bit as both an input channel and a supervised head.

## C++ / Rust / Python split

Each language does a job the thesis needs. Subprocess + files is the bridge; there is no FFI framework.

| Language | Critical-path job | What breaks if you remove it |
| --- | --- | --- |
| **C++** | Plant, 13-byte frame codec, Gilbert–Elliott wire physics, clean `truth.jsonl` | No capture.bin with CRC, bursts, reorder, or clock drift. Python never sees a real channel. |
| **Rust** | Streaming sync hunt, CRC-8, reorder window, hold-fill trust timeline | No `series.jsonl` with committed vs held ticks. The trainer would have to parse the dirty dump itself and would not know which values were measurements. |
| **Python** | Trust-gated GRU SGD, split-conformal-style calibration, direct infer | No weight updates, no forecast. |

On the recorded run the three jobs actually executed, not merely existed as source:

```text
siftcast: sim (C++)
sim: n=4096 temps=3650 seed=1 error_scale=1.000 bytes_out=50734 dropped=272 flipped=324 dups=58 junk_bursts=92 ticks_bad=605
siftcast: reconstruct (Rust)
reconstruct: bytes=50734 rows=4096 trusted=3500 held=596 crc_fail=279 dups_old=12 sync_hits=3824
siftcast: train (Python GRU)
train step 1/500 loss=1.05095
...
train step 500/500 loss=0.58328
siftcast: calibrate
checkpoint wrote ...\artifacts\checkpoint.npz bytes=20580 first=1.05095 last=0.58328 n_params=1776
```

Clean-path tests (`--error-scale 0`, 64 frames) reconstruct to 64 rows, all `trust=1`. A noisy 256-packet capture produces holds and CRC fails (211 trusted / 45 held / 23 CRC fails in the in-repo test).

## Envelope: 16 GB RAM / 6 GB VRAM

Designed to stay far under both ceilings. Training is CPU numpy; CUDA is not used even though this box has a 6 GB RTX 4050. The only download is the 68 KB CSV (vendored at `vendor/daily-min-temperatures.csv` for offline reruns). No foundation-model weights.

Evidence from the recorded one-command run (`python siftcast.py --n 4096 --steps 500 --seed 1 --horizon 16 --error-scale 1.0 --coverage 0.90`), also written to `artifacts/budget.txt` and `artifacts/ram.txt`:

| Quantity | Recorded run | Ceiling |
| --- | --- | --- |
| Python peak working set | **44.941 MB** (`artifacts/budget.txt`) | 16 GB |
| Python working set at budget | 44.277 MB | 16 GB |
| GPU memory used / total (`nvidia-smi`) | **695 MiB / 6,141 MiB** | 6 GB VRAM |
| CSV download / vendor | 67,921 B | — |
| Capture | 50,734 B | — |
| Checkpoint | 20,580 B (1776 params) | — |
| Forecast JSON | 5,054 B | — |

The 695 MiB VRAM figure is the desktop baseline. The process did not allocate GPU memory. Peak working set is ~0.28% of 16 GB. The heaviest files are a 203,264 B `sim.exe` and a 186,880 B `reconstruct.exe`. A second identical command byte-matched `checkpoint.npz`, `forecast.json`, and `metrics.json` (seed is fixed). Budget/RAM/`nvidia-smi` are machine-dependent and were not compared.

## What the run learned

500 SGD steps on 4,096 reconstructed ticks, window 12, horizon 16, batch 32, lr 0.05. Split `train_end=2867`, `cal_end=3481`, `test_end=4096`. Parameter count **1776** (under 2,000). Calibration score counts per horizon: 35, 36, 37, 35, 33, 36, 34, 34, 36, 35, 33, 36, 36, 34, 32, 35. Rolling evaluation: 38 non-overlapping test origins.

From `artifacts/metrics.json`:

| Metric | Value | n |
| --- | --- | --- |
| Train loss first → last | 1.0509535578361746 → 0.5832755172463033 | 500 steps |
| Oracle RMSE / MAE | 2.949036602711687 / 2.2882606012474205 | 608 |
| Oracle persistence RMSE | 3.7470802055883468 | 608 |
| Oracle interval coverage | 0.930921052631579 | 608 |
| Observed RMSE | 3.0083843182158856 | 526 |
| Observed Gaussian NLL | 2.506996483196001 | 526 |
| Observed interval coverage | 0.9296577946768061 | 526 |
| Observed interval mean width (°C) | 11.400011775037257 | 526 |
| Observed interval score (`alpha=0.10`) | 13.897768326441469 | 526 |
| Train-climatology interval score | 16.586315390231647 | 526 |
| Observed coverage Wilson 95% CI | [0.8032633812569494, 0.9771585570739174] | 38 origins |
| Trust BCE / Brier | 0.4349318187950111 / 0.12993815084921917 | 608 |
| Train-rate climatology Brier | 0.11735113084474427 | 608 |

Oracle RMSE beats oracle persistence (2.949036602711687 < 3.7470802055883468). Observed interval score beats the train-climatology interval baseline (13.897768326441469 < 16.586315390231647). Nominal 0.90 lies inside the origin-clustered Wilson 95% interval. Hits inside one 16-step origin share context, so the Wilson sample size is `n_test_origins=38`, not the 526 pooled steps.

Trust Brier **does not** beat the train-rate climatology Brier on this run (0.12993815084921917 vs 0.11735113084474427). The trust head is still trained and reported; it is not omitted because it lost that comparison. Packet trust on this channel is close to a constant rate (~0.85), and a 16-hidden GRU on a 12-tick window did not improve the Brier score over that rate.

The previous recursive one-step head, evaluated on a single reconstructed holdout origin (holds allowed to count as temperature truth), recorded train loss 1.1824016689898553 → 0.47774296029586705 and holdout MSE 6.0205550695570595 vs persistence 9.420125000000004. That is a different protocol, not a like-for-like RMSE.

Forecast records are task-shaped: each step has `x_mean_c`, `x_sigma_c>0`, ordered `x_lo_c`/`x_hi_c`, and `trust` in `[0, 1]`. The GRU is small; beating persistence on the rolling oracle block is the check that parameters moved on this data, not a SOTA claim.

## Limitations

- Split-conformal-style intervals assume exchangeability. This series is serially dependent and the plant plus channel can drift across the 70/15/15 cut; the Wilson interval is an empirical check on this run, not a proof of valid coverage on other captures.
- Oracle metrics exist only because the simulator wrote `truth.jsonl`. A real capture has `observed_*` only.
- No claim of robustness across arbitrary channels, seeds, or horizons beyond the recorded default command.
- No Bayesian weight posterior, cross-horizon covariance, or Monte Carlo rollout.

## Reproduce

```text
python siftcast.py
python tests/test_siftcast.py
```

Setup, structure, and flags: `README.md`.
