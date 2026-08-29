"""Siftcast: trust-gated joint forecast of reconstructed serial telemetry.

C++ simulates a framed Gilbert-Elliott capture; Rust reconstructs a
CRC-validated reordered timeline; this file trains and infers a tiny GRU
with a direct multi-horizon censored head and split-conformal-style intervals.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
VENDOR_CSV = ROOT / "vendor" / "daily-min-temperatures.csv"
DATA_CSV = ROOT / "artifacts" / "daily-min-temperatures.csv"
CSV_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
CAPTURE = ROOT / "artifacts" / "capture.bin"
TRUTH = ROOT / "artifacts" / "truth.jsonl"
SERIES = ROOT / "artifacts" / "series.jsonl"
CKPT = ROOT / "artifacts" / "checkpoint.npz"
FORECAST = ROOT / "artifacts" / "forecast.json"
METRICS = ROOT / "artifacts" / "metrics.json"
BUDGET = ROOT / "artifacts" / "budget.txt"
SIM_EXE = ROOT / "build" / "sim.exe"
RECON_EXE = ROOT / "reconstruct" / "target" / "release" / "reconstruct.exe"

DIN = 3
DH = 16
WIN = 12
HORIZON = 16
SIGMA_FLOOR = 1e-3
SCHEMA_VERSION = 2
MODEL_TYPE = "direct_multi_horizon_gru"
COVERAGE = 0.90
TRAIN_FRAC = 0.70
CAL_FRAC = 0.15
CKPT_RERUN = (
    "incompatible checkpoint (need schema_version=2 direct_multi_horizon_gru "
    "with calibration quantiles); rerun training: python siftcast.py"
)

# --- native build / obtain -------------------------------------------------


def obtain_csv(dest: Path = DATA_CSV) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    try:
        urllib.request.urlretrieve(CSV_URL, dest)
        if dest.stat().st_size == 0:
            raise RuntimeError("empty download")
    except Exception:
        if not VENDOR_CSV.exists():
            raise
        shutil.copyfile(VENDOR_CSV, dest)
    return dest


def find_vcvars() -> Path:
    vswhere = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")
    if vswhere.exists():
        out = subprocess.check_output(
            [
                str(vswhere),
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ],
            text=True,
        ).strip()
        cand = Path(out) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
        if cand.exists():
            return cand
    fallback = Path(
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    )
    if fallback.exists():
        return fallback
    raise RuntimeError("MSVC vcvars64.bat not found; install VS Build Tools")


def build_sim() -> Path:
    SIM_EXE.parent.mkdir(parents=True, exist_ok=True)
    src = ROOT / "sim.cpp"
    if SIM_EXE.exists() and SIM_EXE.stat().st_mtime >= src.stat().st_mtime:
        return SIM_EXE
    vcvars = find_vcvars()
    # vcvars is its own argv. One string `call "…\vcvars64.bat" && cl` is
    # list2cmdline-escaped (path has spaces) and cmd cannot find the bat.
    subprocess.check_call(
        [
            "cmd",
            "/c",
            "call",
            str(vcvars),
            "&&",
            "cl",
            "/nologo",
            "/O2",
            "/EHsc",
            "/std:c++17",
            "/Fo:build\\",
            "/Fe:build\\sim.exe",
            "sim.cpp",
        ],
        cwd=str(ROOT),
    )
    if not SIM_EXE.exists():
        raise RuntimeError("sim.exe missing after compile")
    return SIM_EXE


def build_reconstruct() -> Path:
    subprocess.check_call(
        ["cargo", "build", "--release", "--manifest-path", str(ROOT / "reconstruct" / "Cargo.toml")]
    )
    if not RECON_EXE.exists():
        raise RuntimeError("reconstruct.exe missing after cargo build")
    return RECON_EXE


def build_native() -> None:
    build_sim()
    build_reconstruct()


def run_sim(n: int, seed: int, error_scale: float, in_csv: Path, out_bin: Path, truth: Path) -> None:
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [
            str(build_sim()),
            "--in",
            str(in_csv),
            "--out",
            str(out_bin),
            "--truth",
            str(truth),
            "--n",
            str(n),
            "--seed",
            str(seed),
            "--error-scale",
            str(error_scale),
        ]
    )


def run_reconstruct(inp: Path, outp: Path) -> None:
    outp.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([str(build_reconstruct()), "--in", str(inp), "--out", str(outp)])


# ponytail: numpy GRU on CPU; autograd/CUDA only if H or T outgrow this envelope
# --- GRU (numpy, CPU) ------------------------------------------------------


def head_width(horizon: int) -> int:
    return 3 * int(horizon)


def param_count(params: dict) -> int:
    return int(sum(np.asarray(v).size for v in params.values()))


def _sig(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def _softplus(z: np.ndarray) -> np.ndarray:
    """log(1+exp(z)) stable for large |z|."""
    z = np.asarray(z, dtype=np.float64)
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))


def _softplus_inv(y: np.ndarray) -> np.ndarray:
    """Inverse of softplus; y must be > 0."""
    y = np.asarray(y, dtype=np.float64)
    out = np.empty_like(y, dtype=np.float64)
    big = y > 20.0
    out = np.where(big, y, np.log(np.expm1(np.minimum(y, 20.0))))
    return out


def decode_head(y: np.ndarray, horizon: int):
    """y (3H,) -> mu, sigma, p each (H,)."""
    raw = np.asarray(y, dtype=np.float64).reshape(int(horizon), 3)
    mu = raw[:, 0]
    raw_scale = raw[:, 1]
    logit = raw[:, 2]
    sigma = SIGMA_FLOOR + _softplus(raw_scale)
    p = _sig(logit)
    return mu, sigma, p, raw_scale, logit


def init_params(rng: np.random.Generator, din: int = DIN, dh: int = DH, horizon: int = HORIZON) -> dict:
    dout = head_width(horizon)

    def w(a, b):
        return rng.normal(0.0, 1.0 / np.sqrt(a + b), (a, b)).astype(np.float64)

    bout = np.zeros(dout, dtype=np.float64)
    # scale head: sigma ≈ 1 in normalized units
    bout[1::3] = float(_softplus_inv(np.array(1.0 - SIGMA_FLOOR)))
    return {
        "Wz": w(din, dh),
        "Uz": w(dh, dh),
        "bz": np.zeros(dh, dtype=np.float64),
        "Wr": w(din, dh),
        "Ur": w(dh, dh),
        "br": np.zeros(dh, dtype=np.float64),
        "Wn": w(din, dh),
        "Un": w(dh, dh),
        "bn": np.zeros(dh, dtype=np.float64),
        "Wout": w(dh, dout),
        "bout": bout,
    }


def copy_params(p: dict) -> dict:
    return {k: np.array(v, copy=True) for k, v in p.items()}


def zeros_like_params(p: dict) -> dict:
    return {k: np.zeros_like(v) for k, v in p.items()}


def gru_forward(p: dict, X: np.ndarray):
    """X: (T, din) -> y (3H,), cache. Recurrence unchanged below the linear head."""
    T = X.shape[0]
    dh = p["bz"].shape[0]
    h = np.zeros(dh)
    cache = []
    for t in range(T):
        x = X[t]
        z = _sig(x @ p["Wz"] + h @ p["Uz"] + p["bz"])
        r = _sig(x @ p["Wr"] + h @ p["Ur"] + p["br"])
        n = np.tanh(x @ p["Wn"] + (r * h) @ p["Un"] + p["bn"])
        h_new = (1.0 - z) * n + z * h
        cache.append((x, h, z, r, n, h_new))
        h = h_new
    y = h @ p["Wout"] + p["bout"]
    return y, cache


def _loss_from_y(y: np.ndarray, target: np.ndarray) -> float:
    """Mean over horizons of BCE(o,p) + o*[log(sigma)+0.5*((x-mu)/sigma)^2]."""
    target = np.asarray(target, dtype=np.float64)
    H = int(target.shape[0])
    mu, sigma, p, _, _ = decode_head(y, H)
    x = target[:, 0]
    o = np.clip(target[:, 1], 0.0, 1.0)
    pc = np.clip(p, 1e-8, 1.0 - 1e-8)
    bce = -(o * np.log(pc) + (1.0 - o) * np.log(1.0 - pc))
    z = (x - mu) / sigma
    gauss = np.log(sigma) + 0.5 * z * z
    return float(np.mean(bce + o * gauss))


def forward_loss(p: dict, X: np.ndarray, target: np.ndarray) -> float:
    y, _ = gru_forward(p, X)
    return _loss_from_y(y, target)


def loss_and_grad(p: dict, X: np.ndarray, target: np.ndarray):
    y, cache = gru_forward(p, X)
    target = np.asarray(target, dtype=np.float64)
    H = int(target.shape[0])
    loss = _loss_from_y(y, target)
    mu, sigma, p_hat, raw_scale, _logit = decode_head(y, H)
    x = target[:, 0]
    o = np.clip(target[:, 1], 0.0, 1.0)
    invH = 1.0 / float(H)
    z = (x - mu) / sigma
    dmu = invH * o * (mu - x) / (sigma * sigma)
    dsigma = invH * o * (1.0 - z * z) / sigma
    d_raw_scale = dsigma * _sig(raw_scale)
    dlogit = invH * (p_hat - o)
    dy = np.empty(3 * H, dtype=np.float64)
    dy[0::3] = dmu
    dy[1::3] = d_raw_scale
    dy[2::3] = dlogit

    g = zeros_like_params(p)
    h_last = cache[-1][5]
    g["Wout"] += np.outer(h_last, dy)
    g["bout"] += dy
    dh = p["Wout"] @ dy

    for x_t, h_prev, z_g, r, n, h_new in reversed(cache):
        # h_new = (1-z)*n + z*h_prev
        dz = dh * (h_prev - n)
        dn = dh * (1.0 - z_g)
        dh_prev = dh * z_g

        dn_pre = dn * (1.0 - n * n)
        g["Wn"] += np.outer(x_t, dn_pre)
        g["Un"] += np.outer(r * h_prev, dn_pre)
        g["bn"] += dn_pre
        dx = p["Wn"] @ dn_pre
        d_rh = p["Un"] @ dn_pre
        dr = d_rh * h_prev
        dh_prev = dh_prev + d_rh * r

        dz_pre = dz * z_g * (1.0 - z_g)
        g["Wz"] += np.outer(x_t, dz_pre)
        g["Uz"] += np.outer(h_prev, dz_pre)
        g["bz"] += dz_pre
        dx = dx + p["Wz"] @ dz_pre
        dh_prev = dh_prev + p["Uz"] @ dz_pre

        dr_pre = dr * r * (1.0 - r)
        g["Wr"] += np.outer(x_t, dr_pre)
        g["Ur"] += np.outer(h_prev, dr_pre)
        g["br"] += dr_pre
        dh_prev = dh_prev + p["Ur"] @ dr_pre
        _ = dx  # input has no params
        dh = dh_prev
    return loss, g


def clip_grads(g: dict, maxn: float) -> None:
    tot = 0.0
    for v in g.values():
        tot += float(np.sum(v * v))
    n = np.sqrt(tot)
    if n > maxn and n > 0:
        s = maxn / n
        for k in g:
            g[k] *= s


def train_step(params: dict, batch_x: np.ndarray, batch_y: np.ndarray, lr: float, clip: float = 5.0):
    """One SGD step. batch_x (B,T,din), batch_y (B,H,2). Returns (params, mean_loss)."""
    B = batch_x.shape[0]
    acc = zeros_like_params(params)
    total = 0.0
    for i in range(B):
        loss, g = loss_and_grad(params, batch_x[i], batch_y[i])
        total += loss
        for k in acc:
            acc[k] += g[k]
    scale = 1.0 / max(B, 1)
    for k in acc:
        acc[k] *= scale
    clip_grads(acc, clip)
    for k in params:
        params[k] = params[k] - lr * acc[k]
    return params, total * scale


def save_checkpoint(path: Path, params: dict, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in params.items()}
    payload["_meta"] = np.array(json.dumps(meta))
    np.savez(path, **payload)


def _validate_checkpoint(params: dict, meta: object) -> dict:
    if not isinstance(meta, dict):
        raise RuntimeError(CKPT_RERUN)
    try:
        schema = int(meta.get("schema_version", 0))
    except (TypeError, ValueError):
        raise RuntimeError(CKPT_RERUN) from None
    if schema != SCHEMA_VERSION or meta.get("model_type") != MODEL_TYPE:
        raise RuntimeError(CKPT_RERUN)
    try:
        horizon = int(meta["horizon"])
        win = int(meta["win"])
        coverage = float(meta["coverage"])
    except (KeyError, TypeError, ValueError):
        raise RuntimeError(CKPT_RERUN) from None
    if horizon < 1 or win < 1 or not (0.0 < coverage < 1.0):
        raise RuntimeError(CKPT_RERUN)
    dout = head_width(horizon)
    if "Wout" not in params or np.asarray(params["Wout"]).ndim != 2:
        raise RuntimeError(CKPT_RERUN)
    if int(params["Wout"].shape[1]) != dout:
        raise RuntimeError(CKPT_RERUN)
    if int(np.asarray(params["bout"]).reshape(-1).shape[0]) != dout:
        raise RuntimeError(CKPT_RERUN)
    q = meta.get("q")
    if not isinstance(q, list) or len(q) != horizon:
        raise RuntimeError(CKPT_RERUN)
    if "mean" not in meta or "std" not in meta:
        raise RuntimeError(CKPT_RERUN)
    split = meta.get("split")
    if not isinstance(split, dict):
        raise RuntimeError(CKPT_RERUN)
    for key in ("train_end", "cal_end", "test_end", "n"):
        if key not in split:
            raise RuntimeError(CKPT_RERUN)
    return meta


def load_checkpoint(path: Path):
    data = np.load(path, allow_pickle=True)
    params = {k: data[k] for k in data.files if k != "_meta"}
    try:
        meta = json.loads(str(data["_meta"]))
    except Exception:
        raise RuntimeError(CKPT_RERUN) from None
    meta = _validate_checkpoint(params, meta)
    return params, meta


# --- series / splits / train / calibrate / infer ---------------------------


def load_series(path: Path) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"empty series {path}")
    return rows


def load_truth_map(path: Path) -> dict:
    """seq -> clean plant temperature. Never used as a training label."""
    rows = load_series(path)
    out = {}
    for r in rows:
        if "seq" not in r or "x" not in r:
            raise RuntimeError(f"truth row missing seq/x: {r!r}")
        out[int(r["seq"])] = float(r["x"])
    return out


def oracle_x(truth_by_seq: dict, seq: int) -> float:
    if seq not in truth_by_seq:
        raise RuntimeError(f"truth.jsonl missing seq {seq}; cannot align oracle evaluation")
    return float(truth_by_seq[seq])


def series_matrix(rows: list, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    X = np.zeros((len(rows), DIN), dtype=np.float64)
    for i, r in enumerate(rows):
        X[i, 0] = (r["x"] - mean[0]) / std[0]
        X[i, 1] = (r["y"] - mean[1]) / std[1]
        X[i, 2] = float(r["trust"])
    return X


def trusted_moments(rows: list) -> tuple:
    xs = [r["x"] for r in rows if r.get("trust", 0) >= 0.5]
    ys = [r["y"] for r in rows if r.get("trust", 0) >= 0.5]
    if len(xs) < 8:
        xs = [r["x"] for r in rows]
        ys = [r["y"] for r in rows]
    mean = np.array([float(np.mean(xs)), float(np.mean(ys))], dtype=np.float64)
    std = np.array([float(np.std(xs)), float(np.std(ys))], dtype=np.float64)
    std = np.maximum(std, 1e-3)
    return mean, std


def split_boundaries(n: int, win: int, horizon: int) -> dict:
    """Contiguous 70% train | 15% calibration | 15% test. Targets cannot cross blocks."""
    n = int(n)
    win = int(win)
    horizon = int(horizon)
    if win < 1 or horizon < 1:
        raise ValueError("win and horizon must be >= 1")
    train_end = int(TRAIN_FRAC * n)
    cal_end = int((TRAIN_FRAC + CAL_FRAC) * n)
    test_end = n
    if train_end < win + horizon:
        raise ValueError(
            f"train block too short for win={win} horizon={horizon}: train_end={train_end} n={n}"
        )
    if cal_end - train_end < horizon:
        raise ValueError(
            f"calibration block too short for horizon={horizon}: cal_len={cal_end - train_end} n={n}"
        )
    if test_end - cal_end < horizon:
        raise ValueError(
            f"test block too short for horizon={horizon}: test_len={test_end - cal_end} n={n}"
        )
    return {
        "n": test_end,
        "win": win,
        "horizon": horizon,
        "train_end": train_end,
        "cal_end": cal_end,
        "test_end": test_end,
    }


def iter_origins(n: int, win: int, block_start: int, block_end: int, horizon: int, stride: int = 1) -> list:
    """Origins t whose targets [t, t+H) lie in [block_start, block_end). Context may look back."""
    if stride < 1:
        raise ValueError("stride must be >= 1")
    end = min(int(block_end), int(n))
    t0 = max(int(win), int(block_start))
    t1 = end - int(horizon)  # inclusive last origin
    if t1 < t0:
        return []
    return list(range(t0, t1 + 1, int(stride)))


def make_windows(
    X: np.ndarray,
    win: int,
    horizon: int,
    t_lo: int,
    t_hi: int,
    stride: int = 1,
):
    """Return (N, win, 3) inputs, (N, H, 2) targets [x_norm, trust], origin indices."""
    origins = iter_origins(len(X), win, t_lo, t_hi, horizon, stride)
    if not origins:
        raise ValueError(
            f"no windows in block [{t_lo},{t_hi}) win={win} horizon={horizon} n={len(X)}"
        )
    xs = np.stack([X[t - win : t] for t in origins], axis=0)
    ys = np.stack(
        [np.column_stack((X[t : t + horizon, 0], X[t : t + horizon, 2])) for t in origins],
        axis=0,
    )
    return xs, ys, np.asarray(origins, dtype=np.int64)


def conformal_quantile(scores: np.ndarray, coverage: float) -> float:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    n = int(scores.size)
    if n == 0:
        raise ValueError("conformal quantile needs at least one score")
    if not (0.0 < float(coverage) < 1.0):
        raise ValueError(f"coverage must be in (0,1), got {coverage}")
    k = min(n, int(np.ceil((n + 1) * float(coverage))))
    k = max(k, 1)
    s = np.sort(scores)
    return float(s[k - 1])


def interval_score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    width = hi - lo
    extra = (2.0 / float(alpha)) * (np.maximum(lo - y, 0.0) + np.maximum(y - hi, 0.0))
    return width + extra


def clustered_coverage_interval(hit_counts, total_counts, z: float = 1.959963984540054) -> tuple:
    """Origin-cluster-robust normal CI for pooled coverage p = sum(h_i)/sum(n_i).

    Returns (p, lo, hi, n_origins). lo/hi are None when fewer than two
    non-empty origins exist (point estimate is still returned).
    """
    hits = np.asarray(hit_counts, dtype=np.float64).reshape(-1)
    totals = np.asarray(total_counts, dtype=np.float64).reshape(-1)
    if hits.shape != totals.shape:
        raise ValueError("hit_counts and total_counts must have the same length")
    if np.any(~np.isfinite(hits)) or np.any(~np.isfinite(totals)):
        raise ValueError("hit and total counts must be finite")
    if np.any(hits < 0) or np.any(totals < 0):
        raise ValueError("hit and total counts must be non-negative")
    if np.any(hits > totals):
        raise ValueError("hits cannot exceed totals")
    nonempty = totals > 0
    h = hits[nonempty]
    n = totals[nonempty]
    g = int(n.size)
    n_sum = float(np.sum(n))
    if n_sum <= 0:
        raise ValueError("clustered coverage CI requires at least one observation")
    p = float(np.sum(h) / n_sum)
    if g < 2:
        return p, None, None, g
    u = h - p * n
    se = math.sqrt((g / (g - 1.0)) * float(np.sum(u * u))) / n_sum
    lo = float(min(1.0, max(0.0, p - z * se)))
    hi = float(min(1.0, max(0.0, p + z * se)))
    return p, lo, hi, g


def predict_origin(params: dict, X: np.ndarray, t: int, win: int, horizon: int):
    """One GRU pass on X[t-win:t] -> mu, sigma, p each (H,). No recursive rollout."""
    ctx = X[t - win : t]
    y, _ = gru_forward(params, ctx)
    mu, sigma, p, _, _ = decode_head(y, horizon)
    return mu, sigma, p


def calibrate(params: dict, X: np.ndarray, splits: dict, coverage: float, win: int, horizon: int):
    origins = iter_origins(
        splits["n"], win, splits["train_end"], splits["cal_end"], horizon, stride=horizon
    )
    if not origins:
        raise RuntimeError("no calibration origins; need a longer series")
    scores = [[] for _ in range(horizon)]
    for t in origins:
        mu, sigma, _p = predict_origin(params, X, t, win, horizon)
        for h in range(horizon):
            if X[t + h, 2] >= 0.5:
                scores[h].append(abs(X[t + h, 0] - mu[h]) / sigma[h])
    q = np.zeros(horizon, dtype=np.float64)
    counts = np.zeros(horizon, dtype=np.int64)
    for h in range(horizon):
        if not scores[h]:
            raise RuntimeError(
                f"no trusted calibration targets at horizon {h + 1}; "
                "cannot form split-conformal-style intervals"
            )
        counts[h] = len(scores[h])
        q[h] = conformal_quantile(np.asarray(scores[h], dtype=np.float64), coverage)
    return q, counts


def fit(
    rows: list,
    steps: int,
    seed: int,
    ckpt: Path,
    lr: float = 0.05,
    batch: int = 32,
    win: int = WIN,
    horizon: int = HORIZON,
    coverage: float = COVERAGE,
):
    splits = split_boundaries(len(rows), win, horizon)
    train_rows = rows[: splits["train_end"]]
    mean, std = trusted_moments(train_rows)
    X = series_matrix(rows, mean, std)
    Wx, Wy, _orig = make_windows(X, win, horizon, 0, splits["train_end"], stride=1)
    rng = np.random.Generator(np.random.PCG64(seed))
    params = init_params(rng, horizon=horizon)
    losses = []
    idx = np.arange(len(Wx))
    for step in range(steps):
        rng.shuffle(idx)
        bix = idx[: min(batch, len(idx))]
        params, loss = train_step(params, Wx[bix], Wy[bix], lr)
        losses.append(float(loss))
        if step == 0 or (step + 1) % max(1, steps // 10) == 0 or step + 1 == steps:
            print(f"train step {step+1}/{steps} loss={loss:.5f}", flush=True)
        if not math.isfinite(losses[-1]):
            raise RuntimeError(f"non-finite train loss at step {step + 1}")
    print("siftcast: calibrate", flush=True)
    q, cal_counts = calibrate(params, X, splits, coverage, win, horizon)
    if not np.all(np.isfinite(q)):
        raise RuntimeError("non-finite calibration quantiles")
    n_par = param_count(params)
    meta = {
        "model_type": MODEL_TYPE,
        "schema_version": SCHEMA_VERSION,
        "mean": [float(mean[0]), float(mean[1])],
        "std": [float(std[0]), float(std[1])],
        "win": int(win),
        "din": DIN,
        "dh": int(params["bz"].shape[0]),
        "horizon": int(horizon),
        "dout": head_width(horizon),
        "sigma_floor": float(SIGMA_FLOOR),
        "coverage": float(coverage),
        "q": [float(v) for v in q],
        "cal_score_counts": [int(v) for v in cal_counts],
        "n_params": int(n_par),
        "steps": int(steps),
        "seed": int(seed),
        "split": {
            "train_end": int(splits["train_end"]),
            "cal_end": int(splits["cal_end"]),
            "test_end": int(splits["test_end"]),
            "n": int(splits["n"]),
        },
        "n": int(splits["n"]),
        "train_loss_first": float(losses[0]),
        "train_loss_last": float(losses[-1]),
    }
    save_checkpoint(ckpt, params, meta)
    print(
        f"checkpoint wrote {ckpt} bytes={ckpt.stat().st_size} first={losses[0]:.5f} last={losses[-1]:.5f} "
        f"n_params={n_par} q_min={float(np.min(q)):.4f} q_max={float(np.max(q)):.4f}",
        flush=True,
    )
    return params, meta, losses, X, splits


def _denorm_mu_sigma(mu, sigma, mean, std):
    mu_c = mu * std[0] + mean[0]
    sigma_c = sigma * std[0]
    return mu_c, sigma_c


def _records_for_origins(params, meta, rows, X, origins, truth_by_seq):
    mean = np.array(meta["mean"], dtype=np.float64)
    std = np.array(meta["std"], dtype=np.float64)
    win = int(meta["win"])
    horizon = int(meta["horizon"])
    q = np.array(meta["q"], dtype=np.float64)
    n = len(rows)
    recs = []
    for t in origins:
        t = int(t)
        if t < win or t > n:
            raise ValueError(
                f"origin_t={t} out of range; need {win} <= origin_t <= {n} "
                f"(win={win}, len(rows)={n})"
            )
        mu, sigma, p = predict_origin(params, X, t, win, horizon)
        mu_c, sig_c = _denorm_mu_sigma(mu, sigma, mean, std)
        lo_c = mu_c - q * sig_c
        hi_c = mu_c + q * sig_c
        persist = float(rows[t - 1]["x"])
        for h in range(horizon):
            rec = {
                "origin_t": t,
                "step": h + 1,
                "x_mean_c": float(mu_c[h]),
                "x_sigma_c": float(sig_c[h]),
                "x_lo_c": float(lo_c[h]),
                "x_hi_c": float(hi_c[h]),
                "trust": float(p[h]),
            }
            idx = t + h
            if idx < n:
                row = rows[idx]
                seq = int(row["seq"])
                rec["seq"] = seq
                rec["trust_true"] = float(row["trust"])
                rec["observed"] = rec["trust_true"] >= 0.5
                rec["persist_c"] = persist
                if rec["observed"]:
                    rec["x_observed_c"] = float(row["x"])
                if truth_by_seq is not None:
                    rec["x_oracle_c"] = oracle_x(truth_by_seq, seq)
            recs.append(rec)
    return recs


def _public_step(rec: dict) -> dict:
    out = {
        "step": rec["step"],
        "x_mean_c": rec["x_mean_c"],
        "x_sigma_c": rec["x_sigma_c"],
        "x_lo_c": rec["x_lo_c"],
        "x_hi_c": rec["x_hi_c"],
        "trust": rec["trust"],
    }
    for k in ("trust_true", "x_observed_c", "x_oracle_c"):
        if k in rec:
            out[k] = rec[k]
    return out


def infer_forecast(params: dict, meta: dict, rows: list, truth_by_seq=None, origin_t=None) -> dict:
    """Direct H-step forecast from one origin (end of stream by default)."""
    mean = np.array(meta["mean"], dtype=np.float64)
    std = np.array(meta["std"], dtype=np.float64)
    horizon = int(meta["horizon"])
    X = series_matrix(rows, mean, std)
    if origin_t is None:
        origin_t = len(rows)
    origin_t = int(origin_t)
    recs = _records_for_origins(params, meta, rows, X, [origin_t], truth_by_seq)
    preds = [_public_step(r) for r in recs]
    origin_seq = int(rows[origin_t - 1]["seq"])
    return {
        "units": "deg_C",
        "model_type": MODEL_TYPE,
        "schema_version": SCHEMA_VERSION,
        "nominal_coverage": float(meta["coverage"]),
        "horizon": horizon,
        "origin_seq": origin_seq,
        "forecast": preds,
    }


def _train_climatology(train_rows: list, coverage: float):
    xs = [float(r["x"]) for r in train_rows if r.get("trust", 0) >= 0.5]
    if len(xs) < 2:
        xs = [float(r["x"]) for r in train_rows]
    mu = float(np.mean(xs))
    sig = float(max(np.std(xs), 1e-3))
    scores = np.abs(np.asarray(xs, dtype=np.float64) - mu) / sig
    q = conformal_quantile(scores, coverage)
    p_trust = float(np.mean([float(r["trust"]) for r in train_rows]))
    return mu, sig, q, p_trust


def evaluate_rolling(params: dict, meta: dict, rows: list, truth_by_seq: dict) -> dict:
    """Non-overlapping test-block origins; observed vs seq-aligned oracle metrics."""
    mean = np.array(meta["mean"], dtype=np.float64)
    std = np.array(meta["std"], dtype=np.float64)
    win = int(meta["win"])
    horizon = int(meta["horizon"])
    coverage = float(meta["coverage"])
    alpha = 1.0 - coverage
    split = meta["split"]
    X = series_matrix(rows, mean, std)
    origins = iter_origins(len(rows), win, split["cal_end"], split["test_end"], horizon, stride=horizon)
    if not origins:
        raise RuntimeError("no test origins for evaluation")
    recs = _records_for_origins(params, meta, rows, X, origins, truth_by_seq)
    for r in recs:
        loc = f"origin_t={r.get('origin_t')} step={r.get('step')}"
        for k in ("trust_true", "observed", "persist_c", "x_oracle_c"):
            if k not in r:
                raise RuntimeError(f"evaluate_rolling missing {k} on {loc}")
        if r["observed"] and "x_observed_c" not in r:
            raise RuntimeError(f"evaluate_rolling missing x_observed_c on trusted {loc}")

    obs = [r for r in recs if r["observed"]]
    if not obs:
        raise RuntimeError("no trusted reconstructed test targets for observed metrics")

    oracle_err = np.array([r["x_mean_c"] - r["x_oracle_c"] for r in recs], dtype=np.float64)
    oracle_rmse = float(np.sqrt(np.mean(oracle_err ** 2)))
    oracle_mae = float(np.mean(np.abs(oracle_err)))
    persist_err = np.array([r["persist_c"] - r["x_oracle_c"] for r in recs], dtype=np.float64)
    persist_rmse = float(np.sqrt(np.mean(persist_err ** 2)))
    oracle_cov = float(
        np.mean([(r["x_lo_c"] <= r["x_oracle_c"] <= r["x_hi_c"]) for r in recs])
    )

    obs_err = np.array([r["x_mean_c"] - r["x_observed_c"] for r in obs], dtype=np.float64)
    obs_rmse = float(np.sqrt(np.mean(obs_err ** 2)))
    nll_const = 0.5 * math.log(2.0 * math.pi)
    obs_nll = float(
        np.mean(
            [
                nll_const
                + math.log(r["x_sigma_c"])
                + 0.5 * ((r["x_observed_c"] - r["x_mean_c"]) / r["x_sigma_c"]) ** 2
                for r in obs
            ]
        )
    )
    obs_n = len(obs)
    hit_counts = []
    total_counts = []
    for t in origins:
        recs_t = [r for r in recs if r["origin_t"] == t and r["observed"]]
        n_i = len(recs_t)
        h_i = sum(1 for r in recs_t if r["x_lo_c"] <= r["x_observed_c"] <= r["x_hi_c"])
        hit_counts.append(h_i)
        total_counts.append(n_i)
    obs_cov, cluster_lo, cluster_hi, cluster_g = clustered_coverage_interval(
        hit_counts, total_counts
    )
    widths = np.array([r["x_hi_c"] - r["x_lo_c"] for r in obs], dtype=np.float64)
    mean_width = float(np.mean(widths))
    iscores = interval_score(
        np.array([r["x_observed_c"] for r in obs]),
        np.array([r["x_lo_c"] for r in obs]),
        np.array([r["x_hi_c"] for r in obs]),
        alpha,
    )
    mean_iscore = float(np.mean(iscores))

    p = np.clip(np.array([r["trust"] for r in recs], dtype=np.float64), 1e-8, 1.0 - 1e-8)
    o = np.array([r["trust_true"] for r in recs], dtype=np.float64)
    trust_bce = float(-np.mean(o * np.log(p) + (1.0 - o) * np.log(1.0 - p)))
    trust_brier = float(np.mean((np.array([r["trust"] for r in recs]) - o) ** 2))

    train_rows = rows[: split["train_end"]]
    clim_mu, clim_sig, clim_q, clim_p = _train_climatology(train_rows, coverage)
    clim_lo = clim_mu - clim_q * clim_sig
    clim_hi = clim_mu + clim_q * clim_sig
    clim_iscore = float(
        np.mean(
            interval_score(
                np.array([r["x_observed_c"] for r in obs]),
                np.full(obs_n, clim_lo),
                np.full(obs_n, clim_hi),
                alpha,
            )
        )
    )
    clim_brier = float(np.mean((clim_p - o) ** 2))

    per_h = []
    for h in range(horizon):
        rh = [r for r in recs if r["step"] == h + 1]
        oh = [r for r in rh if r["observed"]]
        per_h.append(
            {
                "horizon": h + 1,
                "oracle_n": len(rh),
                "observed_n": len(oh),
                "cal_score_count": int(meta["cal_score_counts"][h]),
            }
        )

    return {
        "model_type": MODEL_TYPE,
        "schema_version": SCHEMA_VERSION,
        "n_params": int(meta["n_params"]),
        "n_steps": int(meta["steps"]),
        "n_packets": int(meta["n"]),
        "series_rows": len(rows),
        "horizon": horizon,
        "win": win,
        "nominal_coverage": coverage,
        "split": {
            "train_end": int(split["train_end"]),
            "cal_end": int(split["cal_end"]),
            "test_end": int(split["test_end"]),
            "n": int(split["n"]),
        },
        "train_loss_first": float(meta["train_loss_first"]),
        "train_loss_last": float(meta["train_loss_last"]),
        "cal_score_counts": [int(v) for v in meta["cal_score_counts"]],
        "n_test_origins": len(origins),
        "oracle_n": len(recs),
        "oracle_rmse": oracle_rmse,
        "oracle_mae": oracle_mae,
        "oracle_interval_coverage": oracle_cov,
        "oracle_persistence_rmse": persist_rmse,
        "observed_n": obs_n,
        "observed_rmse": obs_rmse,
        "observed_gaussian_nll": obs_nll,
        "observed_interval_coverage": obs_cov,
        "observed_interval_mean_width": mean_width,
        "observed_interval_score": mean_iscore,
        "observed_coverage_cluster_lo": cluster_lo,
        "observed_coverage_cluster_hi": cluster_hi,
        "observed_coverage_cluster_n_origins": int(cluster_g),
        "observed_coverage_cluster_method": "origin_cluster_robust_normal",
        "trust_n": len(recs),
        "trust_bce": trust_bce,
        "trust_brier": trust_brier,
        "climatology_trust_brier": clim_brier,
        "climatology_interval_score": clim_iscore,
        "per_horizon": per_h,
        "checkpoint": "artifacts/checkpoint.npz",
        "forecast": "artifacts/forecast.json",
    }


def process_working_sets() -> tuple:
    """Return (peak_working_set_bytes, current_working_set_bytes) via K32GetProcessMemoryInfo."""
    import ctypes
    from ctypes import wintypes, POINTER, byref, sizeof

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    fn = k32.K32GetProcessMemoryInfo
    fn.argtypes = [ctypes.c_void_p, POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
    fn.restype = wintypes.BOOL
    pmc = PROCESS_MEMORY_COUNTERS()
    pmc.cb = sizeof(pmc)
    if not fn(k32.GetCurrentProcess(), byref(pmc), pmc.cb):
        raise OSError(f"K32GetProcessMemoryInfo failed: {ctypes.get_last_error()}")
    return int(pmc.PeakWorkingSetSize), int(pmc.WorkingSetSize)


def peak_working_set_bytes() -> int:
    return process_working_sets()[0]


def nvidia_smi_snapshot() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,name", "--format=csv,noheader"],
            text=True,
        ).strip()
    except Exception as e:
        return f"unavailable: {e}"


def write_budget(extra: dict) -> dict:
    peak_ws, cur_ws = process_working_sets()
    ws = peak_ws
    sizes = {}
    for label, p in [
        ("csv", DATA_CSV),
        ("vendor_csv", VENDOR_CSV),
        ("capture", CAPTURE),
        ("series", SERIES),
        ("checkpoint", CKPT),
        ("forecast", FORECAST),
        ("sim_exe", SIM_EXE),
        ("reconstruct_exe", RECON_EXE),
    ]:
        sizes[label] = int(p.stat().st_size) if p.exists() else 0
    rec = {
        "python_peak_working_set_bytes": ws,
        "python_peak_working_set_mb": round(ws / (1024 * 1024), 3),
        "python_working_set_bytes": cur_ws,
        "python_working_set_mb": round(cur_ws / (1024 * 1024), 3),
        "nvidia_smi": nvidia_smi_snapshot(),
        "file_bytes": sizes,
        "ram_ceiling_gb": 16,
        "vram_ceiling_mb": 6141,
        "train_device": "cpu",
        **extra,
    }
    BUDGET.parent.mkdir(parents=True, exist_ok=True)
    BUDGET.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    ram_txt = ROOT / "artifacts" / "ram.txt"
    ram_txt.write_text(
        "python_peak_working_set_bytes={python_peak_working_set_bytes}\n"
        "python_peak_working_set_mb={python_peak_working_set_mb}\n"
        "python_working_set_bytes={python_working_set_bytes}\n"
        "python_working_set_mb={python_working_set_mb}\n"
        "nvidia_smi={nvidia_smi}\n"
        "capture_bytes={capture}\n"
        "checkpoint_bytes={ckpt}\n"
        "csv_bytes={csv}\n"
        "train_device=cpu\n".format(
            python_peak_working_set_bytes=rec["python_peak_working_set_bytes"],
            python_peak_working_set_mb=rec["python_peak_working_set_mb"],
            python_working_set_bytes=rec["python_working_set_bytes"],
            python_working_set_mb=rec["python_working_set_mb"],
            nvidia_smi=rec["nvidia_smi"],
            capture=sizes["capture"],
            ckpt=sizes["checkpoint"],
            csv=sizes["csv"],
        ),
        encoding="utf-8",
    )
    print(
        f"budget peak_ws_mb={rec['python_peak_working_set_mb']} "
        f"ws_mb={rec['python_working_set_mb']} "
        f"capture_bytes={sizes['capture']} ckpt_bytes={sizes['checkpoint']} "
        f"csv_bytes={sizes['csv']} nvidia_smi={rec['nvidia_smi']}",
        flush=True,
    )
    return rec


def pipeline(
    n: int,
    steps: int,
    seed: int,
    error_scale: float,
    horizon: int,
    coverage: float,
) -> None:
    print("siftcast: obtain csv", flush=True)
    csv_path = obtain_csv()
    print(f"siftcast: csv {csv_path} bytes={csv_path.stat().st_size}", flush=True)
    print("siftcast: build native", flush=True)
    build_native()
    print("siftcast: sim (C++)", flush=True)
    run_sim(n, seed, error_scale, csv_path, CAPTURE, TRUTH)
    print("siftcast: reconstruct (Rust)", flush=True)
    run_reconstruct(CAPTURE, SERIES)
    rows = load_series(SERIES)
    print(f"siftcast: series rows={len(rows)}", flush=True)
    if not TRUTH.exists():
        raise SystemExit("truth.jsonl missing after sim")
    print("siftcast: train (Python GRU)", flush=True)
    params, meta, losses, _X, _splits = fit(
        rows, steps, seed, CKPT, horizon=horizon, coverage=coverage
    )
    print("siftcast: infer", flush=True)
    params2, meta2 = load_checkpoint(CKPT)
    truth_by_seq = load_truth_map(TRUTH)
    split = meta2["split"]
    test_origins = iter_origins(
        len(rows),
        int(meta2["win"]),
        split["cal_end"],
        split["test_end"],
        int(meta2["horizon"]),
        stride=int(meta2["horizon"]),
    )
    if not test_origins:
        raise RuntimeError("no test origins for forecast")
    result = infer_forecast(
        params2, meta2, rows, truth_by_seq=truth_by_seq, origin_t=int(test_origins[0])
    )
    metrics = evaluate_rolling(params2, meta2, rows, truth_by_seq)
    FORECAST.write_text(json.dumps(result, indent=2), encoding="utf-8")
    METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("siftcast: forecast", json.dumps({k: result[k] for k in result if k != "forecast"}), flush=True)
    clo = metrics["observed_coverage_cluster_lo"]
    chi = metrics["observed_coverage_cluster_hi"]
    if clo is None or chi is None:
        ci_s = "null"
    else:
        ci_s = f"[{clo:.3f},{chi:.3f}]"
    print(
        f"siftcast: wrote {FORECAST} points={len(result['forecast'])} "
        f"oracle_rmse={metrics['oracle_rmse']:.4f} persist={metrics['oracle_persistence_rmse']:.4f} "
        f"obs_cov={metrics['observed_interval_coverage']:.3f} "
        f"cluster_ci={ci_s}",
        flush=True,
    )
    write_budget({"metrics": metrics, "losses_head": losses[:3], "losses_tail": losses[-3:]})
    if not CKPT.exists() or CKPT.stat().st_size == 0:
        raise SystemExit("checkpoint missing")
    if not FORECAST.exists() or FORECAST.stat().st_size == 0:
        raise SystemExit("forecast missing")
    if len(result["forecast"]) != horizon:
        raise SystemExit("forecast not task-shaped")
    for rec in result["forecast"]:
        if not (math.isfinite(rec["x_mean_c"]) and rec["x_sigma_c"] > 0 and rec["x_lo_c"] <= rec["x_hi_c"]):
            raise SystemExit("forecast record invalid")
        if not (0.0 <= rec["trust"] <= 1.0):
            raise SystemExit("forecast trust out of range")
    if losses[-1] >= losses[0]:
        print("siftcast: warning loss did not drop (still wrote checkpoint)", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Siftcast build / train / infer")
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--error-scale", type=float, default=1.0)
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument("--coverage", type=float, default=COVERAGE)
    args = ap.parse_args(argv)
    if not (0.0 < args.coverage < 1.0):
        raise SystemExit("--coverage must be in (0, 1)")
    if args.horizon < 1:
        raise SystemExit("--horizon must be >= 1")
    pipeline(args.n, args.steps, args.seed, args.error_scale, args.horizon, args.coverage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
