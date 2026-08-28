"""Siftcast: trust-gated joint forecast of reconstructed serial telemetry.

C++ simulates a framed Gilbert-Elliott capture; Rust reconstructs a
CRC-validated reordered timeline; this file trains and infers a tiny GRU
with a value head gated by packet trust and a trust head.
"""
from __future__ import annotations

import argparse
import json
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
DOUT = 2
WIN = 12
HORIZON = 16

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


def _sig(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def init_params(rng: np.random.Generator, din: int = DIN, dh: int = DH, dout: int = DOUT) -> dict:
    def w(a, b):
        return rng.normal(0.0, 1.0 / np.sqrt(a + b), (a, b)).astype(np.float64)

    return {
        "Wz": w(din, dh),
        "Uz": w(dh, dh),
        "bz": np.zeros(dh),
        "Wr": w(din, dh),
        "Ur": w(dh, dh),
        "br": np.zeros(dh),
        "Wn": w(din, dh),
        "Un": w(dh, dh),
        "bn": np.zeros(dh),
        "Wout": w(dh, dout),
        "bout": np.zeros(dout),
    }


def copy_params(p: dict) -> dict:
    return {k: np.array(v, copy=True) for k, v in p.items()}


def zeros_like_params(p: dict) -> dict:
    return {k: np.zeros_like(v) for k, v in p.items()}


def gru_forward(p: dict, X: np.ndarray):
    """X: (T, din) -> y (2,), cache."""
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
    # target: [x_next_norm, trust_next]
    w = float(target[1])
    mse = 0.5 * w * (y[0] - target[0]) ** 2
    s = float(_sig(y[1:2])[0])
    s = min(max(s, 1e-8), 1.0 - 1e-8)
    tt = min(max(float(target[1]), 0.0), 1.0)
    bce = -(tt * np.log(s) + (1.0 - tt) * np.log(1.0 - s))
    return float(mse + bce)


def forward_loss(p: dict, X: np.ndarray, target: np.ndarray) -> float:
    y, _ = gru_forward(p, X)
    return _loss_from_y(y, target)


def loss_and_grad(p: dict, X: np.ndarray, target: np.ndarray):
    y, cache = gru_forward(p, X)
    loss = _loss_from_y(y, target)
    w = float(target[1])
    dpred = w * (y[0] - target[0])
    s = float(_sig(y[1:2])[0])
    dlogit = s - float(target[1])
    dy = np.array([dpred, dlogit], dtype=np.float64)

    g = zeros_like_params(p)
    h_last = cache[-1][5]
    g["Wout"] += np.outer(h_last, dy)
    g["bout"] += dy
    dh = p["Wout"] @ dy

    for x, h_prev, z, r, n, h_new in reversed(cache):
        # h_new = (1-z)*n + z*h_prev
        dz = dh * (h_prev - n)
        dn = dh * (1.0 - z)
        dh_prev = dh * z

        dn_pre = dn * (1.0 - n * n)
        g["Wn"] += np.outer(x, dn_pre)
        g["Un"] += np.outer(r * h_prev, dn_pre)
        g["bn"] += dn_pre
        dx = p["Wn"] @ dn_pre
        d_rh = p["Un"] @ dn_pre
        dr = d_rh * h_prev
        dh_prev = dh_prev + d_rh * r

        dz_pre = dz * z * (1.0 - z)
        g["Wz"] += np.outer(x, dz_pre)
        g["Uz"] += np.outer(h_prev, dz_pre)
        g["bz"] += dz_pre
        dx = dx + p["Wz"] @ dz_pre
        dh_prev = dh_prev + p["Uz"] @ dz_pre

        dr_pre = dr * r * (1.0 - r)
        g["Wr"] += np.outer(x, dr_pre)
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
    """One SGD step. batch_x (B,T,din), batch_y (B,2). Returns (params, mean_loss)."""
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


def load_checkpoint(path: Path):
    data = np.load(path, allow_pickle=True)
    params = {k: data[k] for k in data.files if k != "_meta"}
    meta = json.loads(str(data["_meta"]))
    return params, meta


# --- series / train / infer ------------------------------------------------


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


def make_windows(X: np.ndarray, win: int = WIN):
    xs, ys = [], []
    for t in range(win, len(X)):
        xs.append(X[t - win : t])
        ys.append(np.array([X[t, 0], X[t, 2]], dtype=np.float64))
    return np.stack(xs, axis=0), np.stack(ys, axis=0)


def fit(
    rows: list,
    steps: int,
    seed: int,
    ckpt: Path,
    lr: float = 0.05,
    batch: int = 32,
    win: int = WIN,
):
    n = len(rows)
    split = max(win + 2, int(0.8 * n))
    train_rows = rows[:split]
    mean, std = trusted_moments(train_rows)
    X = series_matrix(rows, mean, std)
    Wx, Wy = make_windows(X[:split], win)
    rng = np.random.Generator(np.random.PCG64(seed))
    params = init_params(rng)
    losses = []
    idx = np.arange(len(Wx))
    for step in range(steps):
        rng.shuffle(idx)
        bix = idx[: min(batch, len(idx))]
        params, loss = train_step(params, Wx[bix], Wy[bix], lr)
        losses.append(float(loss))
        if step == 0 or (step + 1) % max(1, steps // 10) == 0 or step + 1 == steps:
            print(f"train step {step+1}/{steps} loss={loss:.5f}", flush=True)
    meta = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "win": win,
        "din": DIN,
        "dh": int(params["bz"].shape[0]),
        "steps": steps,
        "seed": seed,
        "split": split,
        "n": n,
        "train_loss_first": losses[0],
        "train_loss_last": losses[-1],
    }
    save_checkpoint(ckpt, params, meta)
    print(f"checkpoint wrote {ckpt} bytes={ckpt.stat().st_size} first={losses[0]:.5f} last={losses[-1]:.5f}", flush=True)
    return params, meta, losses, X, split


def persist_mse(x_true: np.ndarray, last: float) -> float:
    pred = np.full_like(x_true, last)
    return float(np.mean((pred - x_true) ** 2))


def infer_forecast(params: dict, meta: dict, rows: list, horizon: int = HORIZON) -> dict:
    mean = np.array(meta["mean"], dtype=np.float64)
    std = np.array(meta["std"], dtype=np.float64)
    win = int(meta["win"])
    split = int(meta["split"])
    X = series_matrix(rows, mean, std)
    start = split
    if start + horizon > len(X):
        start = max(win, len(X) - horizon)
    ctx = X[start - win : start].copy()
    preds = []
    for h in range(horizon):
        y, _ = gru_forward(params, ctx)
        x_hat = float(y[0])
        trust = float(_sig(np.array([y[1]]))[0])
        # roll: y-channel kept as last observed y (we don't emit y)
        nxt = np.array([x_hat, ctx[-1, 1], trust], dtype=np.float64)
        ctx = np.vstack([ctx[1:], nxt])
        x_c = x_hat * std[0] + mean[0]
        rec = {"step": h + 1, "x_c": x_c, "trust": trust}
        if start + h < len(rows):
            rec["x_true"] = float(rows[start + h]["x"])
            rec["trust_true"] = float(rows[start + h]["trust"])
        preds.append(rec)
    out = {
        "units": "deg_C",
        "horizon": horizon,
        "origin_seq": int(rows[start - 1]["seq"]) if start else 0,
        "forecast": preds,
    }
    if all("x_true" in p for p in preds):
        xt = np.array([p["x_true"] for p in preds])
        xp = np.array([p["x_c"] for p in preds])
        last_obs = float(rows[start - 1]["x"])
        out["holdout_mse"] = float(np.mean((xp - xt) ** 2))
        out["persistence_mse"] = persist_mse(xt, last_obs)
        tt = np.array([p["trust_true"] for p in preds])
        tp = np.array([p["trust"] for p in preds])
        tp = np.clip(tp, 1e-8, 1 - 1e-8)
        out["trust_bce"] = float(-np.mean(tt * np.log(tp) + (1 - tt) * np.log(1 - tp)))
        out["mean_pred_trust"] = float(np.mean(tp))
        out["mean_true_trust"] = float(np.mean(tt))
    return out


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


def pipeline(n: int, steps: int, seed: int, error_scale: float, horizon: int) -> None:
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
    print("siftcast: train (Python GRU)", flush=True)
    params, meta, losses, _X, _split = fit(rows, steps, seed, CKPT)
    print("siftcast: infer", flush=True)
    params2, meta2 = load_checkpoint(CKPT)
    result = infer_forecast(params2, meta2, rows, horizon=horizon)
    FORECAST.write_text(json.dumps(result, indent=2), encoding="utf-8")
    metrics = {
        "train_loss_first": meta["train_loss_first"],
        "train_loss_last": meta["train_loss_last"],
        "n_steps": steps,
        "n_packets": n,
        "series_rows": len(rows),
        "holdout_mse": result.get("holdout_mse"),
        "persistence_mse": result.get("persistence_mse"),
        "trust_bce": result.get("trust_bce"),
        "checkpoint": str(CKPT),
        "forecast": str(FORECAST),
    }
    METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("siftcast: forecast", json.dumps({k: result[k] for k in result if k != "forecast"}), flush=True)
    print(f"siftcast: wrote {FORECAST} points={len(result['forecast'])}", flush=True)
    write_budget({"metrics": metrics, "losses_head": losses[:3], "losses_tail": losses[-3:]})
    if not CKPT.exists() or CKPT.stat().st_size == 0:
        raise SystemExit("checkpoint missing")
    if not FORECAST.exists() or FORECAST.stat().st_size == 0:
        raise SystemExit("forecast missing")
    if len(result["forecast"]) != horizon:
        raise SystemExit("forecast not task-shaped")
    if losses[-1] >= losses[0]:
        print("siftcast: warning loss did not drop (still wrote checkpoint)", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Siftcast build / train / infer")
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--error-scale", type=float, default=1.0)
    ap.add_argument("--horizon", type=int, default=HORIZON)
    args = ap.parse_args(argv)
    pipeline(args.n, args.steps, args.seed, args.error_scale, args.horizon)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
