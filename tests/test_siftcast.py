"""Drive shipped train/infer and each language job. Run: python tests/test_siftcast.py"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import siftcast as sc  # noqa: E402


def test_peak_working_set_nonzero():
    peak, cur = sc.process_working_sets()
    assert peak > 1_000_000, peak
    assert cur > 1_000_000, cur
    assert peak >= cur


def test_grad_matches_finite_diff():
    rng = np.random.default_rng(0)
    p = sc.init_params(rng)
    X = rng.normal(size=(sc.WIN, sc.DIN))
    X[:, 2] = 1.0
    y = np.array([0.15, 1.0])
    _, g = sc.loss_and_grad(p, X, y)
    eps = 1e-5
    p2 = sc.copy_params(p)
    p2["Wout"][0, 0] += eps
    lp = sc.forward_loss(p2, X, y)
    p2["Wout"][0, 0] -= 2 * eps
    lm = sc.forward_loss(p2, X, y)
    num = (lp - lm) / (2 * eps)
    analytic = float(g["Wout"][0, 0])
    assert abs(num - analytic) < 2e-4, (num, analytic)


def test_train_step_drops_loss_and_writes_checkpoint(tmp_path: Path | None = None):
    rng = np.random.default_rng(1)
    t = np.arange(80, dtype=np.float64)
    x = np.sin(0.25 * t)
    trust = np.where((t % 20) < 3, 0.0, 1.0)
    rows = [{"seq": int(i), "x": float(x[i]), "y": float(x[i]), "trust": float(trust[i])} for i in range(80)]
    dest = (tmp_path or (ROOT / "artifacts")) / "test_ckpt.npz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    params, meta, losses, _X, _split = sc.fit(rows, steps=40, seed=2, ckpt=dest, lr=0.08, batch=16)
    assert dest.exists() and dest.stat().st_size > 0
    assert len(losses) == 40
    assert losses[-1] < losses[0], (losses[0], losses[-1])
    loaded, meta2 = sc.load_checkpoint(dest)
    y0, _ = sc.gru_forward(params, _X[: sc.WIN])
    y1, _ = sc.gru_forward(loaded, _X[: sc.WIN])
    assert np.allclose(y0, y1)
    assert meta2["steps"] == 40
    fc = sc.infer_forecast(loaded, meta2, rows, horizon=8)
    assert len(fc["forecast"]) == 8
    assert "x_c" in fc["forecast"][0] and "trust" in fc["forecast"][0]
    assert all(isinstance(p["x_c"], float) for p in fc["forecast"])


def test_cpp_self_test_and_clean_capture():
    sc.obtain_csv()
    sc.build_sim()
    r = subprocess.run([str(sc.SIM_EXE), "--self-test"], capture_output=True, text=True)
    print((r.stderr + r.stdout).strip())
    assert r.returncode == 0, r.stderr
    assert "self-test ok" in (r.stderr + r.stdout)
    out = ROOT / "artifacts" / "test_clean.bin"
    truth = ROOT / "artifacts" / "test_clean_truth.jsonl"
    subprocess.check_call(
        [
            str(sc.SIM_EXE),
            "--in",
            str(sc.DATA_CSV if sc.DATA_CSV.exists() else sc.VENDOR_CSV),
            "--out",
            str(out),
            "--truth",
            str(truth),
            "--n",
            "64",
            "--seed",
            "3",
            "--error-scale",
            "0",
        ]
    )
    data = out.read_bytes()
    assert len(data) == 64 * 13
    assert data[0] == 0xA5 and data[1] == 0x5A
    assert b"\xA5\x5A" in data


def test_rust_cargo_and_reconstruct_clean():
    r = subprocess.run(
        ["cargo", "test", "--manifest-path", str(ROOT / "reconstruct" / "Cargo.toml")],
        capture_output=True,
        text=True,
    )
    blob = r.stdout + r.stderr
    print(blob)
    assert r.returncode == 0, blob
    assert "test result: ok" in blob
    sc.build_reconstruct()
    capture = ROOT / "artifacts" / "test_clean.bin"
    series = ROOT / "artifacts" / "test_clean.jsonl"
    if not capture.exists():
        test_cpp_self_test_and_clean_capture()
    subprocess.check_call([str(sc.RECON_EXE), "--in", str(capture), "--out", str(series)])
    rows = sc.load_series(series)
    assert len(rows) == 64
    assert all(row["trust"] == 1.0 for row in rows)
    assert all(row["held"] is False for row in rows)
    xs = [row["x"] for row in rows]
    assert max(xs) - min(xs) > 0.5


def test_error_path_produces_untrusted_holds():
    sc.obtain_csv()
    sc.build_native()
    cap = ROOT / "artifacts" / "test_noisy.bin"
    truth = ROOT / "artifacts" / "test_noisy_truth.jsonl"
    series = ROOT / "artifacts" / "test_noisy.jsonl"
    subprocess.check_call(
        [
            str(sc.SIM_EXE),
            "--in",
            str(sc.VENDOR_CSV),
            "--out",
            str(cap),
            "--truth",
            str(truth),
            "--n",
            "256",
            "--seed",
            "9",
            "--error-scale",
            "1",
        ]
    )
    subprocess.check_call([str(sc.RECON_EXE), "--in", str(cap), "--out", str(series)])
    rows = sc.load_series(series)
    held = sum(1 for r in rows if r["held"])
    trusted = sum(1 for r in rows if r["trust"] >= 0.5)
    assert len(rows) >= 8
    assert held >= 1, "channel should create at least one hold"
    assert trusted >= 1
    assert 0 < trusted / len(rows) < 1


if __name__ == "__main__":
    test_peak_working_set_nonzero()
    print("ok ram")
    test_grad_matches_finite_diff()
    print("ok grad")
    test_train_step_drops_loss_and_writes_checkpoint()
    print("ok train_step")
    test_cpp_self_test_and_clean_capture()
    print("ok cpp")
    test_rust_cargo_and_reconstruct_clean()
    print("ok rust")
    test_error_path_produces_untrusted_holds()
    print("ok noisy channel")
    print("all passed")
