"""Drive shipped train/infer and each language job. Run: python tests/test_siftcast.py"""
from __future__ import annotations

import json
import math
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


def test_softplus_finite():
    z = np.array([-1e6, -50.0, 0.0, 50.0, 1e6])
    y = sc._softplus(z)
    assert np.all(np.isfinite(y)), y
    assert y[0] < 1e-8
    assert abs(float(y[2]) - math.log(2.0)) < 1e-10
    assert abs(float(y[3]) - 50.0) < 1e-8
    assert abs(float(y[4]) - 1e6) < 1e-6
    # derivative of softplus is sigmoid; check inv around unit scale
    inv = float(sc._softplus_inv(np.array(1.0 - sc.SIGMA_FLOOR)))
    assert abs(float(sc._softplus(np.array(inv))) - (1.0 - sc.SIGMA_FLOOR)) < 1e-12


def test_conformal_quantile_order_statistic():
    scores = np.array([0.1, 0.4, 0.2, 0.9, 0.3])  # n=5, sorted 0.1,0.2,0.3,0.4,0.9
    # c=0.9: k = min(5, ceil(6*0.9)=ceil(5.4)=6) = 5 -> sorted[4] = 0.9
    assert sc.conformal_quantile(scores, 0.9) == 0.9
    # c=0.5: k = min(5, ceil(6*0.5)=3) = 3 -> sorted[2] = 0.3
    assert sc.conformal_quantile(scores, 0.5) == 0.3
    raised = False
    try:
        sc.conformal_quantile(np.array([]), 0.9)
    except ValueError:
        raised = True
    assert raised


def test_split_targets_do_not_cross_blocks():
    n, win, H = 200, 12, 16
    b = sc.split_boundaries(n, win, H)
    assert b["train_end"] == int(0.70 * n)
    assert b["cal_end"] == int(0.85 * n)
    assert b["test_end"] == n
    X = np.zeros((n, 3))
    blocks = [
        ("train", 0, b["train_end"], 1),
        ("cal", b["train_end"], b["cal_end"], H),
        ("test", b["cal_end"], b["test_end"], H),
    ]
    for name, lo, hi, stride in blocks:
        _Wx, Wy, origins = sc.make_windows(X, win, H, lo, hi, stride=stride)
        assert len(origins) >= 1, name
        assert Wy.shape[1:] == (H, 2), (name, Wy.shape)
        for t in origins:
            assert t >= lo, (name, t, lo)
            last = int(t) + H - 1
            assert last < hi, (name, last, hi)
            assert int(t) - win >= 0
            # context may look backward across lo; targets may not look forward past hi
            assert last >= lo


def test_split_rejects_short_series():
    raised = False
    try:
        sc.split_boundaries(30, 12, 16)
    except ValueError as e:
        raised = True
        assert "short" in str(e).lower()
    assert raised
    raised = False
    try:
        sc.split_boundaries(100, 12, 16)  # 15% of 100 = 15 < horizon 16
    except ValueError as e:
        raised = True
        assert "short" in str(e).lower()
    assert raised


def _fd(p, X, y, key, idx, eps=1e-5):
    p2 = sc.copy_params(p)
    arr = p2[key]
    arr[idx] = arr[idx] + eps
    lp = sc.forward_loss(p2, X, y)
    arr[idx] = arr[idx] - 2 * eps
    lm = sc.forward_loss(p2, X, y)
    return (lp - lm) / (2 * eps)


def test_grad_matches_finite_diff():
    rng = np.random.default_rng(0)
    H = 4
    p = sc.init_params(rng, horizon=H)
    X = rng.normal(size=(sc.WIN, sc.DIN))
    X[:, 2] = 1.0
    y = np.zeros((H, 2))
    y[:, 0] = rng.normal(size=H) * 0.2
    y[:, 1] = 1.0
    _, g = sc.loss_and_grad(p, X, y)
    checks = [
        ("Wout", (0, 0), "mean"),
        ("Wout", (0, 1), "scale"),
        ("Wout", (0, 2), "trust"),
        ("Uz", (0, 0), "recurrent"),
    ]
    for key, idx, name in checks:
        num = _fd(p, X, y, key, idx)
        analytic = float(g[key][idx])
        assert abs(num - analytic) < 2e-4, (name, num, analytic)


def _toy_rows(n: int, seed: int = 1):
    t = np.arange(n, dtype=np.float64)
    x = np.sin(0.25 * t)
    trust = np.where((t % 25) < 2, 0.0, 1.0)
    return [{"seq": int(i), "x": float(x[i]), "y": float(x[i]), "trust": float(trust[i])} for i in range(n)]


def _assert_forecast_records(fc: dict, horizon: int):
    assert fc["horizon"] == horizon
    assert fc["model_type"] == sc.MODEL_TYPE
    assert fc["schema_version"] == sc.SCHEMA_VERSION
    assert "nominal_coverage" in fc and "origin_seq" in fc
    recs = fc["forecast"]
    assert len(recs) == horizon, len(recs)
    for p in recs:
        for k in ("step", "x_mean_c", "x_sigma_c", "x_lo_c", "x_hi_c", "trust"):
            assert k in p, k
            assert math.isfinite(float(p[k if k != "step" else "x_mean_c"]))
        assert p["x_sigma_c"] > 0, p["x_sigma_c"]
        assert p["x_lo_c"] <= p["x_hi_c"], (p["x_lo_c"], p["x_hi_c"])
        assert 0.0 <= p["trust"] <= 1.0, p["trust"]


def test_train_step_drops_loss_and_writes_checkpoint(tmp_path: Path | None = None):
    rows = _toy_rows(160)
    dest = (tmp_path or (ROOT / "artifacts")) / "test_ckpt.npz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    params, meta, losses, _X, _split = sc.fit(
        rows, steps=40, seed=2, ckpt=dest, lr=0.08, batch=16, horizon=8, coverage=0.90
    )
    assert dest.exists() and dest.stat().st_size > 0
    assert len(losses) == 40
    assert losses[-1] < losses[0], (losses[0], losses[-1])
    assert meta["schema_version"] == 2
    assert meta["model_type"] == sc.MODEL_TYPE
    assert meta["horizon"] == 8
    assert meta["n_params"] < 2000
    assert meta["n_params"] == sc.param_count(params)
    assert len(meta["q"]) == 8
    loaded, meta2 = sc.load_checkpoint(dest)
    y0, _ = sc.gru_forward(params, _X[: sc.WIN])
    y1, _ = sc.gru_forward(loaded, _X[: sc.WIN])
    assert np.allclose(y0, y1)
    assert meta2["steps"] == 40
    assert loaded["Wout"].shape[1] == sc.head_width(8)
    fc = sc.infer_forecast(loaded, meta2, rows)
    _assert_forecast_records(fc, 8)


def test_rejects_schema_v1_checkpoint():
    dest = ROOT / "artifacts" / "test_ckpt_v1.npz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    dh, din = 16, 3

    def w(a, b):
        return rng.normal(0.0, 0.1, (a, b))

    np.savez(
        dest,
        Wz=w(din, dh),
        Uz=w(dh, dh),
        bz=np.zeros(dh),
        Wr=w(din, dh),
        Ur=w(dh, dh),
        br=np.zeros(dh),
        Wn=w(din, dh),
        Un=w(dh, dh),
        bn=np.zeros(dh),
        Wout=w(dh, 2),
        bout=np.zeros(2),
        _meta=np.array(json.dumps({"win": 12, "mean": [0.0, 0.0], "std": [1.0, 1.0], "split": 80})),
    )
    raised = False
    try:
        sc.load_checkpoint(dest)
    except RuntimeError as e:
        raised = True
        msg = str(e).lower()
        assert "rerun" in msg and "training" in msg, e
    assert raised


def test_calibrate_fails_without_trusted_scores():
    n, win, H = 200, 12, 16
    rows = _toy_rows(n)
    splits = sc.split_boundaries(n, win, H)
    for r in rows[splits["train_end"] : splits["cal_end"]]:
        r["trust"] = 0.0
    mean, std = sc.trusted_moments(rows[: splits["train_end"]])
    X = sc.series_matrix(rows, mean, std)
    rng = np.random.default_rng(0)
    params = sc.init_params(rng, horizon=H)
    raised = False
    try:
        sc.calibrate(params, X, splits, 0.90, win, H)
    except RuntimeError as e:
        raised = True
        assert "trusted calibration" in str(e).lower() or "horizon" in str(e).lower()
    assert raised


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
    test_softplus_finite()
    print("ok softplus")
    test_conformal_quantile_order_statistic()
    print("ok conformal")
    test_split_targets_do_not_cross_blocks()
    print("ok split bounds")
    test_split_rejects_short_series()
    print("ok split short")
    test_grad_matches_finite_diff()
    print("ok grad")
    test_train_step_drops_loss_and_writes_checkpoint()
    print("ok train_step")
    test_rejects_schema_v1_checkpoint()
    print("ok schema v1 reject")
    test_calibrate_fails_without_trusted_scores()
    print("ok cal fail")
    test_cpp_self_test_and_clean_capture()
    print("ok cpp")
    test_rust_cargo_and_reconstruct_clean()
    print("ok rust")
    test_error_path_produces_untrusted_holds()
    print("ok noisy channel")
    print("all passed")
