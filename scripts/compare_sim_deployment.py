#!/usr/bin/env python3
"""Compare full-mission simulation and deployment MPCC traces.

Focuses on sim-to-deployment gap for:
- reference velocity tracking
- control signals (speed and steering)
- cross-track performance
"""

import argparse
import csv
import json
import math
from typing import Dict, List, Optional


def _to_float(value: str) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    x = sorted(values)
    pos = (len(x) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return x[lo]
    w = pos - lo
    return x[lo] * (1.0 - w) + x[hi] * w


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(max(var, 0.0))


def _trim_to_progress_peak(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    prog = [_to_float(r.get("progress_pct", "")) for r in rows]
    prog = [p if p is not None else 0.0 for p in prog]
    if not prog:
        return rows
    max_prog = max(prog)
    end_idx = next((i for i, p in enumerate(prog) if p >= max_prog - 1e-9), len(rows) - 1)
    return rows[: end_idx + 1]


def load_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    return _trim_to_progress_peak(rows)


def summarize(rows: List[Dict[str, str]]) -> Dict[str, float]:
    elapsed = [_to_float(r.get("elapsed_s", "")) for r in rows]
    elapsed = [t for t in elapsed if t is not None]

    cte = [abs(_to_float(r.get("cross_track_err", "")) or 0.0) for r in rows]
    v_meas = [_to_float(r.get("v_meas", "")) or 0.0 for r in rows]
    v_cmd = [_to_float(r.get("v_cmd", "")) or 0.0 for r in rows]
    delta_cmd = [_to_float(r.get("delta_cmd", "")) or 0.0 for r in rows]
    v_ref = [_to_float(r.get("eff_v_ref_k0", "")) for r in rows]
    v_ref = [v for v in v_ref if v is not None]

    duration = (elapsed[-1] - elapsed[0]) if len(elapsed) > 1 else 0.0
    max_prog = max((_to_float(r.get("progress_pct", "")) or 0.0) for r in rows) if rows else 0.0

    delta_rate = []
    accel = []
    for i in range(1, len(rows)):
        t0 = _to_float(rows[i - 1].get("elapsed_s", ""))
        t1 = _to_float(rows[i].get("elapsed_s", ""))
        d0 = _to_float(rows[i - 1].get("delta_cmd", ""))
        d1 = _to_float(rows[i].get("delta_cmd", ""))
        v0 = _to_float(rows[i - 1].get("v_meas", ""))
        v1 = _to_float(rows[i].get("v_meas", ""))
        if None in (t0, t1, d0, d1, v0, v1):
            continue
        dt = t1 - t0
        if dt <= 1e-6:
            continue
        delta_rate.append(abs((d1 - d0) / dt))
        accel.append((v1 - v0) / dt)

    v_ref_track_rmse = 0.0
    if v_ref:
        errs = []
        for r in rows:
            v = _to_float(r.get("v_meas", ""))
            vr = _to_float(r.get("eff_v_ref_k0", ""))
            if v is None or vr is None:
                continue
            errs.append((v - vr) ** 2)
        if errs:
            v_ref_track_rmse = math.sqrt(_mean(errs))

    return {
        "points": float(len(rows)),
        "duration_s": duration,
        "max_progress_pct": max_prog,
        "max_cte_m": max(cte) if cte else 0.0,
        "avg_cte_m": _mean(cte),
        "avg_v_meas_mps": _mean(v_meas),
        "avg_v_cmd_mps": _mean(v_cmd),
        "avg_v_ref_mps": _mean(v_ref) if v_ref else 0.0,
        "v_ref_track_rmse_mps": v_ref_track_rmse,
        "p95_abs_delta_deg": math.degrees(_percentile([abs(d) for d in delta_cmd], 0.95)),
        "p95_abs_delta_rate_degps": math.degrees(_percentile(delta_rate, 0.95)),
        "p95_abs_accel_mps2": _percentile([abs(a) for a in accel], 0.95),
        "std_v_meas_mps": _std(v_meas),
    }


def print_comparison(sim: Dict[str, float], dep: Dict[str, float]) -> None:
    keys = [
        "duration_s",
        "max_progress_pct",
        "max_cte_m",
        "avg_cte_m",
        "avg_v_meas_mps",
        "avg_v_cmd_mps",
        "avg_v_ref_mps",
        "v_ref_track_rmse_mps",
        "p95_abs_delta_deg",
        "p95_abs_delta_rate_degps",
        "p95_abs_accel_mps2",
        "std_v_meas_mps",
        "points",
    ]
    print("\n=== Sim vs Deployment Gap ===")
    print(f"{'metric':30s} {'sim':>12s} {'deploy':>12s} {'gap(dep-sim)':>14s}")
    for k in keys:
        sv = sim.get(k, 0.0)
        dv = dep.get(k, 0.0)
        print(f"{k:30s} {sv:12.4f} {dv:12.4f} {dv - sv:14.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare sim vs deployment MPCC traces")
    parser.add_argument("--sim", required=True, help="Path to full_mission_sim.csv")
    parser.add_argument("--deploy", required=True, help="Path to deployment mpcc_*.csv")
    parser.add_argument("--json-out", default="", help="Optional output JSON path")
    args = parser.parse_args()

    sim_rows = load_csv(args.sim)
    dep_rows = load_csv(args.deploy)

    sim = summarize(sim_rows)
    dep = summarize(dep_rows)
    print_comparison(sim, dep)

    if args.json_out:
        with open(args.json_out, "w", newline="") as f:
            json.dump({"sim": sim, "deployment": dep}, f, indent=2)
        print(f"\nSaved metrics JSON: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
