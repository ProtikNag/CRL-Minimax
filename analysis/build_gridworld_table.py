"""Build the 6-method x 3-seed GridWorld shared-head comparison table.

Per run: contract_metrics (PERF/forgetting/BWT/FWT); FWT is paired to the
seed-matched sh_baseline (from-scratch reference). Aggregates mean +/- 95% CI
(Student-t) per method, writes metrics.json + SUMMARY.md into reports/<out>.

  python -m analysis.build_gridworld_table
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from analysis.contract_metrics import compute_metrics

# method -> run-dir stem (seed appended). FWT baseline = sh_baseline same seed.
METHODS = {
    "ours":     "biggrid50_sh_ours_seed{s}",
    "cka_rl":   "biggrid50_cka_rl_seed{s}",
    "crelu":    "biggrid50_crelu_seed{s}",
    "cbp":      "biggrid50_cbp_seed{s}",
    "finetune": "biggrid50_sh_finetune_seed{s}",
    "baseline": "biggrid50_sh_baseline_seed{s}",
}
LABEL = {"ours": "Ours (min-max)", "cka_rl": "CKA-RL", "crelu": "CReLUs",
         "cbp": "CbpNet", "finetune": "Finetune", "baseline": "Baseline (scratch)"}
SEEDS = [0, 1, 2]
_T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}
RES = Path("results")
OUT = Path("reports/gridworld_sharedhead")


def _ci(vals):
    v = np.asarray(vals, float)
    n = len(v)
    if n < 2:
        return float(v.mean()), 0.0
    return float(v.mean()), float(_T95.get(n, 1.96) * v.std(ddof=1) / np.sqrt(n))


def main():
    per_run, agg = {}, {}
    for method, stem in METHODS.items():
        rows = []
        for s in SEEDS:
            d = RES / stem.format(s=s)
            if not (d / "eval_matrix.json").exists():
                continue
            base = RES / f"biggrid50_sh_baseline_seed{s}"  # FWT reference
            m = compute_metrics(d, base if base.exists() and method != "baseline" else None)
            done = len(json.loads((d / "eval_matrix.json").read_text()))
            rec = {"tasks_done": done, "complete": done == 50,
                   "perf": m["perf"], "forgetting": m["forgetting"],
                   "bwt": m["bwt"], "fwt": m["fwt"]}
            per_run[d.name] = rec
            if rec["complete"]:
                rows.append(rec)
        if rows:
            pm, ph = _ci([r["perf"] for r in rows])
            fm, fh = _ci([r["forgetting"] for r in rows])
            bm, bh = _ci([r["bwt"] for r in rows])
            fwt_vals = [r["fwt"] for r in rows if r["fwt"] is not None]
            wm, wh = _ci(fwt_vals) if fwt_vals else (None, None)
            agg[method] = {"label": LABEL[method], "n_seeds": len(rows),
                           "perf_mean": pm, "perf_ci": ph,
                           "forgetting_mean": fm, "forgetting_ci": fh,
                           "bwt_mean": bm, "bwt_ci": bh,
                           "fwt_mean": wm, "fwt_ci": wh}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps({**per_run, "_aggregate": agg}, indent=2))

    # SUMMARY.md table, sorted by PERF desc.
    order = sorted(agg, key=lambda m: -agg[m]["perf_mean"])
    lines = ["# GridWorld (50-task, shared-head) — 6-method × 3-seed comparison", "",
             "Fixed-capacity / no-growth regime. PERF/FWT/BWT normalized "
             "(0=random, 1=expert-ceiling); mean ± 95% CI over 3 seeds. "
             "FWT paired to seed-matched from-scratch baseline.", "",
             "| Method | PERF ↑ | FWT ↑ | BWT ↑ | Forgetting ↓ | seeds |",
             "|---|---|---|---|---|---|"]
    for m in order:
        a = agg[m]
        fwt = f"{a['fwt_mean']:+.3f}" if a["fwt_mean"] is not None else "—"
        lines.append(
            f"| {a['label']} | {a['perf_mean']:.3f} ± {a['perf_ci']:.3f} | {fwt} | "
            f"{a['bwt_mean']:+.3f} ± {a['bwt_ci']:.3f} | "
            f"{a['forgetting_mean']:.3f} ± {a['forgetting_ci']:.3f} | {a['n_seeds']} |")
    (OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
