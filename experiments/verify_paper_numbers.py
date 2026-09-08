"""
verify_paper_numbers.py -- Regenerate every empirical number quoted in paper/main.tex
from the raw experiment result JSONs, and check each claim.

Ground-truth sources:
  experiments/results/{dataset}/results_seed*.json        main benchmark (LightGCN encoder)
  experiments/results_tgn/{dataset}/results_seed*.json    TGN encoder ablation
  experiments/results-v2-lambda-{l}/{dataset}/...         lambda_CFR sweep
  experiments/results/{dataset}/ablation_seed*.json       component ablation
  experiments/results/{dataset}_seed*/results_cold_start.json
  experiments/results/{dataset}/cold_start_seed*.json     cold-start partition
  experiments/results/{dataset}/sensitivity_seed*.json    hyperparameter sweep
  data/processed_v2/{campaign}/stats.json + lightgcn_adj.npz
  data/processed_criteo/stats.json

Usage:
    python experiments/verify_paper_numbers.py
"""

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

DATASETS = ["obd-all", "obd-men", "obd-women", "kuairec", "kuairand", "criteo"]
METHODS = [  # (json key, paper label)
    ("GNN-Bandit", "GNN-Bandit (Ours)"),
    ("DecisionTransformer", "Decision Transformer"),
    ("CQL", "CQL"),
    ("Greedy-GNN", "Greedy-GNN (No RL)"),
    ("NeuralUCB", "NeuralUCB"),
    ("IQL", "IQL"),
    ("MF-Bandit", "MF-Bandit"),
    ("LinUCB", "LinUCB"),
    ("Uplift-Only", "Uplift-Only"),
    ("DQN", "DQN"),
    ("Random", "Random Policy"),
    ("BTS", "BTS (Logging Policy)"),
]

failures = []


def check(name, claimed, computed, tol=5e-7):
    ok = claimed is not None and abs(claimed - computed) <= tol
    status = "PASS" if ok else "FAIL"
    if not ok:
        failures.append((name, claimed, computed))
    print(f"  [{status}] {name}: paper={claimed}  computed={round(computed, 8)}")
    return ok


def load_dr(results_dir, dataset):
    """Return {method: [DR values per seed]} for a results directory."""
    ds_dir = ROOT / results_dir / dataset
    out = defaultdict(list)
    seeds = []
    for f in sorted(ds_dir.glob("results_seed*.json")):
        seeds.append(f.stem.replace("results_seed", ""))
        with open(f) as fp:
            data = json.load(fp)
        for model, metrics in data.get("ope_results", {}).items():
            if "DR" in metrics and "value" in metrics["DR"]:
                out[model].append(metrics["DR"]["value"])
    return dict(out), seeds


def fmt_p(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def mean_std(vals):
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.std())  # ddof=0, matches significance_tests.py


def lift_pct(ours, base):
    return (ours - base) / abs(base) * 100.0


# ---------------------------------------------------------------------------
print("=" * 100)
print("SECTION A: MAIN BENCHMARK TABLE (Table 1) -- experiments/results/")
print("=" * 100)

main = {}
seed_report = {}
for ds in DATASETS:
    dr, seeds = load_dr("experiments/results", ds)
    main[ds] = dr
    seed_report[ds] = seeds
    counts = {m: len(v) for m, v in sorted(dr.items())}
    print(f"\n{ds}: result files for seeds {seeds}")
    print(f"  per-method seed counts: {counts}")

print("\n--- Per-dataset summary (mean ± std DR, ddof=0) ---")
summary = {}
for ds in DATASETS:
    summary[ds] = {}
    print(f"\n{ds}:")
    for key, label in METHODS:
        if key not in main[ds]:
            print(f"  {label:<28} MISSING")
            continue
        m, s = mean_std(main[ds][key])
        summary[ds][key] = (m, s)
        print(f"  {label:<28} {m:.6f} ± {s:.6f}  (n={len(main[ds][key])})")

print("\n--- GNN-Bandit vs BEST baseline (paired t-test, two-sided) ---")
best_baseline = {}
for ds in DATASETS:
    gnn = np.asarray(main[ds]["GNN-Bandit"])
    best_key, best_mean = None, -np.inf
    for key, _ in METHODS[1:]:
        if key in main[ds] and len(main[ds][key]) == len(gnn):
            mu = np.mean(main[ds][key])
            if mu > best_mean:
                best_key, best_mean = key, mu
    best = np.asarray(main[ds][best_key])
    _, p = stats.ttest_rel(gnn, best)
    lift = lift_pct(gnn.mean(), best.mean())
    bts = np.asarray(main[ds]["BTS"])
    lift_bts = lift_pct(gnn.mean(), bts.mean())
    best_baseline[ds] = (best_key, best_mean, p, lift)
    print(f"  {ds:<10} best={best_key:<20} GNN={gnn.mean():.6f} lift={lift:+.2f}%  "
          f"p={p:.2e} ({fmt_p(p)})  | lift_vs_BTS={lift_bts:+.2f}%  seeds={len(gnn)}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("SECTION B: TGN COMPARISON (Table 3) -- experiments/results_tgn/")
print("=" * 100)

tgn_datasets = ["obd-all", "obd-men", "obd-women", "criteo"]
for ds in tgn_datasets:
    tgn_dr, tgn_seeds = load_dr("experiments/results_tgn", ds)
    lg = np.asarray(main[ds]["GNN-Bandit"])
    tg = np.asarray(tgn_dr["GNN-Bandit"])
    n = min(len(lg), len(tg))
    _, p = stats.ttest_rel(lg[:n], tg[:n])
    adv = lift_pct(lg[:n].mean(), tg[:n].mean())
    print(f"  {ds:<10} LightGCN={lg.mean():.6f}±{lg.std():.6f} (n={len(lg)})  "
          f"TGN={tg.mean():.6f}±{tg.std():.6f} (n={len(tg)})  "
          f"advantage={adv:+.2f}% (LightGCN)  p={p:.4f} ({fmt_p(p)})")

# Runtime claim: "order of magnitude faster"
print("\n--- Runtime (elapsed_seconds, whole pipeline) LightGCN vs TGN ---")
for ds in tgn_datasets:
    t_lg = []
    for f in sorted((ROOT / "experiments/results" / ds).glob("results_seed*.json")):
        t_lg.append(json.load(open(f)).get("elapsed_seconds", np.nan))
    t_tg = []
    for f in sorted((ROOT / "experiments/results_tgn" / ds).glob("results_seed*.json")):
        t_tg.append(json.load(open(f)).get("elapsed_seconds", np.nan))
    if t_lg and t_tg:
        r = np.nanmean(t_tg) / np.nanmean(t_lg)
        print(f"  {ds:<10} LightGCN={np.nanmean(t_lg):.0f}s  TGN={np.nanmean(t_tg):.0f}s  "
              f"TGN/LightGCN={r:.1f}x")

# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("SECTION C: LAMBDA_CFR PARETO (Table 4) -- experiments/results-v2-lambda-*")
print("=" * 100)

lambda_dirs = {"0.05": "experiments/results-v2-lambda-0.05",
               "0.10": "experiments/results-v2-lambda-0.1",
               "0.20": "experiments/results-v2-lambda-0.2"}
lam_summary = {}
for lam, d in lambda_dirs.items():
    for ds in ["obd-all", "obd-men", "obd-women", "criteo"]:
        dr, seeds = load_dr(d, ds)
        if "GNN-Bandit" not in dr:
            print(f"  lambda={lam} {ds}: MISSING")
            continue
        m, s = mean_std(dr["GNN-Bandit"])
        cv = s / m * 100 if m != 0 else float("nan")
        lam_summary[(ds, lam)] = (m, s, cv, len(dr["GNN-Bandit"]))
        print(f"  lambda={lam} {ds:<10} GNN mean={m:.6f} ± {s:.6f}  CV={cv:.2f}%  n={len(dr['GNN-Bandit'])}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("SECTION D: ABLATION -- experiments/results/{ds}/ablation_seed*.json")
print("=" * 100)

ablation_keys = ["Full GNN-Bandit", "No-Graph (BCQ only)", "No-Constraint (GNN+DQN)"]
for ds in DATASETS:
    files = sorted((ROOT / "experiments/results" / ds).glob("ablation_seed*.json"))
    if not files:
        print(f"  {ds}: no ablation files")
        continue
    vals = defaultdict(list)
    for f in files:
        data = json.load(open(f))
        for k in ablation_keys:
            if k in data and "DR" in data[k]:
                vals[k].append(data[k]["DR"]["value"])
    print(f"\n  {ds} (n={len(files)} seeds):")
    means = {}
    for k in ablation_keys:
        if vals[k]:
            m, s = mean_std(vals[k])
            means[k] = m
            print(f"    {k:<28} {m:.6f} ± {s:.6f}")
    if "Full GNN-Bandit" in means:
        full = means["Full GNN-Bandit"]
        if "No-Graph (BCQ only)" in means:
            print(f"    No-Graph collapse:    {lift_pct(means['No-Graph (BCQ only)'], full):+.2f}%")
        if "No-Constraint (GNN+DQN)" in means:
            print(f"    No-Constraint collapse:{lift_pct(means['No-Constraint (GNN+DQN)'], full):+.2f}%")

# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("SECTION E: COLD-START -- results_cold_start.json / cold_start_seed*.json")
print("=" * 100)

def load_cold(ds):
    vals = defaultdict(list)
    for f in sorted(glob.glob(str(ROOT / "experiments" / "results" / f"{ds}_seed*" / "results_cold_start.json"))):
        data = json.load(open(f))
        for model, metrics in data.items():
            if isinstance(metrics, dict) and "DR" in metrics:
                vals[model].append(metrics["DR"]["value"])
    for f in sorted((ROOT / "experiments/results" / ds).glob("cold_start_seed*.json")):
        data = json.load(open(f))
        for model, metrics in data.items():
            if isinstance(metrics, dict) and "DR" in metrics:
                vals[model].append(metrics["DR"]["value"])
    return dict(vals)

cold = {}
for ds in ["obd-all", "obd-men", "obd-women", "kuairec", "kuairand"]:
    c = load_cold(ds)
    cold[ds] = c
    print(f"\n  {ds}:")
    for key, label in METHODS:
        if key in c:
            m, s = mean_std(c[key])
            print(f"    {label:<28} {m:.6f} ± {s:.6f} (n={len(c[key])})")
    if "GNN-Bandit" in c and "MF-Bandit" in c:
        print(f"    GNN lift vs MF-Bandit: {lift_pct(np.mean(c['GNN-Bandit']), np.mean(c['MF-Bandit'])):+.2f}%")
    if "GNN-Bandit" in c and "CQL" in c:
        print(f"    GNN lift vs CQL:       {lift_pct(np.mean(c['GNN-Bandit']), np.mean(c['CQL'])):+.2f}%")
    if "GNN-Bandit" in c and "BTS" in c:
        print(f"    GNN lift vs BTS:       {lift_pct(np.mean(c['GNN-Bandit']), np.mean(c['BTS'])):+.2f}%")

# Cold-start user counts from adjacency matrices (OBD)
print("\n  Zero-degree user counts from lightgcn_adj.npz:")
try:
    from scipy.sparse import load_npz
    for camp in ["all", "men", "women"]:
        base = ROOT / "data/processed_v2" / camp
        stats_json = json.load(open(base / "stats.json"))
        adj = load_npz(base / "lightgcn_adj.npz")
        n_users = stats_json["n_user_segments"]
        deg = np.asarray(adj[:n_users].sum(axis=1)).flatten()
        n_cold = int((deg == 0).sum())
        print(f"    OBD-{camp:<6} users={n_users}  zero-degree={n_cold}  ({n_cold/n_users*100:.1f}%)"
              f"  impressions={stats_json['n_impressions']:,}")
except Exception as e:
    print(f"    [SKIP] could not load OBD adjacency: {e}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("SECTION F: SENSITIVITY -- experiments/results/kuairec/sensitivity_seed*.json")
print("=" * 100)

sens_files = sorted((ROOT / "experiments/results/kuairec").glob("sensitivity_seed*.json"))
print(f"  files found: {[f.name for f in sens_files]}")
for f in sens_files:
    data = json.load(open(f))
    for param, sweep in data.items():
        print(f"  {param}:")
        for val, m in sweep.items():
            print(f"    {param}={val:<6} DR={m['DR_value']:.6f}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("SECTION G: CLAIM-BY-CLAIM VERIFICATION vs paper/main.tex")
print("=" * 100)

# Abstract / Table 1 claims (value, std, p, lift)
print("\n[A1] Table 1 GNN-Bandit cells:")
for ds, claimed_m, claimed_s in [
    ("obd-all", 0.008404, 0.000099), ("obd-men", 0.010301, 0.000342),
    ("obd-women", 0.010086, 0.000454), ("kuairec", 0.112655, 0.020033),
    ("kuairand", 0.453466, 0.006843), ("criteo", 0.002726, 0.000013)]:
    m, s = summary[ds]["GNN-Bandit"]
    check(f"{ds} GNN mean", claimed_m, m, tol=6e-7)
    check(f"{ds} GNN std", claimed_s, s, tol=6e-7)

print("\n[A2] Lift vs best baseline (true best):")
for ds, claimed in [("obd-all", 25.31), ("obd-men", 16.08), ("obd-women", 17.28),
                    ("kuairec", -9.85), ("kuairand", -9.88), ("criteo", -10.67)]:
    key, mu, p, l = best_baseline[ds]
    print(f"    {ds}: true best baseline = {key} ({mu:.6f}), computed lift = {l:+.2f}%  (paper claim {claimed:+.2f}%)")
    check(f"{ds} lift-vs-best", claimed, l, tol=0.06)

print("\n[A3] Lift vs logging policy (BTS):")
for ds, claimed in [("obd-all", 107.50), ("obd-men", 71.55), ("obd-women", 78.14),
                    ("kuairec", 60.89), ("kuairand", 13.50), ("criteo", 0.35)]:
    g = np.mean(main[ds]["GNN-Bandit"]); b = np.mean(main[ds]["BTS"])
    check(f"{ds} lift-vs-BTS", claimed, lift_pct(g, b), tol=0.06)

print("\n[A4] Abstract p-values:")
for ds, claimed_p in [("obd-all", 1.2e-5), ("obd-men", 7.8e-4), ("obd-women", 0.0018)]:
    key, mu, p, l = best_baseline[ds]
    check(f"{ds} p vs best ({key})", claimed_p, p, tol=max(claimed_p * 0.15, 1e-7))

print("\n[A5] Is GNN-Bandit the top method per dataset (bolding correctness)?")
for ds in DATASETS:
    gnn_m = summary[ds]["GNN-Bandit"][0]
    top = max((summary[ds][k][0], k) for k, _ in METHODS if k in summary[ds])
    verdict = "OK-to-bold" if top[1] == "GNN-Bandit" else f"WRONG -- top is {top[1]} ({top[0]:.6f})"
    print(f"    {ds:<10} GNN={gnn_m:.6f}  -> {verdict}")

print("\n[A6] KuaiRand claims in text (trails LinUCB by 9.88%, p=7.99e-4):")
ds = "kuairand"
gnn = np.asarray(main[ds]["GNN-Bandit"])
dt = np.asarray(main[ds]["DecisionTransformer"])
_, p_dt = stats.ttest_rel(gnn, dt)
lin = np.asarray(main[ds]["LinUCB"])
_, p_lin = stats.ttest_rel(gnn, lin)
print(f"    vs DT:      lift={lift_pct(gnn.mean(), dt.mean()):+.2f}%  p={p_dt:.2e}")
print(f"    vs LinUCB:  lift={lift_pct(gnn.mean(), lin.mean()):+.2f}%  p={p_lin:.2e}  (LinUCB is the true best baseline)")
check("kuairand lift vs LinUCB", -9.88, lift_pct(gnn.mean(), lin.mean()), tol=0.06)
check("kuairand p vs LinUCB", 7.99e-4, p_lin, tol=1e-4)

print("\n[T1] TGN table claims (LightGCN vs TGN):")
for ds, claimed_adv, claimed_p in [("obd-all", 14.27, 0.0280), ("obd-men", 1.18, 0.7224),
                                   ("obd-women", 10.85, 0.0455), ("criteo", 0.52, 0.9314)]:
    tgn_dr, _ = load_dr("experiments/results_tgn", ds)
    lg = np.asarray(main[ds]["GNN-Bandit"]); tg = np.asarray(tgn_dr["GNN-Bandit"])
    n = min(len(lg), len(tg))
    _, p = stats.ttest_rel(lg[:n], tg[:n])
    adv = lift_pct(lg[:n].mean(), tg[:n].mean())
    check(f"{ds} TGN adv%", claimed_adv, adv, tol=0.06)
    check(f"{ds} TGN p", claimed_p, p, tol=0.002)

print("\n[L1] lambda_CFR table claims (paper: lambda=0.05/0.10/0.20 rows):")
paper_lam = {
    ("obd-all", "0.05"): (0.008501, 0.000158, 1.85), ("obd-all", "0.10"): (0.007924, 0.000894, 11.28),
    ("obd-all", "0.20"): (0.006632, 0.000862, 12.99),
    ("obd-men", "0.05"): (0.008891, 0.001162, 13.07), ("obd-men", "0.10"): (0.009425, 0.001082, 11.48),
    ("obd-men", "0.20"): (0.008521, 0.001214, 14.24),
    ("obd-women", "0.05"): (0.010181, 0.000213, 2.09), ("obd-women", "0.10"): (0.009281, 0.001534, 16.52),
    ("obd-women", "0.20"): (0.009258, 0.001217, 13.15),
    ("criteo", "0.05"): (0.002515, 0.000272, 10.80), ("criteo", "0.10"): (0.002482, 0.000217, 8.74),
    ("criteo", "0.20"): (0.002592, 0.000237, 9.14),
}
for (ds, lam), (cm, cs, ccv) in paper_lam.items():
    if (ds, lam) in lam_summary:
        m, s, cv, n = lam_summary[(ds, lam)]
        check(f"{ds} lam={lam} mean", cm, m, tol=6e-7)
        check(f"{ds} lam={lam} std", cs, s, tol=6e-7)
        check(f"{ds} lam={lam} CV%", ccv, cv, tol=0.06)
    else:
        print(f"    [FAIL] {ds} lam={lam}: no sweep results on disk")

print("\n[AB1] Ablation collapse claims (OBD-All: No-Graph -41.71%, No-Constraint -51.11%):")
ds = "obd-all"
files = sorted((ROOT / "experiments/results" / ds).glob("ablation_seed*.json"))
vals = defaultdict(list)
for f in files:
    data = json.load(open(f))
    for k in ablation_keys:
        if k in data and "DR" in data[k]:
            vals[k].append(data[k]["DR"]["value"])
full = np.mean(vals["Full GNN-Bandit"])
check("No-Graph collapse %", 41.71, -lift_pct(np.mean(vals["No-Graph (BCQ only)"]), full), tol=0.06)
check("No-Constraint collapse %", 51.11, -lift_pct(np.mean(vals["No-Constraint (GNN+DQN)"]), full), tol=0.06)

print("\n[CS1] Cold-start claims:")
# OBD-Men: GNN 0.012080±0.000686, +42.86% vs MF; '205 zero-degree users (42.6%)'
if "obd-men" in cold and "GNN-Bandit" in cold["obd-men"]:
    m, s = mean_std(cold["obd-men"]["GNN-Bandit"])
    check("obd-men cold GNN mean", 0.012080, m, tol=6e-7)
    check("obd-men cold GNN std", 0.000686, s, tol=6e-7)
    if "MF-Bandit" in cold["obd-men"]:
        check("obd-men cold lift vs MF", 42.87,
              lift_pct(m, np.mean(cold["obd-men"]["MF-Bandit"])), tol=0.06)
    if "CQL" in cold["obd-men"]:
        check("obd-men cold lift vs CQL", 13.81,
              lift_pct(m, np.mean(cold["obd-men"]["CQL"])), tol=0.06)
if "kuairec" in cold and "GNN-Bandit" in cold["kuairec"]:
    m, s = mean_std(cold["kuairec"]["GNN-Bandit"])
    check("kuairec cold GNN mean", 0.188931, m, tol=6e-7)
    check("kuairec cold GNN std", 0.038067, s, tol=6e-7)
    if "CQL" in cold["kuairec"]:
        check("kuairec cold lift vs CQL", 69.33,
              lift_pct(m, np.mean(cold["kuairec"]["CQL"])), tol=0.06)
    if "MF-Bandit" in cold["kuairec"]:
        check("kuairec cold lift vs MF", 379.97,
              lift_pct(m, np.mean(cold["kuairec"]["MF-Bandit"])), tol=0.5)
if "kuairand" in cold and "GNN-Bandit" in cold["kuairand"]:
    m, s = mean_std(cold["kuairand"]["GNN-Bandit"])
    check("kuairand cold GNN mean", 0.543158, m, tol=6e-7)
    check("kuairand cold GNN std", 0.007002, s, tol=6e-7)
    if "MF-Bandit" in cold["kuairand"]:
        check("kuairand cold lift vs MF", 15.02,
              lift_pct(m, np.mean(cold["kuairand"]["MF-Bandit"])), tol=0.06)
    if "BTS" in cold["kuairand"]:
        check("kuairand cold lift vs BTS", 15.79,
              lift_pct(m, np.mean(cold["kuairand"]["BTS"])), tol=0.06)

print("\n[S1] Sensitivity claims (single-seed table):")
if sens_files:
    sens = json.load(open(sens_files[0]))
    for param, val, claimed in [("embed_dim", "16", 0.087127), ("embed_dim", "32", 0.035681),
                                ("embed_dim", "64", 0.102454), ("embed_dim", "128", 0.102753),
                                ("n_layers", "1", 0.143878), ("n_layers", "2", 0.095720),
                                ("n_layers", "3", 0.144651), ("n_layers", "4", 0.094326),
                                ("bcq_threshold_ratio", "0.1", 0.144534), ("bcq_threshold_ratio", "0.3", 0.163661),
                                ("bcq_threshold_ratio", "0.5", 0.118501), ("bcq_threshold_ratio", "1.0", 0.146318),
                                ("bcq_threshold_ratio", "2.0", 0.138217),
                                ("cvar_alpha", "0.05", 0.150424), ("cvar_alpha", "0.1", 0.127423),
                                ("cvar_alpha", "0.25", 0.119946), ("cvar_alpha", "0.5", 0.103815),
                                ("cvar_alpha", "1.0", 0.133351)]:
        got = sens[param][val]["DR_value"]
        check(f"sens {param}={val}", claimed, got, tol=6e-7)

print("\n[D1] Dataset-level claims:")
try:
    st_all = json.load(open(ROOT / "data/processed_v2/all/stats.json"))
    obd_total = sum(json.load(open(ROOT / f"data/processed_v2/{c}/stats.json"))["n_impressions"]
                    for c in ["all", "men", "women"])
    print(f"    OBD impressions total (all+men+women): {obd_total:,}")
    print(f"    Paper claims '>11.5M logged interactions' across 6 datasets -- OBD alone is {obd_total/1e6:.1f}M")
except Exception as e:
    print(f"    [SKIP] {e}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print(f"VERIFICATION COMPLETE: {len(failures)} FAILURES")
print("=" * 100)
for name, claimed, computed in failures:
    print(f"  FAIL {name}: paper={claimed}  computed={round(computed, 8)}")
sys.exit(1 if failures else 0)
