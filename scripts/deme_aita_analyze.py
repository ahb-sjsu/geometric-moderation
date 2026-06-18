# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.
"""Analyze the DEME-on-AITA evaluation (reads per_item.jsonl from deme_aita_eval).

Computes the three thesis claims from the continuous DEME outputs (the moral
vector + framework polarities), NOT the saturated categorical verdict:

  E1  moral conflict: does cross-framework disagreement / escalation concentrate
      on community-contested (ESH/NAH) posts vs clear (NTA/YTA)?
  E2  per-stakeholder differentiation: do the two ethos produce DIFFERENT moral
      vectors / verdicts, and by how much, on which dimensions?
  E3  validity: does a continuous harm severity (from the ethos-weighted moral
      vector) track the AITA author-culpability ordinal (NTA<NAH<ESH<YTA)?

Spearman/Mann-Whitney are computed with scipy if available, else a numpy
fallback (rank correlation + permutation-free normal approx).
Usage: python scripts/deme_aita_analyze.py [per_item.jsonl]
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.deme_moderation import _ESCALATE_TIER  # noqa: E402

ORD = {"NTA": 0, "NAH": 1, "ESH": 2, "YTA": 3}
CONTESTED = {"ESH", "NAH"}
PRIMARY = "dear_abby"   # non-circular ethos -> lead with this
SECONDARY = "aita"

try:
    from scipy.stats import spearmanr, mannwhitneyu  # type: ignore
    HAVE_SCIPY = True
except Exception:  # pragma: no cover
    HAVE_SCIPY = False


def _spearman(x, y):
    if HAVE_SCIPY:
        r = spearmanr(x, y)
        return float(r.correlation), float(r.pvalue)
    # numpy fallback: Pearson on ranks + t-approx
    import numpy as np
    def rank(a):
        a = np.asarray(a, float)
        order = a.argsort()
        r = np.empty_like(order, float)
        r[order] = np.arange(len(a))
        # average ties
        _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
        sums = np.zeros(len(cnt)); np.add.at(sums, inv, r)
        return (sums / cnt)[inv]
    rx, ry = rank(x), rank(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    n = len(x)
    if n > 3 and abs(rho) < 1:
        t = rho * math.sqrt((n - 2) / (1 - rho * rho))
        # two-sided normal approx
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    else:
        p = float("nan")
    return rho, p


def _harm_severity(mv: dict) -> float:
    """Total harm magnitude across dimensions: sum of -value for negative dims."""
    return sum(-float(v) for v in mv.values() if isinstance(v, (int, float)) and v < 0)


def _vec(mv: dict, keys):
    return [float(mv.get(k, 0.0) or 0.0) for k in keys]


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/aita_eval/per_item.jsonl")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    good = []
    for r in rows:
        ps = r.get("per_stakeholder", {})
        if ps.get(PRIMARY, {}).get("ok") and ps.get(SECONDARY, {}).get("ok"):
            good.append(r)
    print(f"loaded {len(rows)} rows; both-ethos-ok = {len(good)} "
          f"(scipy={'yes' if HAVE_SCIPY else 'no'})")
    if not good:
        return
    by_class = defaultdict(list)
    for r in good:
        by_class[r["verdict_label"]].append(r)
    print("  per class:", {k: len(v) for k, v in sorted(by_class.items(), key=lambda kv: ORD.get(kv[0], 9))})

    # ---- E3: harm severity (PRIMARY ethos) vs culpability ordinal ----
    print("\n=== E3  validity: harm severity vs AITA culpability ===")
    for ethos in (PRIMARY, SECONDARY):
        sev = [_harm_severity(r["per_stakeholder"][ethos].get("moral_vector", {})) for r in good]
        ordv = [r["ordinal"] for r in good]
        rho, p = _spearman(sev, ordv)
        print(f"  [{ethos:9}] Spearman(harm_severity, culpability) rho={rho:+.3f} p={p:.4f}")
    # mean harm severity by class (PRIMARY)
    print("  mean harm severity by class (primary):")
    for c in sorted(by_class, key=lambda k: ORD[k]):
        sevs = [_harm_severity(r["per_stakeholder"][PRIMARY].get("moral_vector", {})) for r in by_class[c]]
        print(f"    {c}: {sum(sevs)/len(sevs):.3f}  (n={len(sevs)})")
    # PER-DIMENSION Spearman vs culpability (which dims, if any, track the author label)
    keys0 = sorted({k for r in good for k in r["per_stakeholder"][PRIMARY].get("moral_vector", {})})
    ordv = [r["ordinal"] for r in good]
    print("  per-dimension Spearman(value, culpability) [primary ethos]:")
    dim_rhos = []
    for k in keys0:
        vals = [float(r["per_stakeholder"][PRIMARY].get("moral_vector", {}).get(k, 0) or 0) for r in good]
        if len(set(vals)) > 1:
            rho, p = _spearman(vals, ordv)
            dim_rhos.append((rho, p, k))
    for rho, p, k in sorted(dim_rhos, key=lambda t: -abs(t[0])):
        flag = " *" if p < 0.05 else ""
        print(f"    {k:24} rho={rho:+.3f} p={p:.4f}{flag}")

    # ---- E2: per-stakeholder differentiation ----
    print("\n=== E2  per-stakeholder differentiation (dear_abby vs aita) ===")
    keys = sorted({k for r in good for k in r["per_stakeholder"][PRIMARY].get("moral_vector", {})})
    dists, vmismatch = [], 0
    perdim = defaultdict(list)
    for r in good:
        a = r["per_stakeholder"][PRIMARY].get("moral_vector", {})
        b = r["per_stakeholder"][SECONDARY].get("moral_vector", {})
        va, vb = _vec(a, keys), _vec(b, keys)
        dists.append(math.sqrt(sum((x - y) ** 2 for x, y in zip(va, vb))))
        for k, x, y in zip(keys, va, vb):
            perdim[k].append(abs(x - y))
        if r["per_stakeholder"][PRIMARY].get("verdict") != r["per_stakeholder"][SECONDARY].get("verdict"):
            vmismatch += 1
    print(f"  mean L2 distance between ethos moral vectors = {sum(dists)/len(dists):.3f}")
    print(f"  verdict-divergence rate = {vmismatch}/{len(good)} = {vmismatch/len(good):.1%}")
    top = sorted(((sum(v)/len(v), k) for k, v in perdim.items()), reverse=True)[:4]
    print("  dimensions that diverge most (mean abs-diff): " + ", ".join(f"{k}={m:.3f}" for m, k in top))

    # ---- A: systematic DIRECTIONAL difference (signed dear_abby - aita per dim) ----
    print("\n=== A  systematic directional difference (signed mean dear_abby - aita) ===")
    signed = defaultdict(list)
    for r in good:
        a = r["per_stakeholder"][PRIMARY].get("moral_vector", {})
        b = r["per_stakeholder"][SECONDARY].get("moral_vector", {})
        for k in keys:
            signed[k].append(float(a.get(k, 0) or 0) - float(b.get(k, 0) or 0))
    for k in keys:
        d = signed[k]
        m = sum(d) / len(d)
        nz = [x for x in d if abs(x) > 1e-9]
        if not nz:
            continue
        pos = sum(1 for x in nz if x > 0)
        # sign test p (binomial, two-sided, normal approx)
        n = len(nz); k_ = max(pos, n - pos)
        z = (k_ - n / 2) / (math.sqrt(n) / 2) if n else 0
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        print(f"  {k:24} mean(da-aita)={m:+.3f}  nonzero={n}/{len(d)}  sign-test p={p:.3f}")

    # ---- B: governance effect of worst-off + escalate ----
    print("\n=== B  governance effect (worst-off + escalate vs single ethos) ===")
    diff_primary = diff_secondary = esc_from_disagree = 0
    for r in good:
        agg = r.get("agg", {})
        action = agg.get("action")
        pa = r["per_stakeholder"][PRIMARY].get("verdict")
        sa = r["per_stakeholder"][SECONDARY].get("verdict")
        # single-ethos action = its verdict, or ESCALATE if it's an escalate-tier verdict
        pa_action = "ESCALATE_TO_HUMAN" if pa in _ESCALATE_TIER else pa
        sa_action = "ESCALATE_TO_HUMAN" if sa in _ESCALATE_TIER else sa
        if action != pa_action:
            diff_primary += 1
        if action != sa_action:
            diff_secondary += 1
        if agg.get("disagreement") and not agg.get("escalate_tier"):
            esc_from_disagree += 1
    n = len(good)
    print(f"  joint decision differs from dear_abby-alone : {diff_primary}/{n} = {diff_primary/n:.1%}")
    print(f"  joint decision differs from aita-alone      : {diff_secondary}/{n} = {diff_secondary/n:.1%}")
    print(f"  escalations caused purely by ethos disagreement (not tier): {esc_from_disagree}/{n} = {esc_from_disagree/n:.1%}")

    # ---- D: cross-framework polarity disagreement (escalation is principled) ----
    print("\n=== D  cross-framework disagreement (the escalation trigger) ===")
    ndis = 0
    polcount = defaultdict(int)
    for r in good:
        proj = r["per_stakeholder"][PRIMARY].get("projections", {})
        pols = [d.get("polarity") for d in proj.values() if d.get("polarity")]
        if len(set(pols)) > 1:
            ndis += 1
        polcount[len(set(pols))] += 1
    print(f"  posts with >=2 distinct framework polarities: {ndis}/{len(good)} = {ndis/len(good):.1%}")
    print(f"  distribution of #distinct polarities: {dict(sorted(polcount.items()))}")

    # ---- E1: conflict concentration (contested vs clear) ----
    print("\n=== E1  conflict concentration: contested (ESH/NAH) vs clear (NTA/YTA) ===")
    def disagree_frac(r):
        proj = r["per_stakeholder"][PRIMARY].get("projections", {})
        pols = {d.get("polarity") for d in proj.values() if d.get("polarity")}
        return len(pols) > 1
    esc = {True: [], False: []}
    dis = {True: [], False: []}
    for r in good:
        c = r["verdict_label"] in CONTESTED
        esc[c].append(1 if r.get("agg", {}).get("escalated") else 0)
        dis[c].append(1 if disagree_frac(r) else 0)
    for label, grp in (("contested", True), ("clear", False)):
        e = esc[grp]; d = dis[grp]
        if e:
            print(f"  {label:9} (n={len(e)}): escalation={sum(e)/len(e):.1%}  "
                  f"cross-framework-disagreement={sum(d)/len(d):.1%}")
    # escalation + disagreement by class
    print("  by class:")
    for c in sorted(by_class, key=lambda k: ORD[k]):
        grp = by_class[c]
        e = sum(1 for r in grp if r.get("agg", {}).get("escalated")) / len(grp)
        d = sum(1 for r in grp if disagree_frac(r)) / len(grp)
        print(f"    {c}: escalation={e:.1%}  cross-framework-disagreement={d:.1%}  (n={len(grp)})")

    # ---- descriptive: mean moral vector by class (primary) ----
    print("\n=== descriptive: mean moral-vector dimensions by class (primary ethos) ===")
    print("  dim                       " + "  ".join(f"{c:>7}" for c in sorted(by_class, key=lambda k: ORD[k])))
    for k in keys:
        cells = []
        for c in sorted(by_class, key=lambda k2: ORD[k2]):
            vals = [float(r["per_stakeholder"][PRIMARY].get("moral_vector", {}).get(k, 0) or 0) for r in by_class[c]]
            cells.append(f"{sum(vals)/len(vals):+.3f}")
        print(f"  {k:24} " + "  ".join(f"{x:>7}" for x in cells))


if __name__ == "__main__":
    main()
