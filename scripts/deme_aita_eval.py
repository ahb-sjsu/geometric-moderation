# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.
"""Quantitative evaluation of the DEME per-stakeholder moderator on AITA.

This evaluates the ACTUAL thesis system (not a classifier): each post is
compiled once per stakeholder ethos into a rank-4 DEME moral tensor via the
ErisML compiler's LLM extractor, and the per-stakeholder verdicts are
aggregated by worst-off + escalate. Ground truth is the AITA community verdict
label (NTA/NAH/ESH/YTA).

Operationalization (stated honestly; AITA gives no per-vote split):
  - author-culpability ordinal  NTA(0) < NAH(1) < ESH(2) < YTA(3)   -> E3
  - "moral conflict" = verdict in {ESH, NAH} (shared/mixed fault)    -> E1
  Circularity note: the `aita` ethos was fit on AITA-style judgments; we lead
  with the `dear_abby` ethos (fit on a DIFFERENT advice column) and report
  `aita` as the expected-stronger secondary.

Claims tested:
  E1  escalation rate higher on morally-complex (ESH/NAH) than clear (NTA/YTA)
  E2  the two ethos produce DIFFERENT verdicts at a non-trivial rate
  E3  worst-off restrictiveness correlates (Spearman) with author culpability

Env:
  ERISML_LLM_API_KEY   required for --extractor llm (NRP token)
  AITA_PER_CLASS       posts sampled per verdict class (default 80)
  AITA_EXTRACTOR       rule | llm   (default llm)
  AITA_RANK            DEME tensor rank (default 4)
  AITA_SEED            sampling seed (default 0)
  AITA_MAXCHARS        truncate post text (default 1800)
  AITA_OUT             output dir (default outputs/aita_eval)

Resumable: each completed post is appended to per_item.jsonl; re-running skips
ids already present.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa

# --- DEME aggregation constants (kept identical to app/deme_moderation.py) ---
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.deme_moderation import RESTRICTIVENESS, _ESCALATE_TIER  # noqa: E402

VERDICT_ORDINAL = {"NTA": 0, "NAH": 1, "ESH": 2, "YTA": 3}  # author culpability
CONTESTED = {"ESH", "NAH"}  # community-flagged shared/mixed fault

PROF = Path("C:/source/erisml-compiler/src/erisml_compiler/em_dag/profiles")
ETHOS = {
    "dear_abby": PROF / "dear_abby_socialchem_v0.1.yaml",  # primary (non-circular)
    "aita": PROF / "aita_socialchem_v0.1.yaml",            # secondary (circular)
}

PER_CLASS = int(os.environ.get("AITA_PER_CLASS", "80"))
EXTRACTOR = os.environ.get("AITA_EXTRACTOR", "llm")
RANK = int(os.environ.get("AITA_RANK", "4"))
SEED = int(os.environ.get("AITA_SEED", "0"))
MAXCHARS = int(os.environ.get("AITA_MAXCHARS", "1800"))
CONC = int(os.environ.get("AITA_CONCURRENCY", "6"))  # NRP fair-use: gpt-oss allows 16
OUT = Path(os.environ.get("AITA_OUT", "outputs/aita_eval"))

try:
    from ftfy import fix_text  # type: ignore
except Exception:  # pragma: no cover
    def fix_text(s: str) -> str:
        return s


def load_aita() -> list[dict]:
    files = sorted(glob.glob(
        "C:/Users/abptl/.cache/huggingface/datasets/"
        "OsamaBsher___aita-reddit-dataset/**/*.arrow", recursive=True))
    tabs = []
    for fp in files:
        with pa.memory_map(fp, "r") as src:
            try:
                tabs.append(pa.ipc.open_file(src).read_all())
            except Exception:
                src.seek(0)
                tabs.append(pa.ipc.open_stream(src).read_all())
    tbl = pa.concat_tables(tabs)
    cols = {c: tbl.column(c).to_pylist() for c in ("id", "title", "text", "verdict")}
    out = []
    for i in range(tbl.num_rows):
        v = (cols["verdict"][i] or "").strip().upper()
        if v in VERDICT_ORDINAL:
            out.append({"id": cols["id"][i], "title": cols["title"][i] or "",
                        "text": cols["text"][i] or "", "verdict": v})
    return out


def stratified_sample(rows: list[dict]) -> list[dict]:
    by = defaultdict(list)
    for r in rows:
        if len((r["title"] + r["text"]).strip()) >= 200:
            by[r["verdict"]].append(r)
    rng = __import__("random").Random(SEED)
    picked = []
    for v in sorted(VERDICT_ORDINAL):
        pool = by[v]
        rng.shuffle(pool)
        picked.extend(pool[:PER_CLASS])
    rng.shuffle(picked)
    return picked


def _make_adapter():
    """NRP adapter for the managed LLM endpoint.

    Defaults to `gpt-oss` (a standard high-throughput instruct model the NRP docs
    recommend for JSON extraction) rather than `qwen3` (a reasoning model that
    spends its whole token budget on hidden chain-of-thought and returns empty
    content). Thinking is disabled defensively (harmless on gpt-oss).

    Two efficiency/etiquette features:
      - cache: LLM extraction depends only on the text, not the ethos, so the
        second ethos's identical calls are served from cache (each post pays the
        LLM cost once, not twice).
      - retry with backoff + bounded concurrency upstream, per NRP fair-use.
    """
    from erisml_compiler.annotation.llm_extractor import ModelAdapter
    from openai import OpenAI

    class NRPAdapter(ModelAdapter):
        name = "nrp_gpt_oss"

        def __init__(self):
            import threading
            self.base_url = os.environ.get("ERISML_LLM_BASE_URL", "https://ellm.nrp-nautilus.io/v1")
            self.api_key = os.environ["ERISML_LLM_API_KEY"]
            self.model = os.environ.get("ERISML_LLM_MODEL", "gpt-oss")
            self.timeout = float(os.environ.get("ERISML_LLM_TIMEOUT_S", "120"))
            self.max_tokens = int(os.environ.get("ERISML_LLM_MAX_TOKENS", "4096"))
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)
            self._cache: dict[tuple[str, str], str] = {}
            self._lock = threading.Lock()
            self.n_calls = 0
            self.n_cache_hits = 0
            self.n_empty = 0

        def call(self, system: str, user: str, **kwargs) -> str:
            key = (system, user)
            with self._lock:
                cached = self._cache.get(key)
                if cached is not None:
                    self.n_cache_hits += 1
                    return cached
            last = None
            for attempt in range(5):
                try:
                    r = self._client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "system", "content": system},
                                  {"role": "user", "content": user}],
                        temperature=kwargs.get("temperature", 0.1),
                        max_tokens=kwargs.get("max_tokens", self.max_tokens),
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                    out = (r.choices[0].message.content or "").strip()
                    if not out:  # intermittent empty completion -> retry, then give up gracefully
                        last = "empty response"
                        if attempt < 4:
                            time.sleep(2 * (attempt + 1))
                            continue
                        with self._lock:
                            self.n_empty += 1
                        out = "[]"  # safe empty list: extractor yields 0 facts, no hard crash
                    with self._lock:
                        self._cache[key] = out
                        self.n_calls += 1
                    return out
                except Exception as e:  # noqa: BLE001
                    last = e
                    if attempt == 4:
                        raise
                    time.sleep(2 * (attempt + 1))  # fair-use: back off, then retry
            raise RuntimeError(f"all retries failed: {last}")

    return NRPAdapter()


def build_compiler():
    from erisml_compiler.pipeline.orchestrator import CompileOptions, compile_document
    from erisml_compiler.tiers import CompilerTier
    from erisml_compiler.canonicalizer.base import auto_canonicalizer
    canon = auto_canonicalizer()
    adapter = None
    if EXTRACTOR == "llm":
        adapter = _make_adapter()  # raises if ERISML_LLM_API_KEY unset
    return CompileOptions, compile_document, CompilerTier, canon, adapter


def restrictiveness(v: str | None) -> int:
    return RESTRICTIVENESS.get(v or "", 3)


def extract_features(ir) -> dict:
    """Pull the continuous, thesis-relevant signal off a compiled IR: the
    ethos-weighted 9-dimension moral vector, the DEME verdict + confidence, and
    the four framework projections' polarities/confidences (the coarse verdict
    saturates; the vector and the cross-framework disagreement do not)."""
    dv = ir.deme_verdict
    mv = {}
    if getattr(ir, "moral_vectors", None):
        try:
            md = ir.moral_vectors[0].model_dump()
            mv = {k: (v.get("value") if isinstance(v, dict) else None) for k, v in md.items()}
        except Exception:  # noqa: BLE001
            mv = {}
    proj = {}
    for fw, d in (getattr(ir, "projections", None) or {}).items():
        proj[fw] = {"polarity": d.get("polarity"), "confidence": d.get("confidence")}
    return {
        "ok": True,
        "verdict": getattr(dv, "verdict", None),
        "confidence": getattr(dv, "confidence", None),
        "n_facts": len(ir.ethical_facts or []),
        "n_commit": len(ir.commitments or []),
        "moral_vector": mv,
        "projections": proj,
    }


def aggregate(per: dict[str, dict]) -> dict:
    ok = [p for p in per.values() if p.get("ok")]
    if not ok:
        return {"ok": False}
    worst = max(ok, key=lambda p: restrictiveness(p["verdict"]))
    distinct = {p["verdict"] for p in ok}
    disagreement = len(distinct) > 1
    escalate_tier = any(p["verdict"] in _ESCALATE_TIER for p in ok)
    escalate = disagreement or escalate_tier
    return {
        "ok": True,
        "worst_off_verdict": worst["verdict"],
        "worst_off_restrictiveness": restrictiveness(worst["verdict"]),
        "disagreement": disagreement,
        "escalate_tier": escalate_tier,
        "escalated": escalate,
        "action": "ESCALATE_TO_HUMAN" if escalate else worst["verdict"],
    }


def main() -> None:
    import concurrent.futures as cf
    import threading

    OUT.mkdir(parents=True, exist_ok=True)
    item_path = OUT / "per_item.jsonl"
    done_ids = set()
    if item_path.exists():
        for line in item_path.read_text(encoding="utf-8").splitlines():
            try:
                done_ids.add(json.loads(line)["id"])
            except Exception:
                pass
    rows = stratified_sample(load_aita())
    pending = [r for r in rows if r["id"] not in done_ids]
    print(f"[setup] sampled {len(rows)} posts ({Counter(r['verdict'] for r in rows)}); "
          f"done={len(done_ids)} pending={len(pending)} extractor={EXTRACTOR} "
          f"rank={RANK} concurrency={CONC}", flush=True)
    if not pending:
        print("[done] nothing to do", flush=True)
        return

    CompileOptions, compile_document, CompilerTier, canon, adapter = build_compiler()
    tmpdir = Path(tempfile.mkdtemp(prefix="aita_"))

    def compile_one(r: dict) -> dict:
        text = fix_text((r["title"] + "\n" + r["text"]).strip())[:MAXCHARS]
        f = tmpdir / f"{r['id']}.txt"
        f.write_text(text, encoding="utf-8")
        try:
            per: dict[str, dict] = {}
            for ename, ep in ETHOS.items():
                try:
                    opts = CompileOptions(
                        tier=CompilerTier.auto_detect(f), extractor=EXTRACTOR,
                        ethos_profile=ep, canonicalizer=canon,
                        llm_adapter=adapter, tensor_rank=RANK)
                    ir = compile_document(f, opts)
                    per[ename] = extract_features(ir)
                except Exception as e:  # noqa: BLE001
                    per[ename] = {"ok": False, "reason": f"{type(e).__name__}: {e}"[:200]}
            return {"id": r["id"], "verdict_label": r["verdict"],
                    "ordinal": VERDICT_ORDINAL[r["verdict"]],
                    "contested": r["verdict"] in CONTESTED,
                    "per_stakeholder": per, "agg": aggregate(per)}
        finally:
            try:
                f.unlink()
            except OSError:
                pass

    t_start = time.time()
    n = 0
    lock = threading.Lock()
    with item_path.open("a", encoding="utf-8") as sink, \
            cf.ThreadPoolExecutor(max_workers=CONC) as ex:
        futs = {ex.submit(compile_one, r): r for r in pending}
        for fut in cf.as_completed(futs):
            r = futs[fut]
            try:
                row = fut.result()
            except Exception as e:  # noqa: BLE001
                row = {"id": r["id"], "verdict_label": r["verdict"],
                       "error": f"{type(e).__name__}: {e}"[:200]}
            with lock:
                sink.write(json.dumps(row) + "\n")
                sink.flush()
                n += 1
                if n % 10 == 0 or n == len(pending):
                    rate = (time.time() - t_start) / n
                    print(f"[{n}/{len(pending)}] {rate:.1f}s/post wall "
                          f"est_remain={(len(pending)-n)*rate/60:.0f}min "
                          f"calls={getattr(adapter,'n_calls',0)} "
                          f"empty={getattr(adapter,'n_empty',0)}", flush=True)
    stats = (f" | LLM calls={getattr(adapter,'n_calls',0)} "
             f"cache_hits={getattr(adapter,'n_cache_hits',0)} "
             f"empty={getattr(adapter,'n_empty',0)}") if adapter else ""
    print(f"[done] {n} posts in {(time.time()-t_start)/60:.1f} min -> {item_path}{stats}", flush=True)


if __name__ == "__main__":
    main()
