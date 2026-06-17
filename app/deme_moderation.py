# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Per-stakeholder conversation moderation via DEME / the ErisML compiler.

This is the thesis system (not a classifier): a conversation is compiled into a
DEME moral tensor *once per stakeholder ethos*, and the framework enforces the
stakeholders' encoded values. A stakeholder's values are an Ethics-Module
weighting (an "ethos profile"); evaluating the same content under different
ethos yields different per-stakeholder verdicts. The platform's policy then
aggregates those verdicts.

Aggregation policy (this build): **worst-off + escalate**
  - worst-off : the decision protects the most-harmed party -> take the MOST
                RESTRICTIVE verdict across stakeholders.
  - escalate  : if stakeholders DISAGREE, or any verdict is in the escalate
                tier (requires_human_review / *_escalate), route to a human
                rather than silently aggregating.

Stakeholders here are the three *shipped* ethos as stand-ins; real deployments
construct stakeholder EMs by elicitation (e.g. the sqnd-probe "Dear Ethicist"
game), data-fitting, or authoring, and drop them in as additional profiles.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

ERIS_BIN = os.environ.get("ERIS_COMPILE_BIN", "eris-compile")
ERIS_TIMEOUT = int(os.environ.get("ERIS_TIMEOUT", "300"))
RANK = int(os.environ.get("DEME_RANK", "4"))


def _profiles_dir() -> Path | None:
    if os.environ.get("ERIS_PROFILES_DIR"):
        return Path(os.environ["ERIS_PROFILES_DIR"])
    try:
        import importlib.util
        spec = importlib.util.find_spec("erisml_compiler")
        if spec and spec.origin:
            return Path(spec.origin).parent / "em_dag" / "profiles"
    except Exception:
        pass
    return None


# Stand-in stakeholders = shipped ethos *weights* profiles. (default.yaml is a
# DAG profile, not an ethos-weights profile, so it is NOT a valid stakeholder
# here.) In production these are replaced/augmented by elicited stakeholder EMs
# (the sqnd-probe "Dear Ethicist" game), data-fitting, or authored weights.
DEFAULT_STAKEHOLDERS = {
    "harm_care_advocate": "dear_abby_socialchem_v0.1.yaml",   # weights harm/care
    "fairness_community": "aita_socialchem_v0.1.yaml",         # weights fairness/fidelity
}

# Restrictiveness ordering for the worst-off rule (higher = more protective).
RESTRICTIVENESS = {
    "prefer": 0,
    "neutral": 1,
    "permitted": 1,
    "requires_human_review": 3,
    "escalate": 3,
    "tragic_conflict_escalate": 3,
    "forbid": 4,
    "forbidden": 4,
}
_ESCALATE_TIER = {"requires_human_review", "escalate", "tragic_conflict_escalate"}


def _compile_under_ethos(text: str, ethos_path: Path, rank: int) -> dict[str, Any]:
    """Compile `text` under one ethos profile; return its DEME verdict + audit."""
    tmp = Path(tempfile.mkdtemp(prefix="deme_"))
    src, out = tmp / "content.txt", tmp / "ir.json"
    try:
        src.write_text(text, encoding="utf-8")
        proc = subprocess.run(
            [ERIS_BIN, "compile", str(src), "--extractor", "rule",
             "--rank", str(rank), "--ethos-profile", str(ethos_path), "--out", str(out)],
            capture_output=True, text=True, timeout=ERIS_TIMEOUT,
        )
        if proc.returncode != 0 or not out.is_file():
            return {"ok": False, "reason": (proc.stderr or proc.stdout or "compile failed")[-300:]}
        ir = json.loads(out.read_text(encoding="utf-8"))
        dv = ir.get("deme_verdict") or {}
        mt = ir.get("moral_tensor_v3") or {}
        proof = ir.get("decision_proof") or {}
        return {
            "ok": True,
            "verdict": dv.get("verdict"),
            "confidence": dv.get("confidence"),
            "rationale": (dv.get("rationale") or "")[:300],
            "tensor_rank": mt.get("rank"),
            "tensor_shape": mt.get("shape"),
            "profile_hash": proof.get("profile_hash"),
            "decision_id": proof.get("decision_id"),
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": f"timeout after {ERIS_TIMEOUT}s"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def moderate_conversation(
    text: str,
    stakeholders: dict[str, str] | None = None,
    rank: int = RANK,
) -> dict[str, Any]:
    """Moderate a conversation per stakeholder, aggregating worst-off + escalate."""
    stakeholders = stakeholders or DEFAULT_STAKEHOLDERS
    pdir = _profiles_dir()
    if pdir is None:
        return {"ok": False, "reason": "could not locate erisml_compiler ethos profiles"}

    per: list[dict[str, Any]] = []
    for name, fname in stakeholders.items():
        res = _compile_under_ethos(text, pdir / fname, rank)
        res["stakeholder"] = name
        res["ethos"] = fname
        per.append(res)

    ok = [p for p in per if p.get("ok")]
    if not ok:
        return {"ok": False, "reason": "all stakeholder compiles failed", "per_stakeholder": per}

    # --- worst-off: most restrictive verdict wins ---
    def rank_of(v: str | None) -> int:
        return RESTRICTIVENESS.get(v or "", 3)  # unknown -> escalate tier

    worst = max(ok, key=lambda p: rank_of(p["verdict"]))
    worst_off_verdict = worst["verdict"]

    # --- escalate: disagreement OR any escalate-tier verdict ---
    distinct = {p["verdict"] for p in ok}
    disagreement = len(distinct) > 1
    escalate_tier = any(p["verdict"] in _ESCALATE_TIER for p in ok)
    escalate = disagreement or escalate_tier

    action = "ESCALATE_TO_HUMAN" if escalate else worst_off_verdict

    return {
        "ok": True,
        "action": action,
        "worst_off": {"stakeholder": worst["stakeholder"], "verdict": worst_off_verdict},
        "escalated": escalate,
        "reason": ("stakeholder values disagree" if disagreement
                   else "escalate-tier verdict" if escalate_tier
                   else "stakeholders concur"),
        "per_stakeholder": [
            {"stakeholder": p["stakeholder"], "verdict": p.get("verdict"),
             "confidence": p.get("confidence"), "ok": p.get("ok"),
             "rationale": p.get("rationale"), "profile_hash": p.get("profile_hash"),
             "reason": p.get("reason")}
            for p in per
        ],
        "policy": "worst_off+escalate",
        "rank": rank,
    }


if __name__ == "__main__":
    # Demo: a borderline incivility/harassment thread where stakeholder ethos
    # are likely to diverge -> escalation.
    THREAD = (
        "UserA: You clearly have no idea what you're talking about, as usual.\n"
        "UserB: Maybe if you actually read the article before commenting you "
        "wouldn't embarrass yourself in front of everyone.\n"
        "UserA: I'm done being polite to people like you. Everyone can see "
        "exactly what you are.\n"
    )
    result = moderate_conversation(THREAD)
    print(json.dumps(result, indent=2))
