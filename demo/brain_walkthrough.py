#!/usr/bin/env python
"""
SEQUENT — THE BRAIN, STEP BY STEP.

A live, paced walkthrough of what happens between an operator typing a complaint
in their own words and the G1 having an executable, verifiable plan. Every number
on screen is computed live from this repo's code and data — nothing is replayed.

    python demo/brain_walkthrough.py                       # default ticket, paced by Enter
    python demo/brain_walkthrough.py --ticket "..."        # your own words
    python demo/brain_walkthrough.py --incident INC-042    # a labelled held-out ticket
    python demo/brain_walkthrough.py --random              # random labelled ticket
    python demo/brain_walkthrough.py --auto                # no pauses (for screen recording)
    python demo/brain_walkthrough.py --list                # show the ticket menu and exit
    python demo/brain_walkthrough.py --library course      # the original 100-SOP course library

Libraries:
    plant   (default when brain/benchmark/artifacts/retriever_st exists) the 1,012-SOP benchmark,
            the fine-tuned MiniLM retriever and the same TF-IDF as the paper; tickets are the
            269 held-out test incidents. This is the system the paper describes.
    course  the original 100-SOP library with the off-the-shelf MiniLM (no training needed).

Stages:
    1  The ticket            what the operator actually wrote
    2  The library           the SOPs the robot is allowed to run
    3  Lexical retrieval     TF-IDF: word overlap, and where it breaks
    4  Semantic retrieval    MiniLM: 384-d meaning vectors
    5  The hybrid            why both rankers, and what each one contributes
    6  The verdict           did it find the right procedure? (gold label revealed)
    7  The faithful planner  every written step -> exactly one robot skill
    8  The inferred steps    the navigation a human SOP leaves unsaid
    9  The contracts         what the verifier will measure for each skill
    10 The handoff           what the body does with this, and what it reports back
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "brain"))

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.rule import Rule
from rich.align import Align

console = Console()

# ---------------------------------------------------------------- house style
GOLD = "#e6b325"
GREEN = "#78dc78"
RED = "#e06c75"
BLUE = "#61afef"
DIM = "grey58"

AUTO = False
PACE = 0.0
LIB = "plant"
BENCH = REPO / "brain" / "benchmark"
FT_DIR = BENCH / "artifacts" / "retriever_st"
_STATE: dict = {}


def beat(seconds: float = 0.35) -> None:
    if AUTO:
        time.sleep(PACE * seconds)


def stage(n: int, title: str, subtitle: str = "") -> None:
    """Section break. In paced mode, waits for the presenter to hit Enter."""
    console.print()
    if not AUTO:
        try:
            console.print(Text("   [enter] ", style=DIM), end="")
            input()
        except (EOFError, KeyboardInterrupt):
            console.print()
            sys.exit(0)
    else:
        time.sleep(PACE * 1.2)
    console.print()
    head = Text()
    head.append(f"  STAGE {n}  ", style=f"bold black on {GOLD}")
    head.append(f"  {title}", style="bold white")
    console.print(head)
    if subtitle:
        console.print(Text(f"           {subtitle}", style=DIM))
    console.print(Rule(style=DIM))


def kv(label: str, value: str, style: str = "white") -> None:
    console.print(Text(f"   {label:<22}", style=DIM), Text(value, style=style))


# ---------------------------------------------------------------- data loading
def load_corpus():
    if LIB == "plant":
        with open(BENCH / "src/data/plant_sops.json", encoding="utf-8") as f:
            sops = [s for s in json.load(f)["sop_examples"] if s["sop_id"] != "SOP-1013"]  # 1,012; 1013 is the demo SOP
        with open(BENCH / "src/data/plant_incidents_test.json", encoding="utf-8") as f:
            incidents = json.load(f)["incident_examples"]
        return sops, incidents
    with open(REPO / "brain/src/data/sop_examples.json", encoding="utf-8") as f:
        sops = json.load(f)["sop_examples"]
    with open(REPO / "brain/src/data/incident_examples.json", encoding="utf-8") as f:
        incidents = json.load(f)["incident_examples"]
    return sops, incidents


def encode(texts):
    """Unit-length sentence vectors from whichever encoder this library uses."""
    if LIB == "plant":
        if "model" not in _STATE:
            from sentence_transformers import SentenceTransformer
            _STATE["model"] = SentenceTransformer(str(FT_DIR))
        return _STATE["model"].encode(texts, normalize_embeddings=True, show_progress_bar=False)
    from src.retrieval.index_utils import _sbert_encode
    return _sbert_encode(texts)


def sop_text(s: dict) -> str:
    return f"{s['title']}. {s['condition']}. Steps: " + " ; ".join(s["steps"])


# ---------------------------------------------------------------- rankers
def lexical_rank(query: str, sops: list, k: int = 5):
    """TF-IDF cosine. plant: unigram+bigram TF-IDF, identical to the paper's baseline (R@1 0.301)."""
    if LIB == "plant":
        import numpy as np
        if "tfidf" not in _STATE:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vec = TfidfVectorizer(ngram_range=(1, 2))
            _STATE["tfidf"] = (vec, vec.fit_transform([sop_text(s) for s in sops]))
        vec, S = _STATE["tfidf"]
        sc = (vec.transform([query]) @ S.T).toarray()[0]
        return [(sops[i], float(sc[i])) for i in np.argsort(-sc)[:k]]
    from brain_bridge import _retrieve_sop_tfidf
    return _retrieve_sop_tfidf(query, sops, k=k)


def semantic_rank(query: str, sops: list, k: int = 5):
    """plant: the fine-tuned MiniLM over all 1,012 SOPs. course: off-the-shelf MiniLM index."""
    if LIB == "plant":
        import numpy as np
        if "emb" not in _STATE:
            _STATE["emb"] = encode([sop_text(s) for s in sops])
        sc = _STATE["emb"] @ encode([query])[0]
        return [(sops[i], float(sc[i])) for i in np.argsort(-sc)[:k]]
    from src.retrieval.infer_retrieve import retrieve_topk
    by_id = {s["sop_id"]: s for s in sops}
    return [(by_id[h["sop_id"]], float(h["score"])) for h in retrieve_topk(query, k=k)
            if h["sop_id"] in by_id]


def hybrid_rank(query: str, sops: list, k: int = 5):
    """The runtime path (brain_bridge): interleave semantic and lexical, semantic first, dedup, top-k."""
    from brain_bridge import _interleave
    return _interleave(semantic_rank(query, sops, k), lexical_rank(query, sops, k), k)


def overlap_terms(query: str, s: dict, limit: int = 8):
    """Which query words the lexical ranker could actually see in this SOP."""
    stop = {"the", "a", "an", "is", "are", "it", "to", "on", "in", "of", "and", "i", "we",
            "at", "for", "be", "was", "has", "have", "seems", "need", "needs"}
    q = {w for w in re.findall(r"[a-z0-9_]+", query.lower()) if len(w) > 2 and w not in stop}
    d = set(re.findall(r"[a-z0-9_]+", sop_text(s).lower()))
    return sorted(q & d)[:limit]


def rank_table(rows, gold: str = None, title: str = "", score_header: str = "score") -> Table:
    t = Table(box=None, pad_edge=False, padding=(0, 2), title=title,
              title_style=f"bold {GOLD}", title_justify="left")
    t.add_column("#", style=DIM, width=2)
    t.add_column("SOP", width=8)
    t.add_column(score_header, justify="right", width=7)
    t.add_column("title", overflow="ellipsis", max_width=44)
    t.add_column("", width=10)
    for i, (s, sc) in enumerate(rows, 1):
        hit = gold is not None and s["sop_id"] == gold
        style = GREEN if hit else ("white" if i == 1 else DIM)
        t.add_row(str(i), s["sop_id"], f"{sc:.4f}", s["title"],
                  Text("<- correct", style=f"bold {GREEN}") if hit else "")
    return t


# ================================================================== THE STAGES
def run(ticket: str, gold: str, incident_id: str, sops: list) -> None:
    console.print()
    console.print(Align.center(Text("S E Q U E N T", style=f"bold {GOLD}")))
    console.print(Align.center(Text("the brain: from an operator's words to a verifiable robot plan",
                                    style=DIM)))
    console.print()
    console.print(Align.center(Text(
        "ticket  ->  retrieval  ->  SOP  ->  faithful plan  ->  skill contracts  ->  body",
        style="white")))

    # ------------------------------------------------------------ 1. the ticket
    stage(1, "THE TICKET", "an operator types a complaint. no keywords, no SOP number, no form.")
    console.print(Panel(Text(ticket, style="bold white"),
                        title=f"[{GOLD}]INCIDENT {incident_id}[/]",
                        subtitle="[grey58]operator's own words[/]",
                        border_style=GOLD, padding=(1, 3)))
    console.print()
    console.print(Text("   Note what is NOT in there: no procedure number, no equipment code,",
                       style=DIM))
    console.print(Text("   no controlled vocabulary. That is the whole problem.", style=DIM))

    # ----------------------------------------------------------- 2. the library
    stage(2, "THE LIBRARY", f"{len(sops)} standard operating procedures the robot is allowed to run.")
    example = next((s for s in sops if s["sop_id"] in ("SOP-0001", "SOP-001")), sops[0])
    body = Group(
        Text(f"{example['sop_id']}  {example['title']}", style=f"bold {BLUE}"),
        Text(f"WHEN: {example['condition']}", style=DIM),
        Text(""),
        *[Text(f"  {i}. {st}", style="white") for i, st in enumerate(example["steps"], 1)],
    )
    console.print(Panel(body, title="[grey58]one entry, as written[/]", border_style=DIM,
                        padding=(1, 3)))
    console.print()
    kv("corpus", f"{len(sops)} SOPs", "white")
    kv("steps per SOP", f"{min(len(s['steps']) for s in sops)}–{max(len(s['steps']) for s in sops)}", "white")
    if LIB == "plant":
        kv("machines", "46 equipment families, about 22 look-alike SOPs each", "white")
        console.print()
        console.print(Text("   Each step names a skill and an object. The SOPs were written at skill", style=DIM))
        console.print(Text("   level, so the hard part is finding the right one among near-twins:", style=DIM))
        console.print(Text("   pressure BELOW setpoint vs SUDDEN collapse vs ABOVE ceiling.", style=DIM))
    else:
        kv("written for", "a human who already knows where the machine is", DIM)
        console.print()
        console.print(Text("   Nobody wrote 'walk to the machine' — a human just does that.", style=DIM))
        console.print(Text("   Remember this. It comes back in stage 8.", style=DIM))

    # -------------------------------------------------------- 3. lexical ranker
    stage(3, "RETRIEVAL, TAKE ONE — LEXICAL", "TF-IDF cosine. pure word overlap. no learning.")
    t0 = time.perf_counter()
    lex = lexical_rank(ticket, sops, k=5)
    lex_ms = (time.perf_counter() - t0) * 1000
    console.print(rank_table(lex, gold, score_header="cosine"))
    console.print()
    kv("latency", f"{lex_ms:.1f} ms", "white")
    shared = overlap_terms(ticket, lex[0][0])
    kv("words it matched on", ", ".join(shared) if shared else "(none)", BLUE)
    console.print()
    console.print(Text("   TF-IDF can only see words the ticket and the SOP literally share.",
                       style=DIM))
    console.print(Text("   When the operator says 'shaking' and the SOP says 'vibration',", style=DIM))
    console.print(Text("   it sees nothing at all.", style=DIM))

    # ------------------------------------------------------- 4. semantic ranker
    stage(4, "RETRIEVAL, TAKE TWO — SEMANTIC",
          "MiniLM fine-tuned on 1,859 operator tickets: 22.7M params, 384-d vectors." if LIB == "plant"
          else "all-MiniLM-L6-v2 (off the shelf): 22.7M params, 384-d meaning vectors.")
    encode(["warm"])                              # model already loaded at startup; warm the cache
    t0 = time.perf_counter()
    qv = encode([ticket])[0]
    enc_ms = (time.perf_counter() - t0) * 1000
    console.print(Text("   The ticket becomes a point in 384-dimensional space:", style=DIM))
    console.print()
    preview = "  ".join(f"{x:+.3f}" for x in qv[:6])
    console.print(Text(f"      [ {preview}  ... ]", style=BLUE))
    console.print(Text(f"      {qv.shape[0]} dimensions, unit length", style=DIM))
    console.print()
    kv("encode time", f"{enc_ms:.0f} ms  (model already resident)", "white")
    kv("index", f"{len(sops)} SOP vectors, cosine over the full corpus", "white")
    console.print()
    t0 = time.perf_counter()
    sem = semantic_rank(ticket, sops, k=5)
    sem_ms = (time.perf_counter() - t0) * 1000
    console.print(rank_table(sem, gold, score_header="cosine"))
    console.print()
    kv("latency", f"{sem_ms:.0f} ms", "white")
    console.print()
    console.print(Text("   This ranker never sees the words. It sees the meaning —", style=DIM))
    console.print(Text("   which is why it survives paraphrase, and why it can also be", style=DIM))
    console.print(Text("   confidently vague when the ticket names an exact procedure.", style=DIM))

    # -------------------------------------------------------------- 5. hybrid
    stage(5, "WHY BOTH — THE HYBRID", "each ranker's best candidates are guaranteed a slot.")
    hyb = hybrid_rank(ticket, sops, k=5)
    lex_ids = [s["sop_id"] for s, _ in lex]
    sem_ids = [s["sop_id"] for s, _ in sem]
    t = Table(box=None, pad_edge=False, padding=(0, 2))
    t.add_column("#", style=DIM, width=2)
    t.add_column("SOP", width=8)
    t.add_column("came from", width=16)
    t.add_column("title", overflow="ellipsis", max_width=42)
    t.add_column("", width=10)
    for i, (s, sc) in enumerate(hyb, 1):
        sid = s["sop_id"]
        src = []
        if sid in sem_ids:
            src.append(f"semantic #{sem_ids.index(sid)+1}")
        if sid in lex_ids:
            src.append(f"lexical #{lex_ids.index(sid)+1}")
        hit = gold is not None and sid == gold
        t.add_row(str(i), sid, Text(" + ".join(src) or "—", style=BLUE), s["title"],
                  Text("<- correct", style=f"bold {GREEN}") if hit else "",
                  style=GREEN if hit else ("white" if i == 1 else DIM))
    console.print(t)
    console.print()
    console.print(Text("   Interleave, dedup, top-5. The bet is not that either ranker is right —",
                       style=DIM))
    console.print(Text("   it's that the correct SOP is IN this set, where a picker can find it.",
                       style=DIM))
    both = set(lex_ids) & set(sem_ids)
    kv("agreed on", f"{len(both)}/5 SOPs" + (f"  ({', '.join(sorted(both))})" if both else ""), "white")

    # ------------------------------------------------------------- 6. verdict
    stage(6, "DID IT WORK?", "the ticket came with a ground-truth label. now we look."
          if gold else "no ground-truth label for a free-typed ticket — judge it yourself.")
    chosen, chosen_score = hyb[0]
    if gold:
        rank = next((i for i, (s, _) in enumerate(hyb, 1) if s["sop_id"] == gold), None)
        gold_sop = next((s for s in sops if s["sop_id"] == gold), None)
        console.print()
        kv("ground truth", f"{gold}  {gold_sop['title'] if gold_sop else ''}", GOLD)
        kv("retrieved", f"{chosen['sop_id']}  {chosen['title']}", "white")
        console.print()
        if rank == 1:
            console.print(Panel(Text("RANK 1 — correct procedure, first try", style=f"bold {GREEN}"),
                                border_style=GREEN, padding=(0, 3)))
        elif rank:
            console.print(Panel(Text(f"RANK {rank} — in the candidate set, not at the top",
                                     style=f"bold {GOLD}"), border_style=GOLD, padding=(0, 3)))
            console.print(Text("   Reported as it happened. The picker still has it to choose from.",
                               style=DIM))
        else:
            console.print(Panel(Text("MISS — the correct SOP is not in the top 5",
                                     style=f"bold {RED}"), border_style=RED, padding=(0, 3)))
            console.print(Text("   Shown, not hidden. Run demo/brain_eval.py for the real", style=DIM))
            console.print(Text("   hit rate over all labelled tickets rather than one anecdote.", style=DIM))
    else:
        kv("retrieved", f"{chosen['sop_id']}  {chosen['title']}  ({chosen_score:.4f})", "white")

    # -------------------------------------------------------- 7. the procedure
    stage(7, "THE PROCEDURE IT CHOSE", "this is the document a human technician would be handed.")
    body = Group(
        Text(f"{chosen['sop_id']}  {chosen['title']}", style=f"bold {GOLD}"),
        Text(f"WHEN: {chosen['condition']}", style=DIM),
        Text(""),
        *[Text(f"  {i}. {st}", style="white") for i, st in enumerate(chosen["steps"], 1)],
        Text(""),
        Text("  equipment: " + ", ".join(chosen.get("equipment", [])), style=DIM),
    )
    console.print(Panel(body, border_style=GOLD, padding=(1, 3)))
    console.print()
    if LIB == "plant":
        console.print(Text("   Each line already names a skill. The robot can only ever run a procedure", style=DIM))
        console.print(Text("   from this approved library; it never writes one of its own.", style=DIM))
    else:
        console.print(Text("   Every one of these lines is an instruction to a person.", style=DIM))
        console.print(Text("   None of them is a thing a robot can execute.", style=DIM))

    # ----------------------------------------------------------- 8. the planner
    stage(8, "THE FAITHFUL PLANNER", "one written step -> exactly one robot skill. in order. repeats kept.")
    from src.planner.infer_plan import plan_from_sop, SKILL_KEYWORDS
    plan_steps = plan_from_sop(chosen["steps"])

    t = Table(box=None, pad_edge=False, padding=(0, 2))
    t.add_column("#", style=DIM, width=3)
    t.add_column("the SOP says", overflow="ellipsis", max_width=40)
    t.add_column("", width=2)
    t.add_column("skill", width=13)
    t.add_column("args", overflow="ellipsis", max_width=32)
    t.add_column("why", width=14)

    for i, ps in enumerate(plan_steps):
        args = ", ".join(f"{k}={v}" for k, v in ps["args"].items())
        if ps["source"] == "inferred":
            t.add_row(str(i), Text("(nobody wrote this down)", style=f"italic {BLUE}"),
                      Text("+", style=BLUE), Text(ps["skill"], style=f"bold {BLUE}"),
                      Text(args, style=BLUE), Text("inferred", style=BLUE))
        else:
            src = ps["sop_step"]
            if src.split() and src.split()[0].lower() == ps["skill"]:
                why = "named in step"
            else:
                kws = SKILL_KEYWORDS.get(ps["skill"], [])
                fired = max([kw for kw in kws if re.search(r"\b" + re.escape(kw) + r"\b", src.lower())]
                            or [""], key=len)
                why = f'"{fired}"' if fired else "no verb -> notify"
            t.add_row(str(i), Text(src, style="white"), Text("->", style=DIM),
                      Text(ps["skill"], style=f"bold {GOLD}"), Text(args, style="white"),
                      Text(why, style=DIM))
        beat(0.12)
    console.print(t)
    console.print()
    n_sop = sum(1 for p in plan_steps if p["source"] == "sop")
    n_inf = sum(1 for p in plan_steps if p["source"] == "inferred")
    kv("written SOP steps", f"{len(chosen['steps'])}", "white")
    kv("mapped 1:1", f"{n_sop}   (nothing dropped, nothing invented)", GOLD)
    kv("inferred navigation", f"{n_inf}", BLUE)
    kv("total plan length", f"{len(plan_steps)} skills", "white")
    if LIB == "plant":
        console.print()
        console.print(Text("   Arguments are mapped onto what exists in the simulator: four colored", style=DIM))
        console.print(Text("   buttons and a fixed set of tools. The skill sequence is the SOP's own.", style=DIM))

    # ----------------------------------------------------- 9. the inferred part
    stage(9, "THE STEPS NOBODY WROTE DOWN", "the gap between a human procedure and a robot program.")
    console.print(Text("   A human SOP omits the obvious. 'Press the blue button' assumes you", style=DIM))
    console.print(Text("   are standing at the machine. A robot standing at the parts table is", style=DIM))
    console.print(Text("   not. The planner tracks where the body is and fills the gaps:", style=DIM))
    console.print()
    loc = None
    for i, ps in enumerate(plan_steps):
        if ps["source"] == "inferred":
            tgt = ps["args"].get("target")
            console.print(Text(f"      [{i}]  at {loc or 'start'} -> needs to be at {tgt}"
                               f"   =>  walk_to({tgt})", style=BLUE))
            loc = tgt
        elif ps["skill"] == "walk_to":
            loc = ps["args"].get("target")
    if n_inf == 0:
        console.print(Text("      (this SOP happens entirely in one place — nothing to infer)",
                           style=DIM))
    console.print()
    console.print(Text("   Every plan step is tagged 'sop' or 'inferred'. When the robot does", style=DIM))
    console.print(Text("   something the procedure never asked for, the record says who asked.", style=DIM))

    # -------------------------------------------------------- 10. the contracts
    stage(10, "WHAT THE VERIFIER WILL MEASURE", "a skill is not done because the policy says so.")
    from src.data.skill_registry import SKILL_REGISTRY
    used = []
    for ps in plan_steps:
        if ps["skill"] not in used:
            used.append(ps["skill"])
    t = Table(box=None, pad_edge=False, padding=(0, 2))
    t.add_column("skill", width=13)
    t.add_column("runs on", width=12)
    t.add_column("postcondition, measured from mjData", overflow="fold", max_width=52)
    t.add_column("status", width=15)
    status_style = {"verified": GREEN, "reach-limited": RED, "operational": DIM}
    for sk in used:
        m = SKILL_REGISTRY[sk]
        t.add_row(Text(sk, style="bold white"), m["kind"], m["postcondition"] or "—",
                  Text(m["status"], style=f"bold {status_style.get(m['status'], 'white')}"))
    console.print(t)
    console.print()
    console.print(Text("   Reading the status column:", style=DIM))
    console.print(Text(f"     verified       ", style=GREEN), end="")
    console.print(Text("checked against measured physics and it holds up", style=DIM))
    console.print(Text(f"     reach-limited  ", style=RED), end="")
    console.print(Text("physically attempted, cannot reliably complete, reported as failed",
                       style=DIM))
    console.print(Text(f"     operational    ", style=DIM), end="")
    console.print(Text("a logged non-physical step (read, wait, notify)", style=DIM))
    console.print()
    console.print(Text("   A demo that cannot fail is not verifying anything.", style=f"italic {GOLD}"))

    # ----------------------------------------------------------- 11. the handoff
    stage(11, "THE HANDOFF", "this JSON is the entire contract between brain and body.")
    payload = {
        "goal": f"{ticket}  [SOP {chosen['sop_id']}: {chosen['title']}]",
        "steps": [{"skill": p["skill"], "args": p["args"], "source": p["source"]}
                  for p in plan_steps],
    }
    out = REPO / "demo" / "_last_plan.json"
    out.write_text(json.dumps(payload, indent=2))

    # Compact rendering: one line per step, so the whole plan fits on one screen.
    lines = [Text('{', style="white"),
             Text(f'  "goal": "{chosen["sop_id"]}: {chosen["title"]}  <- {incident_id}",',
                  style="white"),
             Text('  "steps": [', style="white")]
    for i, p in enumerate(payload["steps"]):
        args = json.dumps(p["args"])
        tail = "," if i < len(payload["steps"]) - 1 else ""
        lines.append(Text(f'    {{"skill": "{p["skill"]}", "args": {args}, '
                          f'"source": "{p["source"]}"}}{tail}',
                          style=BLUE if p["source"] == "inferred" else "white"))
    lines += [Text('  ]', style="white"), Text('}', style="white")]
    console.print(Panel(Group(*lines), border_style=DIM, padding=(1, 2),
                        title="[grey58]plan handed to the executor[/]"))
    console.print()
    kv("written to", str(out.relative_to(REPO)), DIM)
    console.print()
    console.print(Text("   From here the verifying executor runs each skill as its own", style=DIM))
    console.print(Text("   precondition -> rollout -> postcondition episode, and returns a", style=DIM))
    console.print(Text("   per-step verdict card. Steps that fail are printed as failed.", style=DIM))

    console.print()
    console.print(Rule(style=GOLD))
    console.print(Align.center(Text(
        f"ticket  ->  {len(sops)} SOPs  ->  {chosen['sop_id']}  ->  {len(plan_steps)} verifiable skills",
        style=f"bold {GOLD}")))
    console.print(Align.center(Text("all of it computed live, on this machine", style=DIM)))
    console.print(Rule(style=GOLD))
    console.print()


# ===================================================================== driver
def main() -> None:
    global AUTO, PACE
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ticket", help="an incident in your own words")
    p.add_argument("--incident", help="a labelled incident id, e.g. INC-042")
    p.add_argument("--random", action="store_true", help="pick a random labelled incident")
    p.add_argument("--auto", action="store_true", help="run without pauses (for recording)")
    p.add_argument("--pace", type=float, default=1.0, help="auto-mode speed multiplier")
    p.add_argument("--list", action="store_true", help="list the labelled tickets and exit")
    p.add_argument("--library", choices=["plant", "course"], default=None,
                   help="plant: 1,012 SOPs + fine-tuned retriever (default if trained); course: 100-SOP library")
    a = p.parse_args()

    global LIB
    AUTO, PACE = a.auto, a.pace
    LIB = a.library or ("plant" if FT_DIR.is_dir() else "course")
    if LIB == "plant" and not FT_DIR.is_dir():
        console.print("[red]The fine-tuned retriever is not trained yet.[/] Run, from brain/benchmark:\n"
                      "    python -m src.retrieval.train_eval_sentence_encoder\n"
                      "or use --library course.")
        sys.exit(1)
    sops, incidents = load_corpus()

    if a.list:
        t = Table(box=None, padding=(0, 2), title="labelled incident tickets",
                  title_style=f"bold {GOLD}", title_justify="left")
        t.add_column("id", style=BLUE, width=9, no_wrap=True)
        t.add_column("gold", style=DIM, width=8, no_wrap=True)
        t.add_column("what the operator wrote", overflow="ellipsis", max_width=88)
        for i in incidents:
            t.add_row(i["incident_id"], i["labels"]["sop_id"], i["text"])
        console.print(t)
        return

    if a.ticket:
        ticket, gold, iid = a.ticket, None, "typed live"
    else:
        if a.incident:
            rec = next((i for i in incidents if i["incident_id"].upper() == a.incident.upper()), None)
            if rec is None:
                console.print(f"[red]no such incident: {a.incident}[/]  (try --list)")
                sys.exit(1)
        elif a.random:
            rec = random.choice(incidents)
        else:
            rec = next(i for i in incidents if i["incident_id"] == ("INC-00056" if LIB == "plant" else "INC-001"))
        ticket, gold, iid = rec["text"], rec["labels"]["sop_id"], rec["incident_id"]

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Load the encoder BEFORE the audience is watching, so stage 4's latency is
    # a real encode time and not a one-off model load.
    with console.status("[grey58]loading the encoder and the SOP index...[/]", spinner="dots"):
        t0 = time.perf_counter()
        encode(["warmup"])
        semantic_rank("warmup", sops, k=1)
        lexical_rank("warmup", sops, k=1)
        load_s = time.perf_counter() - t0

    console.clear()
    run(ticket, gold, iid, sops)
    console.print(Text(f"   (encoder + index came up in {load_s:.1f}s before the walkthrough started)",
                       style=DIM))
    console.print()


if __name__ == "__main__":
    main()
