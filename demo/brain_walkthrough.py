#!/usr/bin/env python
"""
Sequent brain walkthrough.

Steps through ticket -> SOP -> plan, one stage per Enter. Everything is computed live.

    python demo/brain_walkthrough.py                       # default ticket, paced by Enter
    python demo/brain_walkthrough.py --ticket "..."        # your own words
    python demo/brain_walkthrough.py --incident INC-00026  # a labelled held-out ticket
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
    1 Ticket   2 SOP library   3 TF-IDF retrieval   4 MiniLM retrieval   5 Candidates   6 Result
    7 Selected SOP   8 Plan   9 Inserted navigation   10 Verifier checks   11 Output
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
    console.print()
    console.print(Align.center(Text(
        "ticket  ->  retrieval  ->  SOP  ->  plan  ->  verifier",
        style="white")))

    # ------------------------------------------------------------ 1. the ticket
    stage(1, "Ticket")
    console.print(Panel(Text(ticket, style="bold white"),
                        title=f"[{GOLD}]INCIDENT {incident_id}[/]",
                        border_style=GOLD, padding=(1, 3)))

    # ----------------------------------------------------------- 2. the library
    stage(2, "SOP library")
    example = next((s for s in sops if s["sop_id"] in ("SOP-0001", "SOP-001")), sops[0])
    body = Group(
        Text(f"{example['sop_id']}  {example['title']}", style=f"bold {BLUE}"),
        Text(f"WHEN: {example['condition']}", style=DIM),
        Text(""),
        *[Text(f"  {i}. {st}", style="white") for i, st in enumerate(example["steps"], 1)],
    )
    console.print(Panel(body, title="[grey58]example[/]", border_style=DIM,
                        padding=(1, 3)))
    console.print()
    kv("corpus", f"{len(sops)} SOPs", "white")
    kv("steps per SOP", f"{min(len(s['steps']) for s in sops)}–{max(len(s['steps']) for s in sops)}", "white")
    if LIB == "plant":
        kv("machines", "46 families, ~22 SOPs each", "white")

    # -------------------------------------------------------- 3. lexical ranker
    stage(3, "TF-IDF retrieval")
    t0 = time.perf_counter()
    lex = lexical_rank(ticket, sops, k=5)
    lex_ms = (time.perf_counter() - t0) * 1000
    console.print(rank_table(lex, gold, score_header="cosine"))
    console.print()
    kv("latency", f"{lex_ms:.1f} ms", "white")
    shared = overlap_terms(ticket, lex[0][0])
    kv("matched words", ", ".join(shared) if shared else "(none)", BLUE)

    # ------------------------------------------------------- 4. semantic ranker
    stage(4, "MiniLM retrieval", "fine-tuned on 1,859 tickets" if LIB == "plant" else "off the shelf")
    encode(["warm"])                              # model already loaded at startup; warm the cache
    t0 = time.perf_counter()
    qv = encode([ticket])[0]
    enc_ms = (time.perf_counter() - t0) * 1000
    preview = "  ".join(f"{x:+.3f}" for x in qv[:6])
    console.print(Text(f"      [ {preview}  ... ]", style=BLUE))
    console.print(Text(f"      {qv.shape[0]}-d query vector", style=DIM))
    console.print()
    kv("encode time", f"{enc_ms:.0f} ms", "white")
    kv("index", f"{len(sops)} SOPs", "white")
    console.print()
    t0 = time.perf_counter()
    sem = semantic_rank(ticket, sops, k=5)
    sem_ms = (time.perf_counter() - t0) * 1000
    console.print(rank_table(sem, gold, score_header="cosine"))
    console.print()
    kv("latency", f"{sem_ms:.0f} ms", "white")

    # -------------------------------------------------------------- 5. hybrid
    stage(5, "Candidates", "MiniLM and TF-IDF interleaved, top 5")
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
    both = set(lex_ids) & set(sem_ids)
    kv("in both top 5", f"{len(both)}/5 SOPs" + (f"  ({', '.join(sorted(both))})" if both else ""), "white")

    # ------------------------------------------------------------- 6. verdict
    stage(6, "Result", "" if gold else "typed ticket, no label")
    chosen, chosen_score = hyb[0]
    if gold:
        rank = next((i for i, (s, _) in enumerate(hyb, 1) if s["sop_id"] == gold), None)
        gold_sop = next((s for s in sops if s["sop_id"] == gold), None)
        console.print()
        kv("ground truth", f"{gold}  {gold_sop['title'] if gold_sop else ''}", GOLD)
        kv("retrieved", f"{chosen['sop_id']}  {chosen['title']}", "white")
        console.print()
        if rank == 1:
            console.print(Panel(Text("rank 1: correct", style=f"bold {GREEN}"),
                                border_style=GREEN, padding=(0, 3)))
        elif rank:
            console.print(Panel(Text(f"rank {rank}: correct SOP in candidates, not first",
                                     style=f"bold {GOLD}"), border_style=GOLD, padding=(0, 3)))
        else:
            console.print(Panel(Text("miss: correct SOP not in top 5",
                                     style=f"bold {RED}"), border_style=RED, padding=(0, 3)))
    else:
        kv("retrieved", f"{chosen['sop_id']}  {chosen['title']}  ({chosen_score:.4f})", "white")

    # -------------------------------------------------------- 7. the procedure
    stage(7, "Selected SOP")
    body = Group(
        Text(f"{chosen['sop_id']}  {chosen['title']}", style=f"bold {GOLD}"),
        Text(f"WHEN: {chosen['condition']}", style=DIM),
        Text(""),
        *[Text(f"  {i}. {st}", style="white") for i, st in enumerate(chosen["steps"], 1)],
        Text(""),
        Text("  equipment: " + ", ".join(chosen.get("equipment", [])), style=DIM),
    )
    console.print(Panel(body, border_style=GOLD, padding=(1, 3)))

    # ----------------------------------------------------------- 8. the planner
    stage(8, "Plan", "one skill per SOP step")
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
            t.add_row(str(i), Text("(inserted)", style=f"italic {BLUE}"),
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
    kv("SOP steps", f"{len(chosen['steps'])}", "white")
    kv("mapped", f"{n_sop}", GOLD)
    kv("inserted walk_to", f"{n_inf}", BLUE)
    kv("plan length", f"{len(plan_steps)} skills", "white")
    if LIB == "plant":
        kv("arguments", "mapped to simulator objects", DIM)

    # ----------------------------------------------------- 9. the inferred part
    stage(9, "Inserted navigation", "walk_to added where the location changes")
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
        console.print(Text("      (none: no location change)",
                           style=DIM))

    # -------------------------------------------------------- 10. the contracts
    stage(10, "Verifier checks", "postconditions measured from simulator state")
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
    status_style = {"verified": GREEN, "partial": GOLD, "reach-limited": RED, "operational": DIM}
    for sk in used:
        m = SKILL_REGISTRY[sk]
        t.add_row(Text(sk, style="bold white"), m["kind"], m["postcondition"] or "—",
                  Text(m["status"], style=f"bold {status_style.get(m['status'], 'white')}"))
    console.print(t)
    console.print()

    # ----------------------------------------------------------- 11. the handoff
    stage(11, "Output", "plan sent to the executor")
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
                        title="[grey58]plan[/]"))
    console.print()
    kv("written to", str(out.relative_to(REPO)), DIM)

    console.print()
    console.print(Rule(style=GOLD))
    console.print(Align.center(Text(
        f"ticket  ->  {len(sops)} SOPs  ->  {chosen['sop_id']}  ->  {len(plan_steps)} skills",
        style=f"bold {GOLD}")))
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
    console.print(Text(f"   (models loaded in {load_s:.1f}s)", style=DIM))
    console.print()


if __name__ == "__main__":
    main()
