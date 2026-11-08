#!/usr/bin/env python
"""
Sequent retrieval evaluation.

Runs labelled incident tickets through the rankers against the full SOP index and reports
Recall@1, Recall@5 and MRR. Every number is computed on this machine, right now.

    python demo/brain_eval.py                 # plant library: 269 held-out tickets x 1,012 SOPs
    python demo/brain_eval.py --informed      # plant library: 92 clear tickets from an informed operator
    python demo/brain_eval.py --misses        # also list every ticket the hybrid got wrong
    python demo/brain_eval.py --disagree      # show where lexical and semantic split
    python demo/brain_eval.py --library course   # the original 100-SOP course library
    python demo/brain_eval.py --hard          # ten plant-floor complaints vs the 100-SOP course library

Libraries:
    plant   (default when brain/benchmark/artifacts/retriever_st exists) 1,012 SOPs, the fine-tuned
            retriever, and the paper's baselines (TF-IDF, BM25, untuned MiniLM). The fine-tuned row is
            one trained model; the paper reports the mean of 5 seeds (R@1 0.577 +/- 0.006).
    course  100 SOPs and 100 tickets with the off-the-shelf MiniLM. Saturated: every ranker scores
            near 1.0, which is why the 1,012-SOP benchmark exists.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "brain"))

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.rule import Rule
from rich.panel import Panel

console = Console()
GOLD, GREEN, RED, BLUE, DIM = "#e6b325", "#78dc78", "#e06c75", "#61afef", "grey58"


BENCH = REPO / "brain" / "benchmark"
FT_DIR = BENCH / "artifacts" / "retriever_st"
_STATE: dict = {}


def load_plant(informed: bool = False):
    with open(BENCH / "src/data/plant_sops.json", encoding="utf-8") as f:
        sops = [s for s in json.load(f)["sop_examples"] if s["sop_id"] != "SOP-1013"]
    if informed:
        tk = json.loads((BENCH / "repro/data/informed_tickets.json").read_text())
        incs = [{"incident_id": f"INF-{i + 1:02d}", "text": t["text"], "labels": {"sop_id": t["sop_id"]}}
                for i, t in enumerate(tk)]
    else:
        with open(BENCH / "src/data/plant_incidents_test.json", encoding="utf-8") as f:
            incs = json.load(f)["incident_examples"]
    return sops, incs


def _synth(s):
    return f"{s['title']}. {s['condition']}. Steps: " + " ; ".join(s["steps"])


def plant_rankers(sops):
    """The paper's rankers over the 1,012-SOP index, each returning the full ranking of SOP ids."""
    import re as _re
    import numpy as np
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    ids = [s["sop_id"] for s in sops]
    texts = [_synth(s) for s in sops]
    vec = TfidfVectorizer(ngram_range=(1, 2)); S = vec.fit_transform(texts)
    tok = lambda t: _re.findall(r"[a-z0-9]+", t.lower())
    bm = BM25Okapi([tok(t) for t in texts])
    zs = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"); Ezs = zs.encode(texts, normalize_embeddings=True)
    ft = SentenceTransformer(str(FT_DIR)); Eft = ft.encode(texts, normalize_embeddings=True)
    top = lambda sc: [ids[j] for j in np.argsort(-sc)]  # full ranking, so MRR matches the paper

    def tfidf(q): return top((vec.transform([q]) @ S.T).toarray()[0])
    def bm25(q): return top(bm.get_scores(tok(q)))
    def zero(q): return top(Ezs @ zs.encode([q], normalize_embeddings=True)[0])
    def tuned(q): return top(Eft @ ft.encode([q], normalize_embeddings=True)[0])

    def hybrid(q):  # brain_bridge's runtime rule: interleave semantic and lexical, semantic first
        merged, seen = [], set()
        for x in [y for ab in zip(tuned(q), tfidf(q)) for y in ab]:
            if x not in seen:
                seen.add(x); merged.append(x)
        return merged
    return [("lexical  (TF-IDF)", tfidf), ("BM25", bm25), ("MiniLM, untuned", zero),
            ("semantic (MiniLM)", tuned), ("hybrid   (runtime)", hybrid)]


def load():
    with open(REPO / "brain/src/data/sop_examples.json", encoding="utf-8") as f:
        sops = json.load(f)["sop_examples"]
    with open(REPO / "brain/src/data/incident_examples.json", encoding="utf-8") as f:
        incs = json.load(f)["incident_examples"]
    return sops, incs


def metrics(ranked_ids: list, golds: list) -> dict:
    """Recall@1, Recall@5, MRR over a list of ranked id-lists."""
    r1 = r5 = 0
    rr = 0.0
    for ids, g in zip(ranked_ids, golds):
        if ids and ids[0] == g:
            r1 += 1
        if g in ids[:5]:
            r5 += 1
        if g in ids:
            rr += 1.0 / (ids.index(g) + 1)
    n = len(golds)
    return {"R@1": r1 / n, "R@5": r5 / n, "MRR": rr / n, "n": n}


# Ten tickets (not in the dataset) for the 100-SOP library. `accept` lists every SOP that
# counts as correct, since some tickets are ambiguous.
HARD = [
    ("there is a burning plastic stink coming off the unit",        {"SOP-079"}, ""),
    ("it is making a horrible grinding racket I have never heard before",
                                                                    {"SOP-078"}, ""),
    ("I saw little blue flashes coming out the back of the cabinet", {"SOP-080"}, ""),
    ("the bin is overflowing with offcuts",                          {"SOP-095"}, ""),
    ("the oil bottle looks nearly empty",                            {"SOP-096"}, ""),
    ("nothing lights up at all when I walk past it",                 {"SOP-068"}, ""),
    ("the stop switch will not budge when I push it",                {"SOP-081"}, ""),
    ("someone left tools all over the bench after servicing",        {"SOP-093"}, ""),
    ("been sat off all weekend, first thing monday morning",
     {"SOP-086", "SOP-045"}, "ambiguous: cold start or start-of-shift check"),
    ("the whole thing is rattling itself apart",
     {"SOP-031", "SOP-005", "SOP-053", "SOP-023"}, "ambiguous: four vibration SOPs fit"),
]


def hard_mode() -> None:
    """TF-IDF vs MiniLM on the ten hard tickets (100-SOP library)."""
    sops, _ = load()
    from brain_bridge import _retrieve_sop_tfidf
    from src.retrieval.infer_retrieve import retrieve_topk

    console.print()
    console.print(Rule(Text("  Hard set: 10 tickets, 100-SOP library  ",
                            style=f"bold black on {GOLD}"), style=GOLD))
    console.print()

    with console.status("[grey58]loading the encoder...[/]", spinner="dots"):
        retrieve_topk("warmup", k=1)

    by_id = {s["sop_id"]: s for s in sops}
    lex_hits = sem_hits = 0
    notes = []
    for text, accept, note in HARD:
        lex = [s["sop_id"] for s, _ in _retrieve_sop_tfidf(text, sops, k=5)]
        sem = [h["sop_id"] for h in retrieve_topk(text, k=5)]
        lok, sok = lex[0] in accept, sem[0] in accept
        lex_hits += lok
        sem_hits += sok
        sem_rank = next((i for i, x in enumerate(sem, 1) if x in accept), None)

        console.print(Text(f'   "{text}"', style="bold white"))
        for label, sid, ok in (("lexical ", lex[0], lok), ("semantic", sem[0], sok)):
            title = by_id[sid]["title"] if sid in by_id else sid
            console.print(
                Text(f"      {label}  ", style=DIM),
                Text(f"{sid}  {title:<34.34}", style=GREEN if ok else RED),
                Text("hit" if ok else "miss", style=f"bold {GREEN}" if ok else RED))
        if note and sok:
            console.print(Text(f"      ({note}; counted as a hit)", style=DIM))
        console.print()

        if not sok:
            if note:
                notes.append((text, note))
            elif sem_rank:
                notes.append((text, f"semantic rank {sem_rank}"))
            else:
                notes.append((text, "not in either top 5"))

    n = len(HARD)
    tt = Table(box=None, padding=(0, 3))
    tt.add_column("ranker", width=20)
    tt.add_column("rank-1 hits", width=14)
    tt.add_row(Text("lexical  (TF-IDF)", style="white"), f"{lex_hits}/{n}")
    tt.add_row(Text("semantic (MiniLM)", style=f"bold {GREEN}" if sem_hits > lex_hits else "white"),
               f"{sem_hits}/{n}")
    console.print(tt)

    if notes:
        console.print()
        console.print(Text("   Misses:", style=f"bold {GOLD}"))
        for text, note in notes:
            console.print(Text(f'     "{text[:46]}"', style=DIM))
            console.print(Text(f"        {note}", style=RED))
    console.print()
    console.print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--misses", action="store_true", help="list the tickets the hybrid gets wrong")
    ap.add_argument("--disagree", action="store_true", help="list where the two rankers split")
    ap.add_argument("--hard", action="store_true",
                    help="ten plant-floor complaints against the 100-SOP course library")
    ap.add_argument("--informed", action="store_true", help="plant library: the 92 informed-operator tickets")
    ap.add_argument("--library", choices=["plant", "course"], default=None,
                    help="plant: 1,012 SOPs + fine-tuned retriever (default if trained); course: 100 SOPs")
    a = ap.parse_args()

    if a.hard:
        hard_mode()
        return

    lib = a.library or ("plant" if FT_DIR.is_dir() else "course")
    if lib == "plant" and not FT_DIR.is_dir():
        console.print("[red]The fine-tuned retriever is not trained yet.[/] From brain/benchmark run\n"
                      "    python -m src.retrieval.train_eval_sentence_encoder\nor use --library course.")
        sys.exit(1)
    sops, incs = load_plant(a.informed) if lib == "plant" else load()
    golds = [i["labels"]["sop_id"] for i in incs]

    console.print()
    console.print(Rule(Text("  Retrieval evaluation  ", style=f"bold black on {GOLD}"),
                       style=GOLD))
    console.print()
    console.print(Text(f"   {len(incs)} labelled incident tickets  x  {len(sops)}-SOP index"
                       f"   =  {len(incs) * len(sops):,} scored pairs per ranker", style=DIM))
    console.print()

    if lib == "plant":
        with console.status("[grey58]loading the encoders and indexing 1,012 SOPs...[/]", spinner="dots"):
            rankers = plant_rankers(sops)
    else:
        from brain_bridge import _retrieve_sop_tfidf, _interleave
        from src.retrieval.infer_retrieve import retrieve_topk
        by_id = {s["sop_id"]: s for s in sops}
        with console.status("[grey58]loading the encoder...[/]", spinner="dots"):
            retrieve_topk("warmup", k=1)
        lex = lambda q: [s["sop_id"] for s, _ in _retrieve_sop_tfidf(q, sops, k=10)]
        sem = lambda q: [h["sop_id"] for h in retrieve_topk(q, k=10)]
        rankers = [("lexical  (TF-IDF)", lex), ("semantic (MiniLM)", sem),
                   ("hybrid   (runtime)", lambda q: [s["sop_id"] for s, _ in _interleave(
                       [(by_id[i], 0.0) for i in sem(q) if i in by_id], [(by_id[i], 0.0) for i in lex(q)], 10)])]

    runs = {}
    for name, fn in rankers:
        with console.status(f"[grey58]running {name.strip()} over {len(incs)} tickets...[/]",
                            spinner="dots"):
            t0 = time.perf_counter()
            ranked = [fn(i["text"]) for i in incs]
            elapsed = time.perf_counter() - t0
        m = metrics(ranked, golds)
        m["ms"] = elapsed / len(incs) * 1000
        runs[name] = (m, ranked)
        console.print(Text(f"   {name}  done in {elapsed:.1f}s", style=DIM))

    console.print()
    t = Table(box=None, padding=(0, 3), title="results",
              title_style=f"bold {GOLD}", title_justify="left")
    t.add_column("ranker", width=20)
    t.add_column("R@1", justify="right", width=7)
    t.add_column("R@5", justify="right", width=7)
    t.add_column("MRR", justify="right", width=7)
    t.add_column("per ticket", justify="right", width=11)
    best = max(runs, key=lambda k: runs[k][0]["R@1"])
    for name, (m, _) in runs.items():
        style = f"bold {GREEN}" if name == best else "white"
        t.add_row(Text(name, style=style), f"{m['R@1']:.3f}", f"{m['R@5']:.3f}",
                  f"{m['MRR']:.3f}", f"{m['ms']:.0f} ms", style=style)
    console.print(t)
    console.print()

    lex_ranked = runs["lexical  (TF-IDF)"][1]
    sem_ranked = runs["semantic (MiniLM)"][1]
    lex_hit = [r and r[0] == g for r, g in zip(lex_ranked, golds)]
    sem_hit = [r and r[0] == g for r, g in zip(sem_ranked, golds)]
    only_sem = [i for i, (l, s) in enumerate(zip(lex_hit, sem_hit)) if s and not l]
    only_lex = [i for i, (l, s) in enumerate(zip(lex_hit, sem_hit)) if l and not s]

    console.print(Panel(
        Text.assemble(
            ("Rank-1 disagreements:\n", "white"),
            (f"   {len(only_sem):>3}  only MiniLM correct\n", BLUE),
            (f"   {len(only_lex):>3}  only TF-IDF correct", GOLD),
        ), border_style=DIM, padding=(1, 3)))

    # The 100-SOP set is saturated; say so instead of reporting ~1.0 as a result.
    top = max(m["R@1"] for m, _ in runs.values())
    if top >= 0.95:
        console.print()
        console.print(Panel(
            Text.assemble(
                ("100 SOPs, one paraphrased ticket each: every ranker scores ~1.0.\n", "white"),
                ("On the 1,012-SOP benchmark: fine-tuned 0.577, TF-IDF 0.301.\n", "white"),
                ("    python demo/brain_eval.py --library plant", f"bold {GOLD}"),
            ), border_style=RED, padding=(1, 3),
            title=f"[bold {RED}]saturated[/]"))

    if a.disagree:
        for label, idxs, colour in (("only semantic gets it", only_sem, BLUE),
                                    ("only lexical gets it", only_lex, GOLD)):
            if not idxs:
                continue
            console.print()
            t = Table(box=None, padding=(0, 2), title=label, title_style=f"bold {colour}",
                      title_justify="left")
            t.add_column("ticket", style=DIM, width=8)
            t.add_column("gold", width=8)
            t.add_column("what the operator wrote", overflow="ellipsis", max_width=70)
            for i in idxs[:12]:
                t.add_row(incs[i]["incident_id"], golds[i], incs[i]["text"])
            console.print(t)

    if a.misses:
        hyb_ranked = runs["hybrid   (runtime)"][1]
        miss = [(i, r) for i, (r, g) in enumerate(zip(hyb_ranked, golds)) if not r or r[0] != g]
        console.print()
        t = Table(box=None, padding=(0, 2),
                  title=f"every ticket the hybrid misses at rank 1  ({len(miss)}/{len(incs)})",
                  title_style=f"bold {RED}", title_justify="left")
        t.add_column("ticket", style=DIM, width=8)
        t.add_column("gold", width=8)
        t.add_column("it returned", width=9)
        t.add_column("gold rank", justify="right", width=9)
        t.add_column("what the operator wrote", overflow="ellipsis", max_width=54)
        for i, r in miss:
            g = golds[i]
            rank = (r.index(g) + 1) if g in r else None
            t.add_row(incs[i]["incident_id"], g, r[0] if r else "—",
                      Text(str(rank) if rank else ">10",
                           style=GOLD if rank and rank <= 5 else RED),
                      incs[i]["text"])
        console.print(t)


    console.print()
    if lib == "plant":
        console.print(Text("   One model. Paper: mean of 5 seeds, R@1 0.577 +/- 0.006.", style=DIM))
        if a.informed:
            console.print(Text("   Informed tickets: written by the author from SOP titles.", style=DIM))
    else:
        console.print(Text("   100-SOP course library, off-the-shelf MiniLM.", style=DIM))
    console.print()


if __name__ == "__main__":
    main()
