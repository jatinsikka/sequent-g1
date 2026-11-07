"""Experiment 1: audit the 1,012-SOP benchmark's stated properties."""
import json, re, collections
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from common import load, RESULTS, ROOT, synth

sops, split = load()
ids = [s.sop_id for s in sops]
print("SOPs in plant_sops.json:", len(sops), "| unique ids:", len(set(ids)))
print("split sizes:", {k: len(v) for k, v in split.items()}, "total", sum(map(len, split.values())))
raw = json.load(open(next((ROOT / "src/data").glob("plant_sops.json"))))
print("top-level keys:", list(raw)[:5])
s0 = sops[0]; print("example SOP:", s0.sop_id, "|", s0.title, "| equipment:", s0.equipment)
print("example incident:", split["test"][0].model_dump())

# family = equipment[0] if present, else id prefix
fam = {s.sop_id: (s.equipment[0] if s.equipment else re.sub(r"-?\d+$", "", s.sop_id)) for s in sops}
print("families (equipment[0]):", len(set(fam.values())))
gold = {k: [i.labels.get("sop_id") for i in v] for k, v in split.items()}
missing = {k: sum(g not in fam for g in v) for k, v in gold.items()}
print("gold ids missing from corpus:", missing)
tr = set(gold["train"]); te = gold["test"]
print("test golds seen in train:", sum(g in tr for g in te), "/", len(te), "| unique test golds:", len(set(te)))
print("SOPs never used as a gold anywhere:", len(set(ids) - set(sum(gold.values(), []))))
print("incidents per SOP (all splits):", pd.Series(collections.Counter(sum(gold.values(), []))).describe().round(2).to_dict())

# near-duplicate train<->test incidents
vec = TfidfVectorizer(ngram_range=(1, 2)).fit([i.text for i in split["train"]] + [i.text for i in split["test"]])
A = vec.transform([i.text for i in split["test"]]); B = vec.transform([i.text for i in split["train"]])
sim = (A @ B.T).toarray(); mx = sim.max(1)
print("test->train max cosine: mean %.3f | >0.9: %d | >0.7: %d" % (mx.mean(), (mx > .9).sum(), (mx > .7).sum()))
nn = sim.argmax(1)
same = np.mean([gold["train"][j] == g for j, g in zip(nn, te)])
print("nearest train incident has the same gold SOP: %.3f (a 1-NN over train incidents would score this R@1)" % same)

# title-word overlap
stop = set("the a an of and or to in on for with is are be at by from as".split())
tok = lambda t: {w for w in re.findall(r"[a-z]+", t.lower()) if w not in stop and len(w) > 2}
title = {s.sop_id: tok(s.title) for s in sops}
ov = [len(tok(i.text) & title[g]) / max(1, len(title[g])) for i, g in zip(split["test"], te)]
print("test incident ∩ gold-title words: mean frac %.3f | incidents with zero overlap: %.3f" % (np.mean(ov), np.mean([o == 0 for o in ov])))
per_fam = pd.Series(collections.Counter(fam.values()))
print("SOPs per family:", per_fam.describe().round(1).to_dict())
pd.DataFrame([dict(metric=k, value=v) for k, v in {
    "n_sops": len(sops), "n_families": len(per_fam), "n_train": len(split["train"]), "n_val": len(split["val"]),
    "n_test": len(split["test"]), "test_golds_seen_in_train": sum(g in tr for g in te),
    "test_train_maxcos_mean": float(mx.mean()), "test_train_maxcos_gt0.9": int((mx > .9).sum()),
    "nn_train_incident_same_gold": float(same), "title_overlap_mean": float(np.mean(ov)),
    "title_overlap_zero_frac": float(np.mean([o == 0 for o in ov]))}.items()]).to_csv(RESULTS / "data_audit.csv", index=False)
