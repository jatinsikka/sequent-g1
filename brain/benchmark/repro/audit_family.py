import collections, numpy as np, pandas as pd
from common import load, family_map, RESULTS
sops, split = load()
fam = family_map()
titles_ok = sum(1 for s in sops if s.sop_id in fam)
print("SOPs mapped to a family:", titles_ok, "/", len(sops), "| families:", len(set(fam.values())) - 1)
c = pd.Series(collections.Counter(fam[s.sop_id] for s in sops if s.sop_id != "SOP-1013"))
print("SOPs per family: min %d median %d max %d" % (c.min(), c.median(), c.max()))
for k, v in split.items():
    print(k, "families covered:", len({fam[i.labels['sop_id']] for i in v}))
c.rename("n_sops").to_csv(RESULTS / "sops_per_family.csv", index_label="family")
