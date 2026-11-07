"""Merge per-family generated SOP/incident files into a single validated corpus.

Reads data_gen/raw/<family>.json (shape: {family, sops:[{ref,title,condition,steps,
equipment}], incidents:[{sop_ref,text}]}), assigns global SOP/incident IDs, resolves
each incident to its gold SOP, drops anything unresolvable, and writes a held-out
train/val/test split.

Deterministic: no RNG; splits come from a stable hash so re-runs are reproducible.
Output keys match the existing schema (`sop_examples`, `incident_examples`) so
src/data/schemas.load_json reads them unchanged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT / "data_gen" / "raw"
OUT_DIR = PROJECT / "src" / "data"

SKILL_VERBS = {"walk_to", "press_button", "wait", "read_sensor", "pick", "place", "notify"}


def _stable_bucket(key: str, mod: int = 100) -> int:
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % mod


def main() -> None:
    files = sorted(RAW_DIR.glob("*.json"))
    if not files:
        raise SystemExit(f"No raw family files in {RAW_DIR}")

    sops: list[dict] = []            # global SOP records
    ref_to_id: dict[tuple[str, str], str] = {}
    incidents_raw: list[dict] = []   # (family_slug, sop_ref, text)
    bad_files: list[str] = []
    fam_count = 0

    sop_n = 0
    for fp in files:
        slug = fp.stem
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            fam_sops = data["sops"]
            fam_inc = data.get("incidents", [])
        except Exception as e:
            bad_files.append(f"{fp.name}: {e}")
            continue
        fam_count += 1
        for s in fam_sops:
            ref = str(s.get("ref", "")).strip()
            if not ref or not s.get("title") or not s.get("steps"):
                continue
            sop_n += 1
            sop_id = f"SOP-{sop_n:04d}"
            key = (slug, ref)
            if key in ref_to_id:
                continue  # duplicate ref within family
            ref_to_id[key] = sop_id
            sops.append(
                {
                    "sop_id": sop_id,
                    "title": str(s["title"]).strip(),
                    "condition": str(s.get("condition", "")).strip(),
                    "steps": [str(x).strip() for x in s["steps"]],
                    "equipment": [str(x).strip() for x in s.get("equipment", [])],
                    "_family": slug,
                }
            )
        for inc in fam_inc:
            incidents_raw.append((slug, str(inc.get("sop_ref", "")).strip(), str(inc.get("text", "")).strip()))

    # Resolve incidents -> gold SOP id
    resolved: list[dict] = []
    unresolved = 0
    for slug, ref, text in incidents_raw:
        if not text:
            continue
        sop_id = ref_to_id.get((slug, ref))
        if not sop_id:
            unresolved += 1
            continue
        resolved.append({"sop_id": sop_id, "text": text})

    # Assign incident IDs + split. Guarantee each SOP has >=1 train incident (coverage),
    # then bucket the rest ~70/15/15.
    by_sop: dict[str, list[dict]] = {}
    for r in resolved:
        by_sop.setdefault(r["sop_id"], []).append(r)

    train, val, test = [], [], []
    inc_n = 0
    for sop_id in sorted(by_sop):
        group = by_sop[sop_id]
        for i, r in enumerate(group):
            inc_n += 1
            rec = {"incident_id": f"INC-{inc_n:05d}", "text": r["text"], "labels": {"sop_id": sop_id}}
            if i == 0:
                train.append(rec)  # coverage: first incident of every SOP trains
            else:
                b = _stable_bucket(rec["incident_id"])
                (train if b < 65 else val if b < 80 else test).append(rec)

    # Skill-executability sanity: fraction of steps whose first word is a known skill verb
    total_steps = sum(len(s["steps"]) for s in sops)
    skill_steps = sum(1 for s in sops for st in s["steps"] if st.split()[:1] and st.split()[0].lower() in SKILL_VERBS)
    skill_frac = skill_steps / max(1, total_steps)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # strip private _family before writing the public corpus
    public_sops = [{k: v for k, v in s.items() if not k.startswith("_")} for s in sops]
    (OUT_DIR / "plant_sops.json").write_text(json.dumps({"sop_examples": public_sops}, indent=2), encoding="utf-8")
    for name, split in [("train", train), ("val", val), ("test", test)]:
        (OUT_DIR / f"plant_incidents_{name}.json").write_text(
            json.dumps({"incident_examples": split}, indent=2), encoding="utf-8"
        )

    print("=== merge summary ===")
    print(f"families read:      {fam_count} (bad files: {len(bad_files)})")
    for b in bad_files:
        print(f"   BAD: {b}")
    print(f"SOPs:               {len(sops)}")
    print(f"incidents resolved: {len(resolved)} (unresolved dropped: {unresolved})")
    print(f"  train/val/test:   {len(train)} / {len(val)} / {len(test)}")
    print(f"SOPs with >=1 incident: {len(by_sop)} / {len(sops)}")
    print(f"step skill-verb coverage: {skill_frac:.1%} ({skill_steps}/{total_steps})")
    print(f"wrote: plant_sops.json, plant_incidents_train/val/test.json -> {OUT_DIR}")


if __name__ == "__main__":
    main()
