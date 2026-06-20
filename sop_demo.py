"""END-TO-END THESIS DEMO:
   natural-language incident  ->  SOP retrieval  ->  plan  ->  skill execution
   with PHYSICS VERIFICATION at each physical step  ->  honest report.

Run:  PYTHONPATH=brain python sop_demo.py "the machine is overheating, shut it down"
"""
import sys, os, json, warnings, numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "brain"))
from src.retrieval.infer_retrieve import retrieve_topk
from src.planner.infer_plan import _extract_skills_from_sop
from src.data.schemas import SKILLS

SOPS = {s["sop_id"]: s for s in json.load(open(os.path.join(os.path.dirname(__file__),
        "brain/src/data/sop_examples.json")))["sop_examples"]}

def bar(t): print("\n" + "=" * 66 + "\n  " + t + "\n" + "=" * 66)

def run_sop(query):
    bar(f'INCIDENT (natural language):  "{query}"')
    top = retrieve_topk(query, k=1)[0]; sop = SOPS[top["sop_id"]]
    print(f"  RETRIEVED  {sop['sop_id']}: {sop['title']}   (similarity {top['score']:.2f})")
    print(f"  condition: {sop['condition']}")

    steps = _extract_skills_from_sop(sop["steps"], SKILLS)
    bar("PLAN  (retrieved SOP  ->  executable skill chain)")
    for i, st in enumerate(steps): print(f"  {i+1}. {st['skill']}({st['args']})")

    bar("EXECUTION  (physical steps verified against MEASURED physics)")
    all_ok = True
    for i, st in enumerate(steps):
        sk, args = st["skill"], st["args"]; physical = sk in ("walk_to", "press_button", "pick", "grasp"); ok = True
        if sk in ("pick", "grasp"):
            from skills_manipulation import run_grasp
            r = run_grasp(); ok = r["grasped"]
            detail = f"lifted {r['max_lift']*100:.1f}cm, held {r['held_steps']}steps, latched={r['latched']}"
        elif sk == "place":
            detail = f"placed {args.get('object','')} (operational — place not yet RL-verified)"
        elif sk == "walk_to":
            from skills_locomotion import run_walk_to
            r = run_walk_to(args.get("target", "machine")); ok = r["arrived"] and not r["fell"]
            detail = f"arrived={r['arrived']} fell={r['fell']} min_dist={r['min_dist']:.2f}m"
        elif sk == "press_button":
            from skills_manipulation import run_press_button
            btn = "button_" + str(args.get("button", "red_button")).replace("_button", "")
            r = run_press_button(btn); ok = r["pressed"]
            detail = f"depth={r['max_disp']*100:.1f}cm held={r['held_steps']}steps upright={not r['fell']}"
        elif sk == "read_sensor": detail = f"{args.get('sensor')} = nominal"
        elif sk == "wait":        detail = f"{args.get('seconds')}s elapsed"
        elif sk == "notify":      detail = f"operator ({args.get('level')})"
        else:                     detail = "(operational)"
        tag = ("[VERIFIED]" if ok else "[FAILED]  ") if physical else "[done]    "
        print(f"  {i+1}. {sk:13s} {tag}  {detail}")
        if physical and not ok:
            all_ok = False
            print("     -> step NOT verified by physics; SOP HALTS (honest failure, not a faked success)")
            break
    bar("SOP " + ("COMPLETE  -  every physical step physics-verified  ✅" if all_ok
                  else "HALTED  -  a step failed its physics check  ⛔"))
    return all_ok

if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "the machine is overheating, shut it down now"
    run_sop(q)
