# Sequent: paper outline

## Heilmeier questions

1. **What is the objective?** A humanoid robot that reads a maintenance ticket written by a plant operator, finds the procedure that applies, and carries it out, while reporting truthfully which steps worked.
2. **If it works, what does it enable?** Robots that act from the procedures a plant already has, instead of from hand-written task scripts, with every action checkable afterwards.
3. **What fails today?** Language-to-robot systems usually let a large model plan freely, so the plan is not tied to an approved procedure, and success is often judged by the policy's own reward or the model's own report. In retrieval, off-the-shelf models struggle when operators describe symptoms instead of naming the procedure.
4. **Key insight?** Split the problem. Use a small retriever trained on operator language to pick an approved SOP (it can never invent one), map its steps to skills by rule, and judge every step by measuring the simulator state, not by asking the policy. Use controllers where the task is geometric and RL only where contact makes modeling hard.
5. **Limitations and assumptions?** Simulation only (MuJoCo). The SOP corpus and tickets are synthetic and come from one generator. SOPs are written at skill granularity, so planning is a lookup. The retriever always returns an SOP and cannot say "none of these." The lever skill succeeds about half the time.

## Outline

**I. Introduction**
- P1: Plants run on written procedures; operators describe problems in their own words.
- P2: Gap: LLM planners are not grounded in approved procedures, and robot learning often trusts its own success signal.
- P3: Insight: retrieve, don't generate; verify from physics; controllers vs RL by task type. Fig. 1.
- Related work, in four groups: LLM task planning; dense retrieval; success detection and asking for help; humanoid loco-manipulation and reward hacking.
- Contributions (4), each with a number.

**II. Problem statement.** Corpus, ticket, retriever, plan, skills, verifier predicates, verdict. Problem 1 (retrieval) and Problem 2 (verified execution).

**III. Approach.** Retriever and its loss; rule-based planner; skills (controller vs RL); verifier. Named benchmarks.

**IV. Experiments.**
- A. Retrieval: main table; held-out SOPs and families; ticket clarity (Fig. 2); error analysis; selective acting.
- B. Skills: per-skill results; the action-envelope finding (Fig. 3).
- C. Full system: one-take chain and the finale.
- D. Reward hacks (Table).
- Limitations.

**V. Conclusion and future work.**

**Appendix.** Hyperparameters, data splits, seeds, informed-ticket protocol, checklist.
