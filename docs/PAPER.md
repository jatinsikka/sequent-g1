# Sequent: SOP-Driven, Physics-Verified Loco-Manipulation on a Simulated Humanoid

**Jatin Sikka**, The University of Texas at Austin
Project page: <https://jatinsikka.github.io/sequent-g1/> · Code: <https://github.com/jatinsikka/sequent-g1>

## Abstract

Factories run on written Standard Operating Procedures (SOPs), but operators report problems in their own words. We built Sequent, a system in which a simulated Unitree G1 humanoid reads such a report, retrieves the SOP that applies out of 1,012, maps its steps to robot skills, and executes them in one continuous MuJoCo simulation. A verifier checks every step against the simulator state instead of trusting the policy, so a step that fails is reported as failed. The retriever is a small transformer fine-tuned with a contrastive loss. On vague operator tickets it ranks the correct SOP first 57.7% of the time (5 seeds), against 35.7% for BM25 and 40.9% for the same model without fine-tuning, and it keeps most of this gain on SOPs and machine families held out from training. For the body we use controllers where the problem is geometric (walking, grasping) and reinforcement learning where contact makes modeling hard (pressing a button, pulling a lever). We report what did not work, including the lever, and a set of reward hacks that were each found by watching rollouts rather than from training metrics.

## 1. Introduction

![Sequent pipeline](figures/fig_pipeline.png)

*Figure 1: **Only the retriever and three of the skills are learned.** A ticket is matched to one of 1,012 SOPs by the fine-tuned retriever. The planner maps each written step to a skill by rule. The verifier runs each skill inside a precondition/postcondition check measured from simulator state and reports a verdict. Blue: learned. Grey: rules and controllers.*

A plant floor is full of written procedures: what to do when a press jams, when coolant pressure drops, before anyone opens a cabinet. When something goes wrong, an operator does not name the procedure. They write something like "the dispenser coasted to a stop and the thermal light is red." A robot that could act on that report, using the procedures the plant already trusts, would be useful well before it could do anything open-ended.

Recent language-driven robot systems use large language models to turn instructions into plans [1, 2, 3]. Despite their flexibility, a key open problem is grounding: a freely generated plan is not tied to an approved procedure, and success is often judged by the model's own report or by the policy's reward. In a plant, both are risky. A plan must come from an approved SOP, and "done" must mean the physical state actually changed.

Our approach rests on three choices. First, retrieve rather than generate: a retriever trained on operator language picks an existing SOP, so the robot can only ever run an approved procedure. Second, verify from physics: each skill runs inside a precondition and postcondition check computed from the simulator state, never from the policy's output. Third, pick the tool per skill: walking and grasping are geometric and get controllers, while pressing and pulling are dominated by contact and get RL.

**Related work.** Our work relates to four lines of research. *Language-conditioned task planning* uses LLMs to produce skill sequences, scored by affordances [1], written as code [2], or revised with feedback [3]. These plan freely; we restrict the robot to retrieved, pre-approved procedures, which trades flexibility for auditability. *Dense retrieval* trains bi-encoders on question–passage pairs [4, 5] and underlies retrieval-augmented generation [6]; lexical methods such as BM25 [7] remain strong baselines. We apply a standard bi-encoder recipe to a new setting, maintenance tickets against SOPs with many look-alike procedures, and measure where it fails. *Knowing when to ask* has been studied for LLM planners using calibrated uncertainty [8]; our retriever does not yet abstain, and we measure what a simple confidence cutoff would give. *Humanoid loco-manipulation and reward design*: we build on the AMO whole-body controller [9] for walking, and contrast with systems that learn manipulation from demonstrations [10]. The reward hacks we document are instances of specification gaming [11, 12].

**Contributions.** In light of prior work, our contributions are:

1. A system that goes from an operator's ticket to a verified, multi-skill humanoid execution in one continuous simulation: walk, pick, carry, place, press, and a partial lever pull, with every step checked against simulator state.
2. A retrieval benchmark of 1,012 SOPs across 46 equipment families with 2,338 operator-style incidents, and a reproducible evaluation: the fine-tuned retriever reaches R@1 0.577 ± 0.006 versus 0.357 for BM25 and 0.409 untuned, with held-out-SOP, held-out-family, and ticket-clarity tests. Code, data, and per-run results are released.
3. Evidence for the controller/RL split: RL grasping on the gripper-equipped humanoid reached 0% verified success, while an IK controller with a force-gated latch held 11 of 12 lifts; PPO with a curriculum presses reliably (8 of 8 deterministic starts).
4. A catalog of six reward hacks, each caught by watching rollouts, with the class of fix that worked.

**Organization.** Section 2 states the problem. Section 3 describes the approach and the baselines. Section 4 reports retrieval, skill, and full-system results. Section 5 concludes.

## 2. Problem Statement

**SOP corpus.** A plant has a corpus $D = \{d_1, \dots, d_N\}$ of $N = 1{,}012$ SOPs. Each SOP $d$ has a title, a triggering condition, and an ordered list of steps $(\sigma_1, \dots, \sigma_K)$. Each step names a skill from a fixed set $A = \{$`walk_to`, `pick`, `place`, `press_button`, `read_sensor`, `wait`, `notify`$\}$ and an entity, for example `press_button estop_button`.

**Tickets.** An operator writes a ticket $q$, free text such as "line 2 dispenser coasted to a stop, thermal indicator in the red." Each ticket has one correct SOP $d^*(q)$.

**Retriever.** An encoder $E_\theta$ maps text to a unit vector in $\mathbb{R}^{384}$, and the score of SOP $d$ for ticket $q$ is $s_\theta(q, d) = E_\theta(q)^\top E_\theta(d)$. The robot acts on $\hat d(q) = \arg\max_{d \in D} s_\theta(q, d)$.

**Execution.** A plan $\pi(\hat d)$ is the sequence of skills in $\hat d$. The robot, with simulator state $x_t$, executes skill $k$ with a controller or a learned policy. Each skill has a precondition $\mathrm{pre}_k(x_t) \in \{0, 1\}$ checked before it starts and a postcondition $\mathrm{post}_k(x_t) \in \{0, 1\}$ checked after it ends, both computed from simulator state (for example, "button displaced past threshold and held," "box above the table for a sustained window," "base upright"). The verdict is *resolved* if every postcondition holds and *partially resolved* or *failed* otherwise.

> **Problem 1 (retrieval).** Given training pairs $\{(q_j, d^*(q_j))\}$, find $\theta$ that maximizes $\Pr[\hat d(q) = d^*(q)]$ on held-out tickets, measured as Recall@1. We also report Recall@5 and mean reciprocal rank (MRR).
>
> **Problem 2 (verified execution).** Given $\hat d$, execute $\pi(\hat d)$ on the humanoid so that each $\mathrm{post}_k$ holds, and report a verdict computed only from the $\mathrm{post}_k$ values, never from a policy's own success signal.

**Why this is hard.** Tickets describe symptoms, not procedures, and each machine has around 22 SOPs that differ in small ways (pressure below setpoint versus sudden collapse versus above ceiling). On the execution side, the skills chain on one body in one simulation, so errors carry from step to step: a robot that arrives a few centimeters off makes the next reach harder.

## 3. Approach

### 3.1 Retriever

We fine-tune `all-MiniLM-L6-v2` [5] (22.7M parameters, 6 layers) on the 1,859 training pairs. Each SOP is represented as its title, condition, and steps in one string. Training uses the multiple-negatives ranking loss: for a batch of $B$ pairs, each ticket should score its own SOP above the other $B - 1$ SOPs in the batch,

$$\mathcal{L}(\theta) = -\frac{1}{B} \sum_{j=1}^{B} \log \frac{\exp(\lambda\, s_\theta(q_j, d_j))}{\sum_{l=1}^{B} \exp(\lambda\, s_\theta(q_j, d_l))},$$

with $B = 32$ and $\lambda = 20$, for 3 epochs (about 2 minutes on a laptop). Hyperparameters are in the appendix.

**Baselines.** All are evaluated on the same index and test set.

- **TF-IDF**: unigram and bigram TF-IDF with cosine similarity.
- **BM25** [7]: Okapi BM25 over lowercase word tokens.
- **MiniLM-0**: the same encoder without fine-tuning. This isolates what training adds.
- **bge-small-0**: `bge-small-en-v1.5`, a stronger off-the-shelf encoder of similar size, without fine-tuning.
- **MiniLM-FT + CE**: our model with its top 10 reranked by the off-the-shelf cross-encoder `ms-marco-MiniLM-L-6-v2`.

### 3.2 Planner

The SOPs are written at skill granularity, so the planner is a deterministic mapping: one skill per written step, in order, with a `walk_to` inserted when the required location changes. Step arguments are mapped onto the simulator's objects (four colored buttons and a fixed set of tools). There is no learning in this step.

### 3.3 Skills

- **Walk**: the frozen AMO whole-body policy [9] with a closed-loop yaw servo. AMO cannot turn in place and drifts about 18° in yaw while walking, so the servo re-steers toward the target from the measured base pose.
- **Pick and place**: damped-least-squares IK to the box, then a latch that engages only when both gripper pads measure contact force, so a grasp cannot be declared across a gap.
- **Press**: PPO with a curriculum that moves the arm's start pose from touching the button to the true rest pose as the success rate rises.
- **Lever**: behavior cloning from scripted IK demonstrations, one DAgger pass, then PPO.

### 3.4 Verifier

The verifier wraps every skill with its precondition and postcondition (Section 2). It is independent of how a skill is implemented, so the same checks apply to controllers and learned policies. Before the final system existed, it already separated two grasp checkpoints with similar training curves: one with a claimed 30% success was 0% under verification (it grasped and never lifted), and one with a claimed 60% was 55%.

## 4. Experiments

**Metrics.** For retrieval: Recall@1 (the correct SOP is ranked first), Recall@5, and MRR over the full 1,012-SOP index. For skills and the full system: success as decided by the verifier.

### 4.1 Retrieval

**Setup.** The 269 held-out test tickets are ranked against all 1,012 SOPs. Fine-tuned rows are the mean and standard deviation over 5 seeds.

| Method | R@1 | R@5 | MRR |
|---|---|---|---|
| TF-IDF | 0.301 | 0.535 | 0.408 |
| BM25 | 0.357 | 0.569 | 0.461 |
| MiniLM-0 | 0.409 | 0.677 | 0.529 |
| bge-small-0 | 0.465 | 0.717 | 0.578 |
| **MiniLM-FT (ours)** | **0.577 ± 0.006** | **0.856 ± 0.006** | **0.695 ± 0.005** |
| MiniLM-FT + CE | 0.452 ± 0.006 | 0.843 ± 0.003 | 0.611 ± 0.004 |

*Table 1: **Fine-tuning adds 17 points of R@1 over the same encoder untuned.** The off-the-shelf reranker lowers R@1, so it is not used.*

**Generalization.** Every test SOP also has training tickets, so the test above is closed-set. We add two harder splits. In the first, 20% of SOPs are held out entirely (459 test tickets). In the second, five whole equipment families are held out (257 tickets). The fine-tuned model reaches R@1 0.656 and 0.607, against 0.468 and 0.420 for MiniLM-0 and 0.447 and 0.311 for BM25. Each split has its own test set, so only within-split comparisons are meaningful. The model is learning how operators describe problems, not memorizing the training SOPs.

![R@1 by ticket clarity](figures/fig_retrieval.png)

*Figure 2: **The fine-tuned model's advantage is largest on vague tickets.** R@1 on the vague test tickets, on the same tickets with the machine named up front, and on 92 clear tickets from an operator who knows the system. Fine-tuned bars: mean ± std over 3 seeds. On clear tickets every method does well.*

**Ticket clarity.** Real operators vary. We test the same index with three kinds of ticket (Fig. 2): the vague test tickets (R@1 0.577), the same tickets with the machine named (0.632), and 92 clear tickets written by the author from each sampled SOP's title and condition, two per family, without looking at model outputs (0.920, with the correct SOP in the top 5 every time). BM25 scores 0.357, 0.398, and 0.837 on the same three sets. Clear tickets are easy for every method. The value of training is on the vague ones.

**Where it fails.** About 70% of misses on the vague test pick an SOP for a different machine. Naming the machine raises the share of top-1 picks on the correct machine from 0.71 to 0.85, but R@1 only to 0.63, so most remaining errors are same-machine look-alikes. In the informed-ticket misses of one seed, 4 of 7 picked another machine's SOP for the same symptom even though the ticket named the machine. A miss is costly: only 1–3% of wrong picks share the correct SOP's skill sequence, so a miss means executing a different procedure.

**Acting only when confident.** The retriever always acts on its top SOP. We measured what a confidence cutoff would give, using the gap between the top two scores and choosing the cutoff on the validation split. Acting only above the cutoff covers 41% ± 3% of vague test tickets at 86% ± 3% precision, against 58% precision when acting on all of them. The rest would be escalated to a person. This is measured but not built into the system.

### 4.2 Skills

| Skill | Method | Result | How measured |
|---|---|---|---|
| Pick | IK + force-gated latch | 11 of 12 held lifts | fixed-base validation |
| Pick (in walk→pick run) | same | 11.9 cm held lift | one continuous run |
| Press | PPO + curriculum | 8 of 8, 29–40 mm, no re-pressing | deterministic, near and far starts |
| Lever | BC → DAgger → PPO | 1 of 2 completed throws | deterministic episodes |
| RL grasp, gripper humanoid | PPO, several reward variants | 0% verified | deterministic evaluation |

*Table 2: **Controllers where the task is geometric, RL where contact dominates.** Lever and grasp results are small-sample or negative and are reported as such.*

**Why grasping became a controller.** RL grasping on the gripper-equipped humanoid reached 0% verified success, although the logged "lift" rate was 30–60%. The videos showed the box being squeezed out of a hard pinch and popping upward; the height check fired while it was in the air. A stochastic policy's grip force oscillates, and a stable two-sided pinch is a narrow regime in the contact solver. The IK controller with a force-gated latch held 11 of 12 lifts on its first day. RL did work on an earlier fixed-base rig with a different hand (55% verified), so the conclusion is about this embodiment and contact setup, not about RL grasping in general.

![Action envelope](figures/fig_envelope.png)

*Figure 3: **The policy could not reach the button from rest.** Required rest-to-contact joint offsets versus the action range. The original ±0.5 rad range was smaller than every offset the tasks require, so far-start attempts saturated silently at the clamp. Widening it to 0.8 rad (press) and 1.2 rad (lever) unblocked training.*

**Check the action space before the reward.** A day of failed far-start press training traced to one number: the policy's action range was 0.5 rad per joint around the neutral pose, while reaching the button from rest needs 0.511 rad at the shoulder and 0.672 rad at the elbow, and the lever needs up to 1.1 rad (Fig. 3). Saturated actions look like a timid policy in every training curve.

**The lever is partial.** The lever needed a latching mechanism so partial progress persisted, quieter exploration (initial log-std −1.5), and behavior cloning to reach any reward at all. The best policy completed the throw in 1 of 2 deterministic episodes, with jerky motion. The cause is identified: a per-episode reach offset that changes what an action does is not in the policy's observation, so identical observations call for different actions. The fix (add it to the observation) is designed but not trained.

### 4.3 Full system

**One take.** In one continuous simulation the robot walks to the table, picks the box, carries it to the panel with a 180° turn, places it, presses the button (34.5 mm, no re-pressing), and returns to rest. Every postcondition holds.

**Ticket to robot.** For the final run we wrote one SOP (SOP-1013) that uses all five embodied skills and added it to the corpus. The retriever ranks it first (score 0.897). Because the ticket and SOP were written together, and TF-IDF also ranks it first, this run shows the pipeline end to end rather than retrieval accuracy. Of three takes, the best is:

| Step | Measured outcome | Verdict |
|---|---|---|
| pick | box gripped, lifted, held | pass |
| carry, place | carried to the panel, placed upright | pass |
| press | 33.8 mm, held | pass |
| lever | 0.443 rad; success requires < 0.25 rad (best of 4 stance variants; others 0.595 and 0.627) | fail |
| notify | verdict card sent | pass |

*Table 3: **The system reports its own failure.** Verdict: "partially resolved." In the weakest take, place, press, and lever all failed and the verifier reported each.*

### 4.4 Reward hacks

| Reward term | What the policy did | Fix |
|---|---|---|
| Proximity income near the button | Hovered 5 cm short and collected it indefinitely | Narrow the band from 0.20 to 0.05 m, double the distance penalty |
| Progress along the lever arc | Sat in the lever's 0.17 rad of free travel | Measure progress from past the free travel, make it convex |
| Per-step press depth | Pressed, released, pressed again | End the episode on success |
| Hold counter after the press | Swung the arm while the counter ran | Stillness cost after the press; hold counter tolerates brief dips |
| Jerk penalty | Flailed fast enough that bumping the button paid more than the penalty (jerk 0.83) | Low-pass filter on the arm command, α = 0.12 (jerk 0.001) |
| Lift height | Launched the box out of a crushing pinch | Replace RL with a controller; verify a sustained held state |

*Table 4: **Every hack was found by watching rollouts, not from a metric.** Three fix classes recur: end what can be farmed, move constraints from the reward into the controller, and judge sustained states instead of instantaneous ones.*

**Limitations.** Everything runs in MuJoCo; no hardware transfer was attempted. The SOPs and tickets come from one generator, and there is no evaluation on real maintenance tickets. Because SOPs are written at skill granularity, planning is a lookup, and SOPs written in ordinary prose would need a learned planner. The retriever cannot decline. Skill results are small-sample, and the lever succeeds about half the time. The reward hacks were judged by one reviewer.

## 5. Conclusion

Sequent connects an operator's description of a problem to a humanoid carrying out an approved procedure, with a verifier that reports "partially resolved" when a step fails. A small retriever trained on operator language finds the right SOP first 58% of the time on vague tickets and 92% on clear ones, and its gain over the untuned model holds on procedures and machines it never saw. On the body, controllers outperformed RL for grasping on this embodiment, while RL with a curriculum handled pressing, and each reward hack was caught by watching rollouts.

**Future work.** The most important addition is for the retriever to ask instead of guess, using the confidence measured above, and to filter by the machine ID that a real ticket carries. Beyond that: hard-negative training against same-machine look-alike SOPs; a comparison against an LLM choosing among the retriever's candidates; a learned planner for SOPs written in prose; retraining the lever with the missing observation; moving the body to Isaac Lab for faster contact training; and a test set of real tickets. Code, data, and results: <https://github.com/jatinsikka/sequent-g1>.

## Acknowledgments

The AMO whole-body controller is by Li, Cheng et al. [9] and is used frozen. Training ran on Azure virtual machines.

## References

[1] M. Ahn et al. Do As I Can, Not As I Say: Grounding Language in Robotic Affordances. In *Conference on Robot Learning (CoRL)*, 2022.

[2] J. Liang et al. Code as Policies: Language Model Programs for Embodied Control. In *IEEE International Conference on Robotics and Automation (ICRA)*, 2023.

[3] W. Huang et al. Inner Monologue: Embodied Reasoning through Planning with Language Models. In *Conference on Robot Learning (CoRL)*, 2022.

[4] V. Karpukhin et al. Dense Passage Retrieval for Open-Domain Question Answering. In *Empirical Methods in Natural Language Processing (EMNLP)*, 2020.

[5] N. Reimers and I. Gurevych. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In *Empirical Methods in Natural Language Processing (EMNLP)*, 2019.

[6] P. Lewis et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2020.

[7] S. Robertson and H. Zaragoza. The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in Information Retrieval*, 3(4):333–389, 2009.

[8] A. Z. Ren et al. Robots That Ask For Help: Uncertainty Alignment for Large Language Model Planners. In *Conference on Robot Learning (CoRL)*, 2023.

[9] J. Li, X. Cheng, T. Huang, S. Yang, R.-Z. Qiu, and X. Wang. AMO: Adaptive Motion Optimization for Hyper-Dexterous Humanoid Whole-Body Control. In *Robotics: Science and Systems (RSS)*, 2025.

[10] J. Li et al. OKAMI: Teaching Humanoid Robots Manipulation Skills through Single Video Imitation. In *Conference on Robot Learning (CoRL)*, 2024.

[11] D. Amodei et al. Concrete Problems in AI Safety. arXiv:1606.06565, 2016.

[12] J. Skalse et al. Defining and Characterizing Reward Hacking. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2022.

[13] J. Schulman et al. Proximal Policy Optimization Algorithms. arXiv:1707.06347, 2017.

[14] S. Ross, G. Gordon, and D. Bagnell. A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. In *AISTATS*, 2011.

[15] E. Todorov, T. Erez, and Y. Tassa. MuJoCo: A Physics Engine for Model-Based Control. In *IEEE/RSJ IROS*, 2012.

## Appendix

**Retriever.** `sentence-transformers/all-MiniLM-L6-v2`, mean pooling, 384-dimensional output, maximum sequence length 256. Multiple-negatives ranking loss with cosine similarity and scale $\lambda = 20$. Batch 32, 3 epochs, AdamW with learning rate $2 \times 10^{-5}$, linear warmup over the first 10% of steps. Seeds 0–4 for Table 1 and 0–2 for the other experiments. One epoch gives R@1 0.539 ± 0.013 (3 seeds).

**Data.** 1,012 SOPs, 46 families × 22 SOPs. 2,338 incidents split 1,859 / 210 / 269 (train / validation / test); the split puts each SOP's first incident in training, so every test SOP is seen in training. Held-out-SOP split: 20% of SOPs sampled with seed 0, all their incidents used for test. Held-out-family split: five families sampled with seed 0 (air compressor, filling machine, laser cutter, mechanical press, press brake). No test ticket has a near-duplicate in training (highest TF-IDF cosine 0.31). Informed tickets: two SOPs per family sampled with seed 2026, one ticket each.

**Confidence cutoff.** Signals: the top-1 score, and the gap between the top-1 and top-2 scores. For each target precision (0.80, 0.90, 0.95) the cutoff is the lowest one that meets the target on the validation split; it is then applied to the test split.

**PPO** [13]. Stable-Baselines3, MLP policy with two hidden layers of 128 units, `n_steps` 1024 per environment, batch 1024, 10 epochs per update, learning rate $3 \times 10^{-4}$, $\gamma = 0.99$, 200-step episodes. Entropy coefficient 0.02 (press) and 0.05 (lever, followed by a low-entropy phase). Action range 0.8 rad (press) and 1.2 rad (lever) around the neutral pose; arm command low-pass filter α = 0.12. Lever: initial log-std −1.5, 33-dimensional observation. Training used 24–56 parallel environments on 32- and 64-core Azure VMs. With default threading, parallel workers each spawned a full set of torch threads and throughput fell to 18 steps/s; setting `OMP_NUM_THREADS=1` per worker restored about 2,600 steps/s.

**Simulator.** MuJoCo 3.2.3 [15], 29-DoF Unitree G1 with a Robotiq 2F-85 gripper on the right wrist, AMO at 50 Hz.

**Reproducing.** `brain/benchmark/REPRO_REPORT.md` lists every retrieval number with the script and CSV that produced it. RL checkpoints are not released.

## Paper checklist

1. For all student authors:
   a. Spell and grammar checked the full document? **[Yes]**
   b. LaTeX organized per the lab sample paper? **[No]** (standalone LaTeX generated from this file; lab template not used)
   c. All plots legible, at least 20 pt, using the plotting guidelines? **[Yes]** (Figs. 2 and 3 are generated from CSVs by `docs/figures/make_figures.py`)
   d. Peer reviewed by another student? **[No]**
   e. All references include the venue? **[Yes]**
   f. Common reference files used? **[N/A]**
   g. Paper outlined with Heilmeier answers at the top? **[Yes]** (`docs/OUTLINE.md`)
