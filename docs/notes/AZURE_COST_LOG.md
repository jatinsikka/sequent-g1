# Azure Cost Log — Sequent Robotics RL Training

Sponsorship credits: $10,000 (expires 2028-04-30). Quota: 65 vCPU regional (East US 2). GPU quota pending (A100 ticket filed 2026-06-12).

**Rule: deallocate when idle; stopped-but-not-deallocated VMs still bill. And deallocate only pauses compute (disk + public IP keep dripping ~$0.70/day), so DELETE the resource group when done for a while.**

**STATUS 2026-06-17: `sequent-rl-rg` (VM, 128GB disk, public IP, NIC, VNET) DELETED. Final June spend ~$8.96. Idle drip stopped. Everything reproducible (repo on GitHub, v5.5 + v5.6 checkpoints local). Recreate via scripted setup when cloud compute is needed again. Future GPU path = AWS credits (~5 days out) or an Azure A10 request.**

**STATUS 2026-06-19: RE-PROVISIONED for button-press training. `sequent-cpu32` recreated in `sequent-rl-rg`/eastus2. NOTE: F32as_v7 hit `AllocationFailed` (no regional capacity) → switched to sibling `Standard_F32als_v7` (32c/62GB low-mem, same Fasv7 quota, ~$1.16/hr). Provisioning recipe (no checked-in script): apt python3-pip/venv/git → venv + pip(torch-cpu, mujoco, sb3, gymnasium, tqdm, rich) → `git clone jatinsikka/sequent-g1` (carries amo_jit.pt + g1.xml) → scp the 3 locally-fixed files (env_wrapper_button/reward_fn/train_button). GOTCHAS hit: (1) play_amo.py top-level `import glfw`/`mujoco_viewer` crash headless → guard with try/except; (2) amo_jit.pt has `cuda:0` baked in TorchScript → zip-patch .py entries cuda:0→cpu (kept `_gpu.pt` backup); (3) SubprocVecEnv MUST run with `PYTHONPATH=~/sequent` (workers re-import) + `start_method="spawn"` (forkserver = fork-after-torch segfault of main) + `OMP_NUM_THREADS=1` (else 32×32 threads → load 928). With all three: 1474 fps (30× local), load ~17. Auto-shutdown set 23:30 UTC. DELETE RG when done.**

## Resources

| Resource | Size | Region | $/hr (est) | Created | Auto-shutdown |
|---|---|---|---|---|---|
| sequent-cpu32 | Standard_F32as_v7 (32c/125GB) | East US 2 | ~$1.28 | 2026-06-12 | 19:30 UTC daily |

## Run ledger

| Date | Run name | VM | Hours (est) | Cost (est) | Notes |
|---|---|---|---|---|---|
| 2026-06-12 | (setup + pipeline validation) | sequent-cpu32 | ~1.0 | ~$1.30 | VM created, env install, smoke tests, cuda→cpu JIT patch |
| 2026-06-12 | v5.6-grasp-0612 (10M steps, 32 envs) | sequent-cpu32 | ~2.2 | ~$2.80 | 1,521 fps; det eval: grasp 45%, lift 0% (regression vs v5.5) + GIF render. VM deallocated 15:30 UTC. Day total ≈ $4.20 |
| 2026-06-19 | bp-v0-red (button_red, 2M steps, 32 envs) | sequent-cpu32 (F32als_v7) | ~0.5 (incl. setup) | ~$0.60 | re-provision + provision (~30min) then training at 1,474 fps; 2M steps ≈ 22min. Eval pending. DELETE RG after eval. |
| 2026-06-20 | OVERNIGHT: bp-v4 (2M) + bp-v5 resume (4M) + grasp v6/v7/v8 (3M each) | sequent-cpu32 (F32als_v7) | ~3.5 | ~$4.10 | **press_button SOLVED + verified (bp-v5: held 79 steps).** Full SOP demo working, site live. grasp smoothness retrain (v6-v8) all went timid → v5.5 stands. Auto-shutdown (23:30 UTC) fired mid-grasp-v8 → restarted, removed auto-shutdown, re-evaled. **VM DEALLOCATED at end (compute billing stopped; disk + all checkpoints persist).** ~$13 spent of $10k credits total. Restart with `az vm start` to continue; DELETE RG when fully done. |

## Monthly burn check

| Month | Credits used (portal) | Remaining |
|---|---|---|
| 2026-06 | ~$8.96 (VM run + idle drip; RG deleted 6/17) | ~$9,991 |
