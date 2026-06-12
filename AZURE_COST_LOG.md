# Azure Cost Log — Sequent Robotics RL Training

Sponsorship credits: $10,000 (expires 2028-04-30). Quota: 65 vCPU regional (East US 2). GPU quota pending (A100 ticket filed 2026-06-12).

**Rule: deallocate (`az vm deallocate -g sequent-rl-rg -n sequent-cpu32`) when idle — stopped-but-not-deallocated VMs still bill.**

## Resources

| Resource | Size | Region | $/hr (est) | Created | Auto-shutdown |
|---|---|---|---|---|---|
| sequent-cpu32 | Standard_F32as_v7 (32c/125GB) | East US 2 | ~$1.28 | 2026-06-12 | 19:30 UTC daily |

## Run ledger

| Date | Run name | VM | Hours (est) | Cost (est) | Notes |
|---|---|---|---|---|---|
| 2026-06-12 | (setup + pipeline validation) | sequent-cpu32 | ~1.0 | ~$1.30 | VM created, env install, smoke tests, cuda→cpu JIT patch |
| 2026-06-12 | v5.6-grasp-0612 (10M steps, 32 envs) | sequent-cpu32 | TBD | TBD | first cloud run; auto-shutdown 19:30 UTC may cut it — checkpoints save periodically |

## Monthly burn check

| Month | Credits used (portal) | Remaining |
|---|---|---|
| 2026-06 | — | $10,000 |
