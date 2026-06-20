"""Plot the SB3 CSV training log into the per-run training graph: reward curve +
press-depth curve + fraction-of-steps-held curve over timesteps. No pandas dep.
Run: python plot_bp.py <logdir> <out.png>
"""
import csv, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

logdir, out = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(f"{logdir}/progress.csv")))

def col(key):
    xs, ys = [], []
    for r in rows:
        t, v = r.get("time/total_timesteps", ""), r.get(key, "")
        if t and v:
            try: xs.append(float(t)); ys.append(float(v))
            except ValueError: pass
    return xs, ys

fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
xr, yr = col("rollout/ep_rew_mean")
ax[0].plot(xr, yr, c="tab:purple", lw=2)
ax[0].set_ylabel("ep reward mean"); ax[0].set_title("bp-v3 (BC warm-start + PPO) — training")

xd, yd = col("press/max_disp_cm")
xm, ym = col("press/mean_disp_cm")
ax[1].plot(xd, yd, c="tab:blue", lw=2, label="max press depth (cm)")
ax[1].plot(xm, ym, c="tab:cyan", lw=1.5, label="mean press depth (cm)")
ax[1].axhline(2.0, ls="--", c="tab:red", label="press threshold (2cm)")
ax[1].set_ylabel("press depth (cm)"); ax[1].set_xlabel("timesteps"); ax[1].legend(loc="upper left")
ax2 = ax[1].twinx()
xf, yf = col("press/frac_held_2cm")
ax2.plot(xf, yf, c="tab:green", lw=1.5, ls=":", label="frac steps held >2cm")
ax2.set_ylabel("frac held >2cm"); ax2.set_ylim(0, 1); ax2.legend(loc="upper right")

plt.tight_layout(); plt.savefig(out, dpi=110)
print(f"saved {out}")
