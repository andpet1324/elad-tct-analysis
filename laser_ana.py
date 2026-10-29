import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

SNS_DEEP = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
            "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
ALPHA = 0.08

path = str(sys.argv[1])
path_pedestals = str(sys.argv[2])

df = pd.read_csv(path, header=1)
pedestals = pd.read_csv(path_pedestals, header=1)

# Region of interest

ch_min = 70
ch_max = 120
channels_mask = (df["channel"] > ch_min) & (df["channel"] < ch_max)
df = df[channels_mask]

# Only chose a select time
delay = 50
df = df[abs(df["time"] - delay) < 2]

fig, ax = plt.subplots(1,1)
# Subtract pedestals

df = df.merge(pedestals[["channel", "pedestals"]], on="channel", how="left")
df["signal"] = df["adc"] - df["pedestals"]

# Plot adcs
run_group = df.groupby('run')
for run_id, run_df in run_group:
    ax.plot(run_df["channel"], run_df["signal"], '-o', ms=2, color=SNS_DEEP[9], alpha=ALPHA)

ax.set_title(f"alpha {ALPHA}, with {len(run_group)} triggers")
ax.set_xlabel("Channel")
ax.set_ylabel("ADC")

fig.savefig(f"laser_signal.png", dpi=150)

# HIT MAP

HIT_THRESHOLD = 25

fig_hits, ax_hits = plt.subplots(1,1)

hit_df = df[df["signal"].abs() > HIT_THRESHOLD]

# Plot
hits = (hit_df["signal"] != 0).groupby(hit_df["channel"]).sum()
ax_hits.stairs(hits.values, np.append(hits.index, hits.index[-1] + 1) - 0.5)


ax_hits.set_title(f"{len(run_group)} triggers with hit threshold {HIT_THRESHOLD} ADC")
ax_hits.set_xlabel("Channel")
ax_hits.set_ylabel("# Hits")

fig_hits.savefig(f"hit_signal.png", dpi=150)

# SIGNAL


fig_signal, ax_signal = plt.subplots(1,1)

ax_signal.hist(hit_df["adc"])

ax_signal.set_title(f"{len(run_group)} triggers")
ax_signal.set_xlabel("ADC")
ax_signal.set_ylabel("Counts")

fig_signal.savefig(f"signal_hist.png", dpi=150)