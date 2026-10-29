
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

path = str(sys.argv[1])

df = pd.read_csv(path, header=1)


fig, ax = plt.subplots(1,1)

# FOR to_csv_pedestals

ax.plot(df["channel"], df["pedestals"], '-o', ms=2)

plt.show()


# FOR to_csv_adc

# run_group = df.groupby('run')

# for run_id, run_df in run_group:
#     # if run_id < 1000 or run_id > 1020:
#     #     continue
#     # print(run_id)
#     # run_df = run_df[run_df["adc"] > 0]
#     ax.plot(run_df["channel"], run_df["adc"], '-o', ms=2)

# plt.show()
