import numpy as np
import matplotlib.pyplot as plt

import os
import sys

sim_id = sys.argv[1]
folder = "plots/"
if not os.path.exists(folder):
    os.makedirs(folder)

error = np.load(f"results/sim_{sim_id}/error_hist.npy")

fig, ax = plt.subplots()
ax.plot(error)
ax.set_ylabel("Error")
ax.set_xlabel("Trials")

plt.savefig(folder + f"error_history_sim{sim_id}.pdf")
plt.close(fig)
