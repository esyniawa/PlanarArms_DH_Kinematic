import numpy as np
import matplotlib.pyplot as plt

import os
import sys

sim_ids = np.arange(1, 3, 1)

folder = "plots/"
if not os.path.exists(folder):
    os.makedirs(folder)

cul_error = []
for sim_id in sim_ids:
    error = np.load(f"results/sim_{sim_id}/error_hist.npy")
    cul_error.append(error)

    fig, ax = plt.subplots()
    ax.plot(error)
    ax.set_ylabel("Error in [cm]")
    ax.set_xlabel("Trials")

    ax.set_ylim((0, 50))

    plt.savefig(folder + f"error_history_sim{sim_id}.pdf")
    plt.close(fig)

# plot error over all simulations
cul_error = np.array(cul_error)
mean_error = np.mean(cul_error, axis=0)
sd_error = np.std(cul_error, axis=0)

new_fig, ax = plt.subplots()
ax.plot(mean_error)
ax.fill_between(np.arange(0, len(mean_error)), mean_error+sd_error, mean_error-sd_error, alpha=0.4)

ax.set_ylabel("Mean error in [cm]")
ax.set_xlabel("Trials")

ax.set_ylim((0, 50))

plt.savefig(folder + 'mean_error.pdf')
plt.close(new_fig)
