import numpy as np
import matplotlib.pyplot as plt

import os
import sys

from functions import load_monitors

sim_id = sys.argv[1]
folder = "plots/"
if not os.path.exists(folder):
    os.makedirs(folder)

weights = load_monitors(sim_id, 'w')

# reservoir weights
fig1, axs = plt.subplots(ncols=3, figsize=(10, 5))
axs[0].imshow(weights['w_lat_res'][0, :, :], cmap='RdBu', vmin=-0.2, vmax=0.2)
axs[1].imshow(weights['w_lat_res'][1, :, :], cmap='RdBu', vmin=-0.2, vmax=0.2)
axs[2].imshow(weights['w_lat_res'][1, :, :] - weights['w_lat_res'][0, :, :], cmap='RdBu', vmin=-0.02, vmax=0.02)

plt.savefig(folder + f"weight_changes_sim{sim_id}.pdf")
plt.close(fig1)

# input weights
fig2, axs = plt.subplots(nrows=3, ncols=2, figsize=(8, 10))
# arm modality
axs[0, 0].imshow(weights['w_input_arm'][0, :, 0].reshape((20, 20)), cmap='RdBu', vmin=-1, vmax=1)
axs[0, 1].imshow(weights['w_input_arm'][0, :, 1].reshape((20, 20)), cmap='RdBu', vmin=-1, vmax=1)
# theta
axs[1, 0].imshow(weights['w_input_theta'][0, :, 0].reshape((20, 20)), cmap='RdBu', vmin=-1, vmax=1)
axs[1, 1].imshow(weights['w_input_theta'][0, :, 1].reshape((20, 20)), cmap='RdBu', vmin=-1, vmax=1)
# gradients
axs[2, 0].imshow(weights['w_input_gradients'][0, :, 0].reshape((20, 20)), cmap='RdBu', vmin=-1, vmax=1)
axs[2, 1].imshow(weights['w_input_gradients'][0, :, 1].reshape((20, 20)), cmap='RdBu', vmin=-1, vmax=1)

plt.savefig(folder + f"input_weights_sim{sim_id}.pdf")
plt.close(fig2)
