import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import os
import sys

from functions import load_monitors

sim_id = sys.argv[1]
tInit = 0

# load monitors
rates = load_monitors(sim_id, 'r')
nt, _ = rates['r_reservoir'].shape

# plot populations in subplot row wise
fig = plt.figure()

# input modality
ax1 = fig.add_subplot(3, 2, 1)
plt.title('Modality', fontsize=10)
l_mod = ax1.bar(x=['right', 'left'], height=rates['r_input_arm'][tInit, :], width=.5)
ax1.set_ylim((0, 1.5))

# input theta
ax2 = fig.add_subplot(3, 2, 3)
plt.title('Theta', fontsize=10)
l_th = ax2.bar(x=['shoulder', 'elbow'], height=rates['r_input_theta'][tInit, :], width=.5)
ax2.set_ylim((0, 3.5))

# input gradients
ax3 = fig.add_subplot(3, 2, 5)
plt.title('Delta theta', fontsize=10)
l_gr = ax3.bar(x=['delta shoulder', 'delta elbow'], height=rates['r_input_gradients'][tInit, :], width=.5)
ax3.set_ylim((-0.2, 0.2))

# reservoir
rates['r_reservoir'] = rates['r_reservoir'].reshape((nt, 20, 20))
ax4 = fig.add_subplot(1, 2, 2)
l_res = ax4.imshow(rates['r_reservoir'][tInit, :, :], cmap='RdBu', origin='lower', vmin=-1, vmax=1)

axslider = plt.axes([0.25, 0.05, 0.5, 0.03])
time_slider = Slider(
    ax=axslider,
    label='Time [ms]',
    valmin=0,
    valmax=nt-1,
    valinit=0,
)

def update(val):
    t = int(time_slider.val)
    l_res.set_data(rates['r_reservoir'][t, :, :])

    for i, bar in enumerate(l_mod):
        bar.set_height(rates['r_input_arm'][t, i])

    for i, bar in enumerate(l_th):
        bar.set_height(rates['r_input_theta'][t, i])

    for i, bar in enumerate(l_gr):
        bar.set_height(rates['r_input_gradients'][t, i])


time_slider.on_changed(update)
plt.show()
