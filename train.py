import os

import numpy as np
import sys

from kinematics import VisPlanarArms
from reservoir import *
from monitoring import Con_Monitor

sim_id = int(sys.argv[1])
training_trials = 2000

# parameters for training
train_arm = 'right'
learn_theta = np.array((60, 90))
init_thetas = (np.array((40, 40)),
               np.array((90, 50)))
num_init_thetas = len(init_thetas)

# presentation times
schedule = 20.  # in [ms]
t_execution = 20. * schedule
t_relax = 50.

# compile reservoir
compile_folder = f"networks/sim_{sim_id}/"
if not os.path.exists(compile_folder):
    os.makedirs(compile_folder)
ann.compile(directory=compile_folder)

# init monitors
m = ann.Monitor(output_pop, ['r'])

# init weights
w_res = Con_Monitor([w_recurrent])
w_res.extract_weights()

# initialize kinematics
learn_end_effector = VisPlanarArms.forward_kinematics(arm=train_arm, thetas=learn_theta, radians=False)[:, -1]
my_arms = VisPlanarArms(init_angles_left=init_thetas[0], init_angles_right=init_thetas[0], radians=False)

# init parameters
R_mean = np.zeros(num_init_thetas)
alpha = 0.75  # 0.33
error_history = []


def train_forward_model(sim_id: int, trial: int):
    global alpha, R_mean
    # Reinitialize network
    res_population.x = ann.Uniform(-0.1, 0.1).get_values(N)
    res_population.r = np.tanh(res_population.x)
    res_population[0].r = np.tanh(1.0)
    res_population[1].r = np.tanh(1.0)
    res_population[2].r = np.tanh(-1.0)

    # create input
    save_name = f"trajectories/sim_{sim_id}/run_{trial}"
    my_arms.training_fixed_position(arm=train_arm,
                                    init_angles=init_thetas[trial % num_init_thetas],
                                    radians=False,
                                    position=learn_end_effector,
                                    t_min=10, t_max=60,
                                    trajectory_save_name=save_name)

    t_trajectory = float(len(my_arms.trajectory_thetas_right))

    # input modality
    if train_arm == 'right':
        inp_arm.r = np.array((1., 0.))
    else:
        inp_arm.r = np.array((0., 1.))

    @ann.every(period=schedule)
    def set_inputs(n):
        # Set inputs to the network
        inp_theta.r = my_arms.trajectory_thetas_right[n]
        inp_gradient.r = my_arms.trajectory_gradient_right[n]

    ann.simulate(t_trajectory * schedule)

    # Relaxation
    for inp in input_pop:
        inp.r = 0.0

    ann.simulate(50)

    # Read the output
    rec = m.get()

    # Compute the target
    if train_arm == 'right':
        target = np.array(my_arms.end_effector_right) / 1000.0  # in [m]
    else:
        target = np.array(my_arms.end_effector_left) / 1000.0  # in [m]

    # Response is over the last 200 ms
    output_r = rec['r'][-int(t_execution):]  # neuron 100 over the last 200 ms

    # Compute the error
    error = np.linalg.norm(target - np.mean(output_r))

    # The first 25 trial do not learn, to let R_mean get realistic values
    if trial > 25:
        # Apply the learning rule
        w_recurrent.learning_phase = 1.0
        w_recurrent.error = error
        w_recurrent.mean_error = R_mean[trial % num_init_thetas]

        # Learn for one step
        ann.step()

        # Reset the traces
        w_recurrent.learning_phase = 0.0
        w_recurrent.trace = 0.0
        _ = m.get()  # to flush the recording of the last step

    # Update mean reward
    R_mean[trial % num_init_thetas] = alpha * R_mean + (1. - alpha) * error

    return error


# execute training
for trial in range(training_trials):
    error_history.append(train_forward_model(sim_id, trial))

# error history
np.save("error_hist.npy", error_history)
# weights
weight_folder = f"networks/sim_{sim_id}/"
w_res.extract_weights()
w_res.save_cons(weight_folder)
