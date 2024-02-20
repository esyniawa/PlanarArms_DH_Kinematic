import os

import numpy as np
import sys

sim_id = int(sys.argv[1])
# fix seed for testing
np.random.seed()

from kinematics import VisPlanarArms
from reservoir import *
from monitoring import Con_Monitor, Pop_Monitor

training_trials = 5000

# parameters for training
train_arm = 'right'
learn_theta = np.array((60, 90))
init_thetas = (np.array((40, 40)),
               np.array((90, 50)))
num_init_thetas = len(init_thetas)

# presentation times
schedule = 20.  # in [ms]
t_relax = 200.  # in [ms]
t_output = t_relax  # in [ms]

# compile reservoir
compile_folder = f"networks/sim_{sim_id}/"
if not os.path.exists(compile_folder):
    os.makedirs(compile_folder)
ann.compile(directory=compile_folder)

# init monitors
m = ann.Monitor(output_pop, ['r'])
m_pops = Pop_Monitor(list(input_pops) + [res_population], sampling_rate=1)

# init weights
w_res = Con_Monitor([w_recurrent])
w_res.extract_weights()
w_in = Con_Monitor([w_input[pop.name] for pop in input_pops])
w_in.extract_weights()

# initialize kinematics
learn_end_effector = VisPlanarArms.forward_kinematics(arm=train_arm, thetas=learn_theta, radians=False)[:, -1]
my_arms = VisPlanarArms(init_angles_left=init_thetas[0], init_angles_right=init_thetas[0], radians=False)

# init parameters
R_mean = np.zeros(num_init_thetas)
alpha = 0.33  # 0.33

error_history = []
prediction_history = []


def train_forward_model(sim_id: int, trial: int):
    global alpha, R_mean

    if trial == (training_trials - 10):
        m_pops.start()

    id_init = trial % num_init_thetas
    # Reinitialize network
    res_population.x = ann.Uniform(-0.1, 0.1).get_values(N)
    res_population.r = np.tanh(res_population.x)
    res_population[0].r = np.tanh(1.0)
    res_population[1].r = np.tanh(1.0)
    res_population[2].r = np.tanh(-1.0)

    # create input
    save_name = f"trajectories/sim_{sim_id}/run_{trial}"
    my_arms.training_fixed_position(arm=train_arm,
                                    init_angles=init_thetas[id_init],
                                    radians=False,
                                    position=learn_end_effector,
                                    t_min=10, t_max=60, t_wait=1,
                                    trajectory_save_name=save_name)

    t_trajectory = float(len(my_arms.trajectory_thetas_right))

    # input modality
    if train_arm == 'right':
        inp_arm.baseline = np.array((1., 0.))
    else:
        inp_arm.baseline = np.array((0., 1.))

    @ann.every(period=schedule)
    def set_inputs(n):
        # Set inputs to the network
        inp_theta.baseline = my_arms.trajectory_thetas_right[n]
        inp_gradient.baseline = my_arms.trajectory_gradient_right[n]

    ann.simulate(t_trajectory * schedule)

    # Relaxation
    for inp in input_pop:
        inp.baseline = 0.0

    ann.simulate(t_relax)

    # Read the output
    rec = m.get()

    # Compute the target (last end effector)
    if train_arm == 'right':
        target = my_arms.end_effector_right[-1] / 10.0  # in [cm]
    else:
        target = my_arms.end_effector_left[-1] / 10.0  # in [cm]

    # Response is over the last ms (dim: (t, neuron, space))
    output_r = np.array(rec['r'][-int(t_output):, :]).reshape((int(t_output), int(dim_output/2), 2))

    # mean over time
    output_r = np.mean(output_r, axis=0)

    # sum over neurons
    output_r = np.sum(output_r, axis=0)

    # Compute the error
    error = np.linalg.norm(target - output_r)

    # The first 20 trial do not learn, to let R_mean get realistic values
    if trial > 20:
        # Apply the learning rule
        w_recurrent.learning_phase = 1.0
        w_recurrent.error = error
        w_recurrent.mean_error = R_mean[id_init]

        # Learn for one step
        ann.step()

        # Reset the traces
        w_recurrent.learning_phase = 0.0
        w_recurrent.trace = 0.0
        _ = m.get()  # to flush the recording of the last step

    # Update mean reward
    R_mean[id_init] = alpha * R_mean[id_init] + (1. - alpha) * error

    if trial == training_trials:
        m_pops.stop()

    # reset network
    ann.reset(monitors=False)

    return error, output_r


# execute training
for trial in range(training_trials):
    my_error, my_prediction = train_forward_model(sim_id, trial)
    error_history.append(my_error)
    prediction_history.append(my_prediction)

# error history
results_folder = f"results/sim_{sim_id}/"
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

np.save(results_folder + "error_hist.npy", error_history)
np.save(results_folder + "r_mean.npy", R_mean)

# monitors
m_pops.save(results_folder)

# weights
w_in.save_cons(results_folder)

w_res.extract_weights()
w_res.save_cons(results_folder)
