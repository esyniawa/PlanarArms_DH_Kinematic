import sys
import os

import numpy as np
sim_id = int(sys.argv[1])
# fix seed for testing
np.random.seed(sim_id)

from reservoir import *
from kinematics import VisPlanarArms
from functions import moving_average

from pybads.bads import BADS
from contextlib import contextmanager

import matplotlib.pyplot as plt

# define variables
do_plot = True

error_history: list = []
prediction: list = []

# supress standard output
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# For resetting weights
class Initial_weights(object):
    def __init__(self, connections):
        self.connections = connections

    def write_weights(self, folder):
        for con in self.connections:
            if not os.path.exists(folder):
                os.makedirs(folder + 'projections/')

            con.save(folder + 'projections/' + con.name + '.npz')

    def load_weights(self, folder):
        for con in self.connections:
            con.load(folder + 'projections/' + con.name + '.npz')


# Compute the mean reward per trial
def fit_reservoir(arm: str,
                  end_effector: np.ndarray,
                  training_trials: int,
                  init_thetas: tuple | list = (np.array((40, 40)), np.array((90, 50))),
                  accumulate_trials: int = 50,  # calc mean last
                  alpha: float = 0.75,
                  initial_eta: float = 0.8,
                  initial_A: float = 20.,
                  initial_f: float = 9.):

    my_params = (initial_eta, initial_A, initial_f)
    num_init_thetas = len(init_thetas)
    R_mean = np.zeros(num_init_thetas)

    # presentation times
    schedule = 20.  # in [ms]
    t_relax = 200.  # in [ms]
    t_output = t_relax  # in [ms]

    # compile reservoir
    compile_folder = f"networks/fit_seed_{sim_id}/"
    if not os.path.exists(compile_folder):
        os.makedirs(compile_folder)
    ann.compile(directory=compile_folder)

    # init reservoir weights
    init_w = Initial_weights([w_recurrent, ])
    init_w.write_weights(compile_folder)

    # init monitors
    m = ann.Monitor(output_pop, ['r'])

    # initialize kinematics
    my_arms = VisPlanarArms(init_angles_left=init_thetas[0], init_angles_right=init_thetas[0], radians=False)

    def loss_function(res_params,
                      weight_mean=1.0,
                      weight_sd=1.0):

        param_eta, param_A, param_f = res_params

        # set parameters in reservoir
        w_recurrent.eta = param_eta
        res_population.A = param_A
        res_population.f = param_f

        # reset weights
        init_w.load_weights(compile_folder)

        # reset history
        global error_history, prediction
        
        error_history = []
        prediction = []

        # learn with reservoir
        for trial in range(training_trials):

            id_init = trial % num_init_thetas
            # Reinitialize network
            res_population.x = ann.Uniform(-0.1, 0.1).get_values(N)
            res_population.r = np.tanh(res_population.x)
            res_population[0].r = np.tanh(1.0)
            res_population[1].r = np.tanh(1.0)
            res_population[2].r = np.tanh(-1.0)

            # create input
            save_name = f"trajectories/fit_seed_{sim_id}/run_{trial}"
            my_arms.training_fixed_position(arm=arm,
                                            init_angles=init_thetas[id_init],
                                            radians=False,
                                            position=end_effector,
                                            t_min=10, t_max=60, t_wait=1,
                                            trajectory_save_name=save_name)

            t_trajectory = float(len(my_arms.trajectory_thetas_right))

            # input modality
            if arm == 'right':
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
            if arm == 'right':
                current_position = my_arms.end_effector_right[-1] / 10.0  # in [cm]
            else:
                current_position = my_arms.end_effector_left[-1] / 10.0  # in [cm]

            # Response is over the last ms (dim: (t, neuron, space))
            output_r = np.array(rec['r'][-int(t_output):, :]).reshape((int(t_output), int(dim_output / 2), 2))

            # mean over time
            output_r = np.mean(output_r, axis=0)

            # sum over neurons
            output_r = np.sum(output_r, axis=0)

            # Compute the error
            error = np.linalg.norm(current_position - output_r)

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

            # reset network
            ann.reset(monitors=False)

            error_history.append(error)
            prediction.append(output_r)

        my_mean = np.mean(error_history[-accumulate_trials:])
        my_sd = np.std(error_history[-accumulate_trials:])

        fitting_error = weight_mean * np.log(my_mean) + weight_sd * np.log(my_sd)

        return fitting_error

    target = loss_function

    bads = BADS(target, np.array(my_params),
                lower_bounds=np.array((0, 0, 0)),
                plausible_lower_bounds=np.array((0.01, 2, 1)),
                upper_bounds=np.array((10, 100, 40)),
                plausible_upper_bounds=np.array((5, 50, 20)))

    optimize_result = bads.optimize()
    fitted_params = optimize_result['x']

    return fitted_params


# run optimization
if __name__ == '__main__':

    training_arm = 'right'
    end_effector = VisPlanarArms.forward_kinematics(arm=training_arm, thetas=np.array((60, 90)), radians=False)[:, -1]

    with suppress_stdout():
        res = fit_reservoir(arm=training_arm,
                            end_effector=end_effector,
                            training_trials=250)

    results_folder = f'results/fit_seed_{sim_id}/'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    np.save(results_folder + 'fitted_params.npy', res)
    np.save(results_folder + 'error_history.npy', error_history)
    np.save(results_folder + 'prediction_history.npy', prediction)

    if do_plot:
        c = 5
        mov_error = moving_average(error_history, c)
        x = np.arange(c, len(error_history) + 1)
        fig, ax = plt.subplots()
        ax.plot(error_history, color='k')
        ax.plot(x, mov_error, color='r')
        ax.set_xlabel('Number of trials')
        ax.set_ylabel('Error in [cm]')
        plt.savefig(results_folder + f'/error_seed_{sim_id}.pdf')
        plt.close(fig)
