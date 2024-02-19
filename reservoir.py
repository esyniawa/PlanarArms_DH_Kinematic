import numpy as np
import ANNarchy as ann
ann.clear()
# fixed seed to control network init
ann.setup(dt=1.0, num_threads=1)

# size reservoir
N = 400
dim_output = 40  # number of output neurons

BaselineNeuron = ann.Neuron(
    parameters="""
        baseline = 0.0
        noise = 0.0 : population
    """,
    equations="""
        r = baseline + noise * Uniform(-1.0,1.0): min=0.0
    """
)

BaselineNeuron_dynamic = ann.Neuron(
    parameters="""
        tau_up = 1.0 : population
        tau_down = 20.0 : population
        baseline = 0.0 : population
        noise = 0.0 : population
    """,
    equations="""
        base = baseline + noise * Uniform(-1.0,1.0): min=0.0
        dr/dt = if (baseline>0.01): (base-r)/tau_up else: -r/tau_down : min=0.0
    """,
    name = "Baseline Neuron",
    description = "Time dynamic Neuron with baseline to be set. "
)

miconi_neuron = ann.Neuron(
    parameters="""
        tau = 30.0 : population # Time constant
        constant = 0.0 # The four first neurons have constant rates
        alpha = 0.05 : population # To compute the sliding mean
        f = 12. : population # Frequency of the perturbation
        A = 20. : population # Perturbation amplitude. dt*A/tau should be 0.5...
    """,
    equations="""
        # Perturbation
        perturbation = if Uniform(0.0, 1.0) < f/1000.: 1.0 else: 0.0 
        noise = if perturbation > 0.5: A*Uniform(-1.0, 1.0) else: 0.0

        # ODE for x
        x += dt*(sum(in) + sum(exc) - x + noise)/tau

        # Output r
        rprev = r
        r = if constant == 0.0: tanh(x) else: tanh(constant)

        # Sliding mean
        delta_x = x - x_mean
        x_mean = alpha * x_mean + (1 - alpha) * x
    """
)

res_synapse = ann.Synapse(
    parameters="""
        eta = 1.2 : projection # Learning rate
        learning_phase = 0.0 : projection # Flag to allow learning only at the end of a trial
        error = 0.0 : projection # Reward received
        mean_error = 0.0 : projection # Mean Reward received
        max_weight_change = 0.0003 : projection # Clip the weight changes
    """,
    equations="""
        # Trace
        trace += if learning_phase < 0.5:
                    power(pre.rprev * (post.delta_x), 3)
                 else:
                    0.0

        # Weight update only at the end of the trial
        delta_w = if learning_phase > 0.5:
                eta * trace * (mean_error) * (error - mean_error)
             else:
                 0.0 : min=-max_weight_change, max=max_weight_change
        w -= if learning_phase > 0.5:
                delta_w
             else:
                 0.0
    """
)

# Input population
inp_theta = ann.Population(geometry=2, neuron=BaselineNeuron, name="input_theta")
inp_gradient = ann.Population(geometry=2, neuron=BaselineNeuron, name="input_gradients")
inp_arm = ann.Population(geometry=2, neuron=BaselineNeuron, name="input_arm")

input_pops = (inp_theta, inp_gradient, inp_arm)

# Recurrent population
res_population = ann.Population(geometry=N, neuron=miconi_neuron, name="reservoir")
res_population[0].constant = 1.0
res_population[1].constant = 1.0
res_population[2].constant = -1.0
res_population.x = ann.Uniform(-0.2, 0.2)

# projections
# input weights
w_input = {}
for input_pop in input_pops:
    w_input[input_pop.name] = ann.Projection(pre=input_pop, post=res_population,
                                             target='exc',
                                             name=input_pop.name)
    w_input[input_pop.name].connect_all_to_all(weights=ann.Uniform(-1.0, 1.0))

# Recurrent weights
g = 1.5
w_recurrent = ann.Projection(res_population, res_population, 'exc', res_synapse, name='lat_res')
w_recurrent.connect_all_to_all(weights=ann.Normal(0., g/np.sqrt(N)), allow_self_connections=True)

# output populations (don't use negative indeces)
output_pop = res_population[(N - dim_output - 1):(N - 1)]
