using AdvancedHMC

n_samples, n_adapts, alpha_target = 10_000, 2_000, 0.8 # sampling parameter
q_init = randn(D)   # Draw a random starting points

### Building up NUTS

metric = DiagEuclideanMetric(D)             # metric space
h = Hamiltonian(metric, logdensity, grad)   # hamiltonian
eps_init = find_good_eps(h, q_init)         # initial step size
int = Leapfrog(eps_init)                    # integrator

# Multinomial sampling with generalised no U-turn
traj = NUTS{Multinomial,GeneralisedNoUTurn}(int)    

# Stan's windowed adaptor
adaptor = StanHMCAdaptor(
    n_adapts, 
    Preconditioner(metric), 
    NesterovDualAveraging(alpha_target, eps_init)
)

# Draw samples via simulating Hamiltonian dynamics
# - `samples` will store the samples
# - `stats` will store statistics for each sample
samples, stats = sample(h, traj, q_init, n_samples, adaptor, n_adapts)