using AdvancedHMC

n_samples, n_adapts, target = 10_000, 2_000, 0.8 # sampling parameter
q_init = randn(D)   # Draw a random starting points

### Building up NUTS

# Metric space
metric = DiagEuclideanMetric(D)
# Hamiltonian
h = Hamiltonian(metric, logdensity, grad)
# Initial step size
eps_init = find_good_eps(h, q_init)
# Integrator
int = Leapfrog(eps_init)                    
# Multinomial sampling with generalised no U-turn
traj = NUTS{Multinomial,GeneralisedNoUTurn}(int)    
# Stan's windowed adaptor
adaptor = StanHMCAdaptor(
    n_adapts, 
    Preconditioner(metric), 
    NesterovDualAveraging(target, eps_init)
)

# Draw samples via simulating Hamiltonian dynamics
# - `samples` will store the samples and `stats` will store statistics for each sample
samples, stats = sample(h, traj, q_init, n_samples, adaptor, n_adapts)