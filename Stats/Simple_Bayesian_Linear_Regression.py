# This is a conceptual example for PyMC.
# Installation: pip install pymc arviz
# PyMC requires a bit more setup than a few lines of code
# due to its probabilistic programming nature.

# import pymc as pm
# import arviz as az
# import numpy as np

# # Sample Data
# np.random.seed(42)
# size = 100
# true_alpha = 1
# true_beta = 2
# x = np.linspace(0, 1, size)
# y = true_alpha + true_beta * x + np.random.normal(0, 0.5, size=size)

# with pm.Model() as linear_model:
#     # Priors for the parameters
#     alpha = pm.Normal('alpha', mu=0, sigma=10)
#     beta = pm.Normal('beta', mu=0, sigma=10)
#     sigma = pm.HalfNormal('sigma', sigma=1)

#     # Expected value of y
#     mu = pm.Deterministic('mu', alpha + beta * x)

#     # Likelihood (sampling distribution) of observations
#     Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y)

#     # Sample from the posterior distribution
#     trace = pm.sample(2000, tune=1000, return_inferencedata=True, cores=1) # cores=1 for simplicity

# # Posterior Analysis (conceptual)
# # az.summary(trace, var_names=['alpha', 'beta', 'sigma'])
# # az.plot_trace(trace)
# # plt.show()
