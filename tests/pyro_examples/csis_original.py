import torch
import torch.nn as nn
import torch.functional as F

import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.optim

import os
smoke_test = ('CI' in os.environ)
n_steps = 2 if smoke_test else 2000

def model(prior_mean, observations={"x1": 0, "x2": 0}):
    x = pyro.sample("z", dist.Normal(prior_mean, torch.tensor(5**0.5)))
    y1 = pyro.sample("x1", dist.Normal(x, torch.tensor(2**0.5)), obs=observations["x1"])
    y2 = pyro.sample("x2", dist.Normal(x, torch.tensor(2**0.5)), obs=observations["x2"])
    return x

class Guide(nn.Module):
    def __init__(self):
        super(Guide, self).__init__()
        self.neural_net = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2))

    def forward(self, prior_mean, observations={"x1": 0, "x2": 0}):
        pyro.module("guide", self)
        x1 = observations["x1"]
        x2 = observations["x2"]
        v = torch.cat((x1.view(1, 1), x2.view(1, 1)), 1)
        v = self.neural_net(v)
        mean = v[0, 0]
        std = v[0, 1].exp()
        pyro.sample("z", dist.Normal(mean, std))

guide = Guide()

optimiser = pyro.optim.Adam({'lr': 1e-3})
csis = pyro.infer.CSIS(model, guide, optimiser, num_inference_samples=50)
prior_mean = torch.tensor(1.)

for step in range(n_steps):
    csis.step(prior_mean)

posterior = csis.run(prior_mean,
                     observations={"x1": torch.tensor(8.),
                                   "x2": torch.tensor(9.)})

marginal = pyro.infer.EmpiricalMarginal(posterior, "z")


import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

# Draw samples from empirical marginal for plotting
csis_samples = [marginal().detach() for _ in range(1000)]

# Calculate empirical marginal with importance sampling
is_posterior = pyro.infer.Importance(model, num_samples=50).run(prior_mean,
                                                                observations={"x1": torch.tensor(8.),
                                                                              "x2": torch.tensor(9.)})
is_marginal = pyro.infer.EmpiricalMarginal(is_posterior, "z")
is_samples = [is_marginal().detach() for _ in range(1000)]

# Calculate true prior and posterior over z
true_posterior_z = np.arange(-10, 10, 0.05)
true_posterior_p = np.array([np.exp(scipy.stats.norm.logpdf(p, loc=7.25, scale=(5/6)**0.5)) for p in true_posterior_z])
prior_z = true_posterior_z
prior_p = np.array([np.exp(scipy.stats.norm.logpdf(z, loc=1, scale=5**0.5)) for z in true_posterior_z])

plt.rcParams['figure.figsize'] = [30, 15]
plt.rcParams.update({'font.size': 30})
fig, ax = plt.subplots()
plt.plot(prior_z, prior_p, 'k--', label='Prior')
plt.plot(true_posterior_z, true_posterior_p, color='k', label='Analytic Posterior')
plt.hist(csis_samples, range=(-10, 10), bins=100, color='r', density=1, label="Inference Compilation")
plt.hist(is_samples, range=(-10, 10), bins=100, color='b', density=1, label="Importance Sampling")
plt.xlim(-8, 10)
plt.ylim(0, 5)
plt.xlabel("z")
plt.ylabel("Estimated Posterior Probability Density")
plt.legend()
plt.show()
