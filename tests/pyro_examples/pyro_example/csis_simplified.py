# Compiled Sequential Importance Sampling 
# http://pyro.ai/examples/csis.html
#   "The model is specified in the same way as any Pyro model, except that a keyword argument, observations, must be used to input a dictionary with each observation as a key..."  
# 
# csis = pyro.infer.CSIS(model, ...) 
# pyro.infer.CSIS forces model to have a keyword argument 'observations' of dictionary type ... Right now, I do not know how to simplify it.

import torch
import torch.nn as nn
import torch.functional as F

import pyro
# import pyro.distributions as dist
from pyro.distributions import Normal
import pyro.infer
import pyro.optim

import os
smoke_test = ('CI' in os.environ)
n_steps = 2 if smoke_test else 2000
data = {'x1': torch.tensor(8.), 'x2': torch.tensor(9.)}

#########
# model #
#########
prior_mean = torch.tensor(1.)

# def model(prior_mean, observations={'x1': 0, 'x2': 0}):
def model(observations={'x1': 0, 'x2': 0}):
    obs = torch.tensor([float(observations['x1']),
                        float(observations['x2'])])
    x = pyro.sample("z", Normal(prior_mean, torch.tensor(5**0.5)))
    # y1 = pyro.sample("x1", Normal(x, torch.tensor(2**0.5)), obs=observations['x1'])
    # y2 = pyro.sample("x2", Normal(x, torch.tensor(2**0.5)), obs=observations['x2'])
    y1 = pyro.sample("x1", Normal(x, torch.tensor(2**0.5)), obs=obs[0])
    y2 = pyro.sample("x2", Normal(x, torch.tensor(2**0.5)), obs=obs[1])
    return x

#########
# guide #
#########
first  = nn.Linear(2, 10)
second = nn.Linear(10, 20)
third  = nn.Linear(20, 10)
fourth = nn.Linear(10, 5)
fifth  = nn.Linear(5, 2)
relu = nn.ReLU()

# def guide(prior_mean, observations={'x1': 0, 'x2': 0}):
def guide(observations={'x1': 0, 'x2': 0}):
    pyro.module("first", first)
    pyro.module("second", second)
    pyro.module("third", third)
    pyro.module("fourth", fourth)
    pyro.module("fifth", fifth)

    obs = torch.tensor([float(observations['x1']),
                        float(observations['x2'])])
    # x1 = observations['x1']
    # x2 = observations['x2']
    x1 = obs[0]
    x2 = obs[1]
    # v = torch.cat((x1.view(1, 1), x2.view(1, 1)), 1)
    v = torch.cat((torch.Tensor.view(x1, [1, 1]),
                   torch.Tensor.view(x2, [1, 1])), 1)

    h1  = relu(first(v))
    h2  = relu(second(h1))
    h3  = relu(third(h2))
    h4  = relu(fourth(h3))
    out = fifth(h4)

    mean = out[0, 0]
    # std = out[0, 1].exp()
    std = torch.exp(out[0, 1])
    pyro.sample("z", Normal(mean, std))

optimiser = pyro.optim.Adam({'lr': 1e-3})
csis = pyro.infer.CSIS(model, guide, optimiser, num_inference_samples=50)

for step in range(n_steps):
    # During this iteration, model and guide are called. The input 'observations' of the guide keeps changing.
    if step % 100 == 0: print('step={}'.format(step))
    # csis.step(prior_mean)
    csis.step()

# posterior = csis.run(prior_mean, observations=data)
posterior = csis.run(observations=data)

marginal = pyro.infer.EmpiricalMarginal(posterior, "z")


import numpy as np
import scipy.stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Draw samples from empirical marginal for plotting
csis_samples = [marginal().detach() for _ in range(1000)]

# Calculate empirical marginal with importance sampling
is_posterior = (pyro.infer.Importance(model, num_samples=50)
                         # .run(prior_mean, observations=data)
                         .run(observations=data))
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
# plt.show()
plt.savefig('pyro_example_csis_simplified.jpg')
