# An example guide from "Bayesian Regression - Inference Algorithms (Part2)" in the pyro webpage
#
# - This guide does not capture correlation between latent variables.
#   There is another better guide in the same pyro webpage that is 
#   automatically generated and can capture such correlation.
# - Strictly speaking, this guide is wrong. Its support is not a subset 
#   of the model. But the probability of this problem happens is very very
#   small in some sense.
# - Original code:
# def guide(is_cont_africa, ruggedness, log_gdp):
#    a_loc = pyro.param('a_loc', torch.tensor(0.))
#    a_scale = pyro.param('a_scale', torch.tensor(1.),
#                         constraint=constraints.positive)
#    sigma_loc = pyro.param('sigma_loc', torch.tensor(1.),
#                           constraint=constraints.positive)
#    weights_loc = pyro.param('weights_loc', torch.rand(3))
#    weights_scale = pyro.param('weights_scale', torch.ones(3),
#                               constraint=constraints.positive)
#    a = pyro.sample("a", dist.Normal(a_loc, a_scale))
#    b_a = pyro.sample("bA", dist.Normal(weights_loc[0], weights_scale[0]))
#    b_r = pyro.sample("bR", dist.Normal(weights_loc[1], weights_scale[1]))
#    b_ar = pyro.sample("bAR", dist.Normal(weights_loc[2], weights_scale[2]))
#    sigma = pyro.sample("sigma", dist.Normal(sigma_loc, torch.tensor(0.05)))
#    mean = a + b_a * is_cont_africa + b_r * ruggedness + b_ar * is_cont_africa * ruggedness

# Non-deterministic initialisation artifically added to help the analysis
is_cont_africa = torch.rand([170])
ruggedness = torch.rand([170])
log_gdp = torch.rand([170])

# Actual code
a_loc = pyro.param('a_loc', torch.tensor(0.))
a_scale = pyro.param('a_scale', torch.tensor(1.), constraint=constraints.positive)
sigma_loc = pyro.param('sigma_loc', torch.tensor(1.), constraint=constraints.positive)
weights_loc = pyro.param('weights_loc', torch.rand(3))
weights_scale = pyro.param('weights_scale', torch.ones(3), constraint=constraints.positive)
a = pyro.sample("a", Normal(a_loc, a_scale))
b_a = pyro.sample("bA", Normal(weights_loc[0], weights_scale[0]))
b_r = pyro.sample("bR", Normal(weights_loc[1], weights_scale[1]))
b_ar = pyro.sample("bAR", Normal(weights_loc[2], weights_scale[2]))
sigma = pyro.sample("sigma", Normal(sigma_loc, torch.tensor(0.05)))
mean = a + b_a * is_cont_africa + b_r * ruggedness + b_ar * is_cont_africa * ruggedness
