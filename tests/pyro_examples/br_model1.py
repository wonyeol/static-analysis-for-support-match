# An example model from "Bayesian Regression - Inference Algorithms (Part2)" in the pyro webpage
# 
# - We replaced iarange by plate.
# - Original code:
# def model(is_cont_africa, ruggedness, log_gdp):
#     a = pyro.sample("a", dist.Normal(8., 1000.))
#     b_a = pyro.sample("bA", dist.Normal(0., 1.))
#     b_r = pyro.sample("bR", dist.Normal(0., 1.))
#     b_ar = pyro.sample("bAR", dist.Normal(0., 1.))
#     sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
#     mean = a + b_a * is_cont_africa + b_r * ruggedness + b_ar * is_cont_africa * ruggedness 
#     with pyro.plate("data", len(ruggedness)):
#         pyro.sample("obs", dist.Normal(mean, sigma), obs=log_gdp)

# Non-deterministic initialisation artifically added to help the analysis
is_cont_africa = torch.rand([170])
ruggedness = torch.rand([170])
log_gdp = torch.rand([170])

# Non-deterministic initialisation
a = pyro.sample("a", Normal(8., 1000.))
b_a = pyro.sample("bA", Normal(0., 1.))
b_r = pyro.sample("bR", Normal(0., 1.))
b_ar = pyro.sample("bAR", Normal(0., 1.))
sigma = pyro.sample("sigma", Uniform(0.1, 10.))
mean = a + b_a * is_cont_africa + b_r * ruggedness + b_ar * is_cont_africa * ruggedness 
with pyro.plate("data", 170):
    pyro.sample("obs", Normal(mean, sigma), obs=log_gdp)
