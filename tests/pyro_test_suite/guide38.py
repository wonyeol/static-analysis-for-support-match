# test_mean_field_warn
loc = pyro.param("loc", torch.tensor(0.))
y = pyro.sample("y", Normal(loc, 1.))
pyro.sample("x", Normal(y, 1.))
