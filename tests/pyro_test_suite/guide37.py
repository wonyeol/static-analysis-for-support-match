# test_mean_field_ok
loc = pyro.param("loc", torch.tensor(0.))
x = pyro.sample("x", Normal(loc, 1.))
pyro.sample("y", Normal(x, 1.))
