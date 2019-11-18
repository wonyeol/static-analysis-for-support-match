# test_mean_field_warn
x = pyro.sample("x", Normal(0., 1.))
pyro.sample("y", Normal(x, 1.))
