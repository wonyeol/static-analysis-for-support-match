# test_variable_clash_in_guide_error
p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
pyro.sample("x", Bernoulli(p))
pyro.sample("x", Bernoulli(p))  # Should error here.
