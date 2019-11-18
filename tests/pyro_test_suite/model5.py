# test_variable_clash_in_guide_error
p = torch.tensor(0.5)
pyro.sample("x", Bernoulli(p))
