# test_variable_clash_in_model_error
p = torch.tensor(0.5)
pyro.sample("x", Bernoulli(p))
pyro.sample("x", Bernoulli(p))  # Should error here.
