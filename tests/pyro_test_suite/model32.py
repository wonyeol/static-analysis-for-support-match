# test_enum_discrete_misuse_warning
p = torch.tensor(0.5)
pyro.sample("x", Bernoulli(p))
