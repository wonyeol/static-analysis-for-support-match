# test_enum_discrete_misuse_warning
p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
pyro.sample("x", Bernoulli(p), infer={"enumerate": "parallel"})
