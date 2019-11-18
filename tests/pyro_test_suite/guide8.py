# test_iplate_variable_clash_error
p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
for i in pyro.plate("plate", 2):
    pyro.sample("x", Bernoulli(p))
