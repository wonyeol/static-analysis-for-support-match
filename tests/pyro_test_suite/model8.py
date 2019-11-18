# test_iplate_variable_clash_error
p = torch.tensor(0.5)
for i in pyro.plate("plate", 2):
    pyro.sample("x", Bernoulli(p))
