# test_iplate_ok [subsample_size = 2]
p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
for i in pyro.plate("plate", 4, 2):
    pyro.sample("x_{}".format(i), Bernoulli(p))
