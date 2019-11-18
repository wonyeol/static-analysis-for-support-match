# test_iplate_ok [subsample_size=None]
p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
for i in pyro.plate("plate", 4, None):
    pyro.sample("x_{}".format(i), Bernoulli(p))
