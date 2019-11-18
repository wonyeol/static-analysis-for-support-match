# test_iplate_ok [subsample_size=None]
p = torch.tensor(0.5)
for i in pyro.plate("plate", 4, None):
    pyro.sample("x_{}".format(i), Bernoulli(p))
