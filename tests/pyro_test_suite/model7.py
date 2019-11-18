# test_iplate_ok [subsample_size=2]
p = torch.tensor(0.5)
for i in pyro.plate("plate", 4, 2):
    pyro.sample("x_{}".format(i), Bernoulli(p))
