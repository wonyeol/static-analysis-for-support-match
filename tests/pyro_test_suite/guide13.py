# test_iplate_iplate_ok [subsample_size=2]
p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
outer_iplate = pyro.plate("plate_0", 3, 2)
inner_iplate = pyro.plate("plate_1", 3, 2)
for i in outer_iplate:
    for j in inner_iplate:
        # pyro.sample("x_{}_{}".format(i, j), Bernoulli(p))
        pyro.sample("__x_{}_{}".format(i, j), Bernoulli(p)) # to enable equality checking
