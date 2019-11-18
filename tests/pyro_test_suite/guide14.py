# test_iplate_iplate_swap_ok [subsample_size=None]
p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
outer_iplate = pyro.plate("plate_0", 3, None)
inner_iplate = pyro.plate("plate_1", 3, None)
for j in inner_iplate:
    for i in outer_iplate:
        # pyro.sample("x_{}_{}".format(i, j), Bernoulli(p))
        pyro.sample("__x_{}_{}".format(i, j), Bernoulli(p)) # to enable equality checking
