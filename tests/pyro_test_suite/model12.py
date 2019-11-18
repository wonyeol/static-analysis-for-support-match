# test_iplate_iplate_ok [subsample_size=None]
p = torch.tensor(0.5)
outer_iplate = pyro.plate("plate_0", 3, None)
inner_iplate = pyro.plate("plate_1", 3, None)
for i in outer_iplate:
    for j in inner_iplate:
        # pyro.sample("x_{}_{}".format(i, j), Bernoulli(p))
        pyro.sample("__x_{}_{}".format(i, j), Bernoulli(p)) # to enable equality checking
