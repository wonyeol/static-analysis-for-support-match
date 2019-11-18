# test_iplate_iplate_swap_ok [subsample_size=None]: its variant. added one more loop.
p = torch.tensor(0.5)
iplate_0 = pyro.plate("plate_0", 24, None)
iplate_1 = pyro.plate("plate_1", 12, None)
iplate_2 = pyro.plate("plate_2",  8, None)
for i in iplate_0:
    for j in iplate_1:
        for k in iplate_2:
            pyro.sample("__x_{}_{}_{}".format(i, j, k), Bernoulli(p))
