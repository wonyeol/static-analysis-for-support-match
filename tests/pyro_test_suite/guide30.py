# test_three_indep_plate_at_different_depths_ok
p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
inner_plate = pyro.plate("plate2", 10, 5)
for i in pyro.plate("plate0", 2):
    # pyro.sample("x_%d" % i, Bernoulli(p))
    pyro.sample("x_{}".format(i), Bernoulli(p))
    if i == 0:
        for j in pyro.plate("plate1", 2):
            with inner_plate as ind:
                # pyro.sample("y_%d" % j, Bernoulli(p).expand_by([len(ind)]))
                pyro.sample("y_{}".format(j), Bernoulli(p).expand_by([5]))
    elif i == 1:
        with inner_plate as ind:
            # pyro.sample("z", Bernoulli(p).expand_by([len(ind)]))
            pyro.sample("z", Bernoulli(p).expand_by([5]))
