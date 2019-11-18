# test_nonnested_plate_plate_ok
p = torch.tensor(0.5, requires_grad=True)
with pyro.plate("plate_0", 10, 5) as ind1:
    # pyro.sample("x0", Bernoulli(p).expand_by([len(ind1)]))
    pyro.sample("x0", Bernoulli(p).expand_by([5]))
with pyro.plate("plate_1", 11, 6) as ind2:
    # pyro.sample("x1", Bernoulli(p).expand_by([len(ind2)]))
    pyro.sample("x1", Bernoulli(p).expand_by([6]))
