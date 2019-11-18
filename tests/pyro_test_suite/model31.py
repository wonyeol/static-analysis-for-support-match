# test_plate_wrong_size_error
p = torch.tensor(0.5)
with pyro.plate("plate", 10, 5) as ind:
    # pyro.sample("x", Bernoulli(p).expand_by([1 + len(ind)]))
    pyro.sample("x", Bernoulli(p).expand_by([6]))
