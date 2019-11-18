# test_plate_wrong_size_error
p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
with pyro.plate("plate", 10, 5) as ind:
    # pyro.sample("x", Bernoulli(p).expand_by([1 + len(ind)]))
    pyro.sample("x", Bernoulli(p).expand_by([6]))
