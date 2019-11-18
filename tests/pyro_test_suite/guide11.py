# test_plate_no_size_ok
p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
with pyro.plate("plate"):
    pyro.sample("x", Bernoulli(p).expand_by([10]))
