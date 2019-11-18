# test_plate_ok [subsample_size=None]
p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
with pyro.plate("plate", 10, None) as ind:
#    pyro.sample("x", Bernoulli(p).expand_by([len(ind)])) - orginal code
    pyro.sample("x", Bernoulli(p).expand_by([10]))
