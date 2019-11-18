# test_plate_ok [subsample_size=None]
p = torch.tensor(0.5)
with pyro.plate("plate", 10, None) as ind:
#    pyro.sample("x", Bernoulli(p).expand_by([len(ind)])) - original code
    pyro.sample("x", Bernoulli(p).expand_by([10]))
