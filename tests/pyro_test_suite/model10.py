# test_plate_ok [subsample_size=5]
p = torch.tensor(0.5)
with pyro.plate("plate", 10, 5) as ind:
#    pyro.sample("x", Bernoulli(p).expand_by([len(ind)])) - original code
    pyro.sample("x", Bernoulli(p).expand_by([5]))
