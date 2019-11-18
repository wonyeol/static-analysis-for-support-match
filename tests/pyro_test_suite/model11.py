# test_plate_no_size_ok 
p = torch.tensor(0.5)
with pyro.plate("plate"):
    pyro.sample("x", Bernoulli(p).expand_by([10]))
