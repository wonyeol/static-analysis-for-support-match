# test_plate_reuse_ok
p = torch.tensor(0.5, requires_grad=True)
plate_outer = pyro.plate("plate_outer", 10, 5, dim=-1)
plate_inner = pyro.plate("plate_inner", 11, 6, dim=-2)
with plate_outer as ind_outer:
    # pyro.sample("x", Bernoulli(p).expand_by([len(ind_outer)]))
    pyro.sample("x", Bernoulli(p).expand_by([5]))
with plate_inner as ind_inner:
    # pyro.sample("y", Bernoulli(p).expand_by([len(ind_inner), 1]))
    pyro.sample("y", Bernoulli(p).expand_by([6, 1]))
with plate_outer as ind_outer, plate_inner as ind_inner:
    # pyro.sample("z", Bernoulli(p).expand_by([len(ind_inner), len(ind_outer)]))
    pyro.sample("z", Bernoulli(p).expand_by([6, 5]))
