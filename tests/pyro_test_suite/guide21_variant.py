# test_plate_iplate_ok: its variant. subsample_size in iplate=None.
p = pyro.param("p", torch.tensor(0.5, requires_grad=True)) 
with pyro.plate("plate", 3, 2) as ind: 
    for i in pyro.plate("iplate", 3, None): 
        # pyro.sample("x_{}".format(i), dist.Bernoulli(p).expand_by([len(ind)]))
        pyro.sample("x_{}".format(i), Bernoulli(p).expand_by([2]))
