# test_plate_iplate_ok 
p = torch.tensor(0.5)
with pyro.plate("plate", 3, 2) as ind: 
    for i in pyro.plate("iplate", 3, 2): 
        # pyro.sample("x_{}".format(i), dist.Bernoulli(p).expand_by([len(ind)]))
        pyro.sample("x_{}".format(i), Bernoulli(p).expand_by([2]))
