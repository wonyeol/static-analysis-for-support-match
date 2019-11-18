# test_iplate_plate_ok
p = pyro.param("p", torch.tensor(0.5, requires_grad=True)) 
inner_plate = pyro.plate("plate", 3, 2) 
for i in pyro.plate("iplate", 3, 2): 
    with inner_plate as ind: 
        # pyro.sample("x_{}".format(i), Bernoulli(p).expand_by([len(ind)]))
        pyro.sample("x_{}".format(i), Bernoulli(p).expand_by([2]))
