# test_iplate_plate_ok: its variant. subsample_size in iplate=None.
p = torch.tensor(0.5)
inner_plate = pyro.plate("plate", 3, 2)
for i in pyro.plate("iplate", 3, None): 
     with inner_plate as ind: 
          # pyro.sample("x_{}".format(i), Bernoulli(p).expand_by([len(ind)]))
          pyro.sample("x_{}".format(i), Bernoulli(p).expand_by([2]))
