# test_dim_allocation_ok
p = torch.tensor(0.5, requires_grad=True)
with pyro.plate("plate_outer", 10, 5, dim=-3):
    x = pyro.sample("x", Bernoulli(p))
    with pyro.plate("plate_inner_1", 11, 6):
        y = pyro.sample("y", Bernoulli(p))
        # allocated dim is rightmost available, i.e. -1
        with pyro.plate("plate_inner_2", 12, 7):
            z = pyro.sample("z", Bernoulli(p))
            # allocated dim is next rightmost available, i.e. -2
            # since dim -3 is already allocated, use dim=-4
            with pyro.plate("plate_inner_3", 13, 8):
                q = pyro.sample("q", Bernoulli(p))

# # check shapes
# assert x.shape == (5, 1, 1)
# assert y.shape == (5, 1, 6)
# assert z.shape == (5, 7, 6)
# assert q.shape == (8, 5, 7, 6)
