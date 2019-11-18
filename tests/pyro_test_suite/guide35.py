# test_dim_allocation_error
p = torch.tensor(0.5, requires_grad=True)
with pyro.plate("plate_outer", 10, 5, dim=-2):
    x = pyro.sample("x", Bernoulli(p))
    # allocated dim is rightmost available, i.e. -1
    with pyro.plate("plate_inner_1", 11, 6):
        y = pyro.sample("y", Bernoulli(p))
        # throws an error as dim=-1 is already occupied
        with pyro.plate("plate_inner_2", 12, 7, dim=-1):
            pyro.sample("z", Bernoulli(p))

# # check shapes
# assert x.shape == (5, 1)
# assert y.shape == (5, 6)
