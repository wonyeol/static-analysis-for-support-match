# test_plate_shape_broadcasting
data = torch.ones(1000, 2)

with pyro.plate("num_particles", 10, dim=-3):
    with pyro.plate("components", 2, dim=-1):
        p = pyro.sample("p", Beta(torch.tensor(1.1), torch.tensor(1.1)))
        # assert p.shape == torch.Size((10, 1, 2))
    # with pyro.plate("data", data.shape[0], dim=-2):
    with pyro.plate("data", 1000, dim=-2):
        pyro.sample("obs", Bernoulli(p), obs=data)
