# test_plate_shape_broadcasting
data = torch.ones(1000, 2)

with pyro.plate("num_particles", 10, dim=-3):
    with pyro.plate("components", 2, dim=-1):
        pyro.sample("p", Beta(torch.tensor(1.1), torch.tensor(1.1)))
