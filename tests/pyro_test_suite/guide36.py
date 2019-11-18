# test_vectorized_num_particles
data = torch.ones(1000, 2)

with pyro.plate("components", 2):
    pyro.sample("p", Beta(torch.tensor(1.1), torch.tensor(1.1)))
