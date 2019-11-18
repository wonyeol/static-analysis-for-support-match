# test_vectorized_num_particles
data = torch.ones(1000, 2)

with pyro.plate("components", 2):
    p = pyro.sample("p", Beta(torch.tensor(1.1), torch.tensor(1.1)))
    # assert p.shape == torch.Size((10, 1, 2))
    # with pyro.plate("data", data.shape[0]):
    with pyro.plate("data", 1000):
        pyro.sample("obs", Bernoulli(p), obs=data)
