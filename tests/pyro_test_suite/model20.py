# test_plate_broadcast_error
p = torch.tensor(0.5, requires_grad=True)
with pyro.plate("plate", 10, 5):
    pyro.sample("x", Bernoulli(p).expand_by([2]))
