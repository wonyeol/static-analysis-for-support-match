# test_iplate_in_guide_not_model_error [subsample_size=None]
p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
for i in pyro.plate("plate", 10, None):
    pass
pyro.sample("x", Bernoulli(p))
