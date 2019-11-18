# test_iplate_in_guide_not_model_error [subsample_size=5]
p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
for i in pyro.plate("plate", 10, 5):
    pass
pyro.sample("x", Bernoulli(p))
