# test_iplate_in_model_not_guide_ok [subsample_size=5]
p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
pyro.sample("x", Bernoulli(p))
