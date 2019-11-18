# test_iplate_in_model_not_guide_ok [subsample_size=None]
p = torch.tensor(0.5)
for i in pyro.plate("plate", 10, None):
    pass
pyro.sample("x", Bernoulli(p))
