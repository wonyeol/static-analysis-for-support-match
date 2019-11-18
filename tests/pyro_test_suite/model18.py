# test_iplate_in_guide_not_model_error [subsample_size=None]
p = torch.tensor(0.5)
pyro.sample("x", Bernoulli(p))
