# test_iplate_in_guide_not_model_error [subsample_size=5]
p = torch.tensor(0.5)
pyro.sample("x", Bernoulli(p))
