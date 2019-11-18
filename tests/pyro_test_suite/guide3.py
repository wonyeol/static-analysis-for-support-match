# test_model_guide_dim_mismatch_error
loc = pyro.param("loc", torch.zeros(2, 1, requires_grad=True))
scale = pyro.param("scale", torch.ones(2, 1, requires_grad=True))
pyro.sample("x", Normal(loc, scale).to_event(2))
