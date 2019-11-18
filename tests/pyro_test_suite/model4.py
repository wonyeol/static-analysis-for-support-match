# test_model_guide_shape_mismatch_error 
loc = torch.zeros(1, 2)
scale = torch.ones(1, 2)
pyro.sample("x", Normal(loc, scale).to_event(2))
