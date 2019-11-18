# test_model_guide_dim_mismatch_error 
loc = torch.zeros(2) 
scale = torch.ones(2) 
pyro.sample("x", Normal(loc, scale).to_event(1))
