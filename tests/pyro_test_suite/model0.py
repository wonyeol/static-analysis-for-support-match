# test_nonempty_model_empty_guide_ok
loc = torch.tensor([0.0, 0.0])
scale = torch.tensor([1.0, 1.0])
pyro.sample("x", Normal(loc, scale).to_event(1), obs=loc)
