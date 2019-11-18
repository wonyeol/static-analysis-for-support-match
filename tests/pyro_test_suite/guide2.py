# test_variable_clash_in_model_error
p = pyro.param("p", torch.tensor(0.5, requires_grad=True)) 
pyro.sample("x", Bernoulli(p))
