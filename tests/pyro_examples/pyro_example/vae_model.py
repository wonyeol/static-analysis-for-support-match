# Variational autoencoder example from the pyro webpage.
#
# 1) We made multiple changes so that our front end does not cause an issue.
#
# - x.shape[0] is replaced by its value 256 in the code.
# - z_dim is replaced by 50
# - hidden_dim is replaced by 400
# - Classes are removed as much as possible.
# - We removed the part that declares the name of a procedure.
# - If the removed part is restored, the code still trains a vae on MNIST. 
#   The code is available at pyro_example_vae_simplified.py. Its behaviour 
#   is almost identical to the original except that the last batch is 
#   ignored during trainig and testing.

m_fc1 = nn.Linear(50, 400)
m_fc21 = nn.Linear(400, 784)
softplus = nn.Softplus()
sigmoid = nn.Sigmoid()

x = torch.reshape(x, [256, 784])

pyro.module("decoder_fc1", m_fc1)
pyro.module("decoder_fc21", m_fc21)
with pyro.plate("data", 256): 
    z_loc = torch.zeros([256, 50]) 
    z_scale = torch.ones([256, 50]) 
    z = pyro.sample("latent", Normal(z_loc, z_scale).to_event(1)) 
    hidden = softplus(m_fc1(z)) 
    loc_img = sigmoid(m_fc21(hidden)) 
    pyro.sample("obs", Bernoulli(loc_img).to_event(1), obs=x)
