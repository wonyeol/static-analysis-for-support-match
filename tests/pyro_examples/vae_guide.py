# Variational autoencoder example from the pyro webpage.
#
# 1) We made multiple changes so that our front end does not cause an issue.
#
# - z_dim is replaced by 50
# - hidden_dim is replaced by 400
# - x.shape[0] is replaced by its value 256 in the code.
# - Classes are removed as much as possible.
# - We removed the part that declares the name of a procedure.
# - If the removed part is restored, the code still trains a vae on MNIST. 

g_fc1 = nn.Linear(784, 400)
g_fc21 = nn.Linear(400, 50)
g_fc22 = nn.Linear(400, 50)
softplus = nn.Softplus()

x = torch.reshape(x, [256, 784]) 

pyro.module("encoder_fc1", g_fc1)
pyro.module("encoder_fc21", g_fc21)
pyro.module("encoder_fc22", g_fc22)
with pyro.plate("data", 256): 
    hidden = softplus(g_fc1(x)) 
    z_loc = g_fc21(hidden) 
    z_scale = torch.exp(g_fc22(hidden)) 
    pyro.sample("latent", Normal(z_loc, z_scale).to_event(1))
