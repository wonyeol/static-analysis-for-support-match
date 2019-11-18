# http://pyro.ai/examples/ss-vae.html
# torch.distributions.OneHotCategorical
# self.encoder_y.forward, self.encoder_z.forward



xs = torch.rand([200,784])
ys = torch.rand([200,10])

softplus = nn.Softplus()
softmax = nn.Softmax(dim=-1)

encoder_y_fst = nn.Linear(784, 500)
encoder_y_fst.weight.data.normal_(0, 0.001)
encoder_y_fst.bias.data.normal_(0, 0.001)
encoder_y_snd = nn.Linear(500, 10)

encoder_z_fst = nn.Linear(794, 500)
encoder_z_fst.weight.data.normal_(0, 0.001)
encoder_z_fst.bias.data.normal_(0, 0.001)
encoder_z_out1 = nn.Linear(500, 50)
encoder_z_out2 = nn.Linear(500, 50)

pyro.module("encoder_y_fst", encoder_y_fst)
pyro.module("encoder_y_snd", encoder_y_snd)
pyro.module("encoder_z_fst", encoder_z_fst)
pyro.module("encoder_z_out1", encoder_z_out1)
pyro.module("encoder_z_out2", encoder_z_out2)

# inform Pyro that the variables in the batch of xs, ys are conditionally independent
with pyro.plate("data"):
    # if the class label (the digit) is not supervised, sample
    # (and score) the digit with the variational distribution
    # q(y|x) = categorical(alpha(x))
    if ys is None:
        hidden = softplus(encoder_y_fst(xs))
        alpha = softmax(encoder_y_snd(hidden))             
        ys = pyro.sample("y", OneHotCategorical(alpha))

    # sample (and score) the latent handwriting-style with the variational
    # distribution q(z|x,y) = normal(loc(x,y),scale(x,y))
    hidden_z = softplus(encoder_z_fst(torch.cat([xs, ys], -1))) 
    loc = encoder_z_out1(hidden_z) 
    scale = torch.exp(encoder_z_out2(hidden_z))
    pyro.sample("z", Normal(loc, scale).to_event(1))

