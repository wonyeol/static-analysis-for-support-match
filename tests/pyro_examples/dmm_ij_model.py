# neural nets in emitter
e_lin_z_to_hidden = nn.Linear(100, 100)
e_lin_hidden_to_hidden = nn.Linear(100, 100)
e_lin_hidden_to_input = nn.Linear(100, 88)
e_relu = nn.ReLU()

# neural nets in gated transition
t_lin_gate_z_to_hidden = nn.Linear(100, 200)
t_lin_gate_hidden_to_z = nn.Linear(200, 100)
t_lin_proposed_mean_z_to_hidden = nn.Linear(100, 200)
t_lin_proposed_mean_hidden_to_z = nn.Linear(200, 100)
t_lin_sig = nn.Linear(100, 100)
t_lin_z_to_loc = nn.Linear(100, 100)
t_lin_z_to_loc.weight.data = torch.eye(100)
t_lin_z_to_loc.bias.data = torch.zeros(100)
t_relu = nn.ReLU()
t_softplus = nn.Softplus()

z_0 = nn.Parameter(torch.zeros(100))     

pyro.module("e_lin_z_to_hidden", e_lin_z_to_hidden)
pyro.module("e_lin_hidden_to_hidden", e_lin_hidden_to_hidden)
pyro.module("e_lin_hidden_to_input", e_lin_hidden_to_input)
pyro.module("t_lin_gate_z_to_hidden", t_lin_gate_z_to_hidden)
pyro.module("t_lin_gate_hidden_to_z", t_lin_gate_hidden_to_z)
pyro.module("t_lin_proposed_mean_z_to_hidden", t_lin_proposed_mean_z_to_hidden)
pyro.module("t_lin_proposed_mean_hidden_to_z", t_lin_proposed_mean_hidden_to_z)
pyro.module("t_lin_sig", t_lin_sig)
pyro.module("t_lin_z_to_loc", t_lin_z_to_loc)

#===== init tensor shape 
mini_batch = torch.reshape(mini_batch, [20, 160, 88])
mini_batch_reversed = torch.reshape(mini_batch_reversed, [20, 160, 88])
mini_batch_mask = torch.reshape(mini_batch_mask, [20, 160])
mini_batch_seq_lengths = torch.reshape(mini_batch_seq_lengths, [20])
#===== init tensor shape 

# this is the number of time steps we need to process in the mini-batch
# # T_max = mini_batch.size(1)
# T_max = 160
# set z_prev = z_0 to setup the recursive conditioning in p(z_t | z_{t-1})
# z_prev = z_0.expand(mini_batch.size(0), z_0.size(0))
z_init = torch.Tensor.expand(z_0, [20, 100])

# we enclose all the sample statements in the model in a plate.
# this marks that each datapoint is conditionally independent of the others
# with pyro.plate("z_minibatch", len(mini_batch)): #len(mini_batch)= 20
for i in pyro.plate("z_minibatch", 20):
    # sample the latents z and observed x's one time step at a time
    # # for t in range(1, T_max + 1):
    # for t in range(T_max):
    z_prev = z_init[i]
    for t in range(160):
        # the next chunk of code samples z_t ~ p(z_t | z_{t-1})
        # note that (both here and elsewhere) we use poutine.scale to take care
        # of KL annealing. we use the mask() method to deal with raggedness
        # in the observed data (i.e. different sequences in the mini-batch
        # have different lengths)            
        _gate = t_relu(t_lin_gate_z_to_hidden(z_prev))
        gate = torch.sigmoid(t_lin_gate_hidden_to_z(_gate))
        _proposed_mean = t_relu(t_lin_proposed_mean_z_to_hidden(z_prev))
        proposed_mean = t_lin_proposed_mean_hidden_to_z(_proposed_mean)
        loc = (1 - gate) * t_lin_z_to_loc(z_prev) + gate * proposed_mean
        scale = t_softplus(t_lin_sig(t_relu(proposed_mean)))

        # first compute the parameters of the diagonal gaussian distribution p(z_t | z_{t-1})
        z_loc = loc
        z_scale = scale
        # then sample z_t according to dist.Normal(z_loc, z_scale)
        # note that we use the reshape method so that the univariate Normal distribution
        # is treated as a multivariate Normal distribution with a diagonal covariance.
        with poutine.scale(scale=annealing_factor):
            # z_t = pyro.sample("z_{}_{}".format(i, t),
            z_t = pyro.sample("__z_{}_{}".format(i, t), # wy: to enable use of zone domain
                              Normal(z_loc, z_scale)
                              # .mask(mini_batch_mask[:, t - 1:t])
                              .mask(mini_batch_mask[i, t:t+1])
                              .to_event(1))

        # compute the probabilities that parameterize the bernoulli likelihood
        h1 = e_relu(e_lin_z_to_hidden(z_t))
        h2 = e_relu(e_lin_hidden_to_hidden(h1))
        ps = torch.sigmoid(e_lin_hidden_to_input(h2))
        emission_probs_t = ps
        # the next statement instructs pyro to observe x_t according to the
        # bernoulli distribution p(x_t|z_t)
        pyro.sample("obs_x_{}_{}".format(i, t),
                    Bernoulli(emission_probs_t)
                    # .mask(mini_batch_mask[:, t - 1:t])
                    .mask(mini_batch_mask[i, t:t+1])
                    .to_event(1),
                    # obs=mini_batch[:, t - 1, :])
                    obs=mini_batch[i, t, :])
        # the latent sampled at this time step will be conditioned upon
        # in the next time step so keep track of it
        z_prev = z_t

