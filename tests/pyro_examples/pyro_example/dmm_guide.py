#=======#
# guide #
#=======#

# neural nets in combiner
c_lin_z_to_hidden = nn.Linear(100, 600)
c_lin_hidden_to_loc = nn.Linear(600, 100)
c_lin_hidden_to_scale = nn.Linear(600, 100)
c_tanh = nn.Tanh()
c_softplus = nn.Softplus()
# rnn = nn.RNN(input_size=88, hidden_size=600, nonlinearity='relu',
#              batch_first=True, bidirectional=False, num_layers=1,
#              dropout=0.0)
rnn = nn.RNN(88,    # input_size,
             600,   # hidden_size
             1,     # num_layers
             True,  # bias
             True,  # batch_first
             0.0,   # dropout
             False, # bidirectional
             nonlinearity='relu')

z_q_0 = nn.Parameter(torch.zeros(100))
h_0 = nn.Parameter(torch.zeros(1, 1, 600))

# the guide q(z_{1:T} | x_{1:T}) (i.e. the variational distribution)
pyro.module("c_lin_z_to_hidden", c_lin_z_to_hidden)
pyro.module("c_lin_hidden_to_loc", c_lin_hidden_to_loc)
pyro.module("c_lin_hidden_to_scale ", c_lin_hidden_to_scale)
pyro.module("rnn", rnn)

#===== init tensor shape 
mini_batch = torch.reshape(mini_batch, [20, 160, 88])
mini_batch_reversed = torch.reshape(mini_batch_reversed, [20, 160, 88])
mini_batch_mask = torch.reshape(mini_batch_mask, [20, 160])
mini_batch_seq_lengths = torch.reshape(mini_batch_seq_lengths, [20])
#===== init tensor shape 

# this is the number of time steps we need to process in the mini-batch
# # T_max = mini_batch.size(1)
# T_max = 160
# register all PyTorch (sub)modules with pyro
pyro.module("rnn", rnn)
# if on gpu we need the fully broadcast view of the rnn initial state
# to be in contiguous gpu memory
# h_0_contig = h_0.expand(1, mini_batch.size(0), rnn.hidden_size).contiguous()
h_0_contig = torch.Tensor.expand(h_0, [1, 20, 600])
# push the observed x's through the rnn;
# rnn_output contains the hidden state at each time step
rnn_output, _ = rnn(mini_batch_reversed, h_0_contig)

# reverse the time-ordering in the hidden state and un-pack it
# rnn_output = poly.pad_and_reverse(rnn_output, mini_batch_seq_lengths)
#===== pad_and_reverse
#=== wy: disallow using PackedSequence in the whole program
# rnn_output, _ = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)
#=== wy
# # reversed_output = reverse_sequences(rnn_output, seq_lengths)
# rnn_output = reverse_sequences(rnn_output, mini_batch_seq_lengths)
#======= reverse_sequences
# shape = [20, 160, 600]
_mini_batch = rnn_output
_seq_lengths = mini_batch_seq_lengths

# reversed_mini_batch = _mini_batch.new_zeros(_mini_batch.size())
reversed_mini_batch = torch.zeros(20, 160, 600)

# for b in range(_mini_batch.size(0)):
for b in range(20):
    T = _seq_lengths[b]
    # time_slice = torch.arange(T - 1, -1, -1, device=_mini_batch.device)
    time_slice = torch.arange(T - 1, -1, -1)
    reversed_sequence = torch.index_select(_mini_batch[b, :, :], 0, time_slice)
    reversed_mini_batch[b, 0:T, :] = reversed_sequence

# return reversed_mini_batch
rnn_output = reversed_mini_batch
#======= reverse_sequences
#===== pad_and_reverse

# set z_prev = z_q_0 to setup the recursive conditioning in q(z_t |...)
# z_prev = z_q_0.expand(mini_batch.size(0), z_q_0.size(0))
z_prev = torch.Tensor.expand(z_q_0, [20, 100])

# we enclose all the sample statements in the guide in a plate.
# this marks that each datapoint is conditionally independent of the others.
# with pyro.plate("z_minibatch", len(mini_batch)):
with pyro.plate("z_minibatch", 20):
    # sample the latents z one time step at a time
    # # for t in range(1, T_max + 1):
    # for t in range(T_max):
    for t in range(160):
        # h_rnn = rnn_output[:, t - 1, :]
        h_rnn = rnn_output[:, t, :]
        h_combined = 0.5 * (c_tanh(c_lin_z_to_hidden(z_prev)) + h_rnn)
        loc = c_lin_hidden_to_loc(h_combined)
        scale = c_softplus(c_lin_hidden_to_scale(h_combined))
        z_loc = loc
        z_scale = scale
        z_dist = Normal(z_loc, z_scale)
        # assert z_dist.event_shape == ()
        # assert z_dist.batch_shape == (len(mini_batch), z_q_0.size(0))

        # sample z_t from the distribution z_dist
        with pyro.poutine.scale(scale=annealing_factor):
            # when no normalizing flow used, ".to_event(1)" indicates latent dimensions are independent
            # z_t = pyro.sample("z_%d" % t,
            z_t = pyro.sample("z_{}".format(t),
                              Normal(z_loc, z_scale)
                              # .mask(mini_batch_mask[:, t - 1:t])
                              .mask(mini_batch_mask[:, t:t+1])
                              .to_event(1))

        # the latent sampled at this time step will be conditioned upon in the next time step
        # so keep track of it
        z_prev = z_t
