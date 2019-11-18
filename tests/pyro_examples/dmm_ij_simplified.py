"""
Source: /examples/pyro/dmm/dmm_simp_new.py

This code is a variant of dim_simplified.py. 
Everything is same, but vectorized plates are replaced with sequential plates. 
Index i is used in new for loops, so now the parameter z_t is named via double indices i and t.

...
for i in plate("z_minibatch", 20):
    ...
    for t in range(..):
        ...
        z_t = pyro.sample("z_{}_{}".format(i, t), ... )


"""

import argparse
import time
from os.path import exists

import numpy as np
import torch
import torch.nn as nn

#=== wy
# import polyphonic_data_loader as poly
import utils.polyphonic_data_loader_simp_new as poly
#=== wy
import pyro
from pyro.distributions import Normal, Bernoulli
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions import InverseAutoregressiveFlow, TransformedDistribution
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.nn import AutoRegressiveNN
from pyro.optim import ClippedAdam
#=== wy
# from util import get_logger
import logging

def get_logger(log_file):
    logging.basicConfig(level=logging.DEBUG, format='%(message)s', filename=log_file, filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    def log(s):
        logging.info(s)

    return log
#=== wy

#=== wy
# debug = True
debug = False
#=== wy

#=======#
# model #
#=======#

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

# the model p(x_{1:T} | z_{1:T}) p(z_{1:T}) 
def model(mini_batch, mini_batch_reversed, mini_batch_mask,
        mini_batch_seq_lengths, annealing_factor=1.0):
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

#=======#
# guide #
#=======#

# neural nets in combiner
c_lin_z_to_hidden = nn.Linear(100, 600)
c_lin_hidden_to_loc = nn.Linear(600, 100)
c_lin_hidden_to_scale = nn.Linear(600, 100)
c_tanh = nn.Tanh()
c_softplus = nn.Softplus()
rnn = nn.RNN(input_size=88, hidden_size=600, nonlinearity='relu',
        batch_first=True, bidirectional=False, num_layers=1,
        dropout=0.0)

z_q_0 = nn.Parameter(torch.zeros(100))
h_0 = nn.Parameter(torch.zeros(1, 1, 600))

# the guide q(z_{1:T} | x_{1:T}) (i.e. the variational distribution)
def guide(mini_batch, mini_batch_reversed, mini_batch_mask,
          mini_batch_seq_lengths, annealing_factor=1.0):
    pyro.module("c_lin_z_to_hidden", c_lin_z_to_hidden)
    pyro.module("c_lin_hidden_to_loc", c_lin_hidden_to_loc)
    pyro.module("c_lin_hidden_to_scale ", c_lin_hidden_to_scale)
    pyro.module("rnn", rnn)

    if debug:
        print("===== guide:S =====")
        print("mini_batch:\t type={}, shape={}".
              format(type(mini_batch), mini_batch.size()))
        print("mini_batch_reversed:\t type={}, shape={}".
              format(type(mini_batch_reversed), mini_batch_reversed.size()))
        print("mini_batch_mask:\t type={}, shape={}".
              format(type(mini_batch_mask), mini_batch_mask.size()))
        print("mini_batch_seq_lengths:\t type={}, shape={}".
              format(type(mini_batch_seq_lengths), mini_batch_seq_lengths.size()))
        print("===== guide:E =====")

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
    z_init = torch.Tensor.expand(z_q_0, [20, 100])

    # we enclose all the sample statements in the guide in a plate.
    # this marks that each datapoint is conditionally independent of the others.
    # with pyro.plate("z_minibatch", len(mini_batch)):
    for i in pyro.plate("z_minibatch", 20):
        # sample the latents z one time step at a time
        # # for t in range(1, T_max + 1):
        # for t in range(T_max):
        z_prev = z_init[i]
        for t in range(160):
            # h_rnn = rnn_output[:, t - 1, :]
            h_rnn = rnn_output[i, t, :]
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
                # z_t = pyro.sample("z_{}_{}".format(i, t),
                z_t = pyro.sample("__z_{}_{}".format(i, t), # wy: to enable use of zone domain
                                  Normal(z_loc, z_scale)
                                  # .mask(mini_batch_mask[:, t - 1:t])
                                  .mask(mini_batch_mask[i, t:t+1])
                                  .to_event(1))
                
            # the latent sampled at this time step will be conditioned upon in the next time step
            # so keep track of it
            z_prev = z_t

class DMM(nn.Module):
    """
    This PyTorch Module encapsulates the model as well as the
    variational distribution (the guide) for the Deep Markov Model
    """

    def __init__(self, input_dim=88, z_dim=100, emission_dim=100,
                 transition_dim=200, rnn_dim=600, num_layers=1, rnn_dropout_rate=0.0,
                 num_iafs=0, iaf_dim=50, use_cuda=False):
        super(DMM, self).__init__()
        # instantiate PyTorch modules used in the model and guide below
        # if we're using normalizing flows, instantiate those too
        self.iafs = [InverseAutoregressiveFlow(AutoRegressiveNN(z_dim, [iaf_dim])) for _ in range(num_iafs)]
        self.iafs_modules = nn.ModuleList(self.iafs)

        # define a (trainable) parameters z_0 and z_q_0 that help define the probability
        # distributions p(z_1) and q(z_1)
        # (since for t = 1 there are no previous latents to condition on)
        # define a (trainable) parameter for the initial hidden state of the rnn
        self.h_0 = nn.Parameter(torch.zeros(1, 1, rnn_dim))

        self.use_cuda = use_cuda
        # if on gpu cuda-ize all PyTorch (sub)modules
        if use_cuda:
            self.cuda()

# setup, training, and evaluation
def main(args):
    # setup logging
    log = get_logger(args.log)
    log(args)

    data = poly.load_data(poly.JSB_CHORALES)
    training_seq_lengths = data['train']['sequence_lengths']
    training_data_sequences = data['train']['sequences']
    test_seq_lengths = data['test']['sequence_lengths']
    test_data_sequences = data['test']['sequences']
    val_seq_lengths = data['valid']['sequence_lengths']
    val_data_sequences = data['valid']['sequences']
    #=== wy: to force batch_size be 20 during training, testing, and validation.
    if debug:
        print("===== after load_data =====")
        print("training_data_sequences:\t shape={}".
              format(training_data_sequences.size()))
        print("test_data_sequences:\t shape={}".
              format(test_data_sequences.size()))
        print("val_data_sequences:\t shape={}".
              format(val_data_sequences.size()))
        print("===== after load_data =====")
        
    d1_training = int(len(training_seq_lengths)/args.mini_batch_size)*args.mini_batch_size
    d1_test = args.mini_batch_size
    d1_val  = args.mini_batch_size
    training_seq_lengths = training_seq_lengths[:d1_training]
    training_data_sequences = training_data_sequences[:d1_training]
    test_seq_lengths = test_seq_lengths[:d1_test]
    test_data_sequences = test_data_sequences[:d1_test]
    val_seq_lengths = val_seq_lengths[:d1_val]
    val_data_sequences = val_data_sequences[:d1_val]
    #=== wy 
    N_train_data = len(training_seq_lengths)
    N_train_time_slices = float(torch.sum(training_seq_lengths))
    N_mini_batches = int(N_train_data / args.mini_batch_size +
                         int(N_train_data % args.mini_batch_size > 0))

    log("N_train_data: %d     avg. training seq. length: %.2f    N_mini_batches: %d" %
        (N_train_data, training_seq_lengths.float().mean(), N_mini_batches))

    # how often we do validation/test evaluation during training
    #=== wy
    # val_test_frequency = 50
    val_test_frequency = 1
    #=== wy
    # the number of samples we use to do the evaluation
    n_eval_samples = 1

    # package repeated copies of val/test data for faster evaluation
    # (i.e. set us up for vectorization)
    def rep(x):
        rep_shape = torch.Size([x.size(0) * n_eval_samples]) + x.size()[1:]
        repeat_dims = [1] * len(x.size())
        repeat_dims[0] = n_eval_samples
        return x.repeat(repeat_dims).reshape(n_eval_samples, -1).transpose(1, 0).reshape(rep_shape)

    # get the validation/test data ready for the dmm: pack into sequences, etc.
    val_seq_lengths = rep(val_seq_lengths)
    test_seq_lengths = rep(test_seq_lengths)
    val_batch, val_batch_reversed, val_batch_mask, val_seq_lengths = poly.get_mini_batch(
        torch.arange(n_eval_samples * val_data_sequences.shape[0]), rep(val_data_sequences),
        val_seq_lengths, cuda=args.cuda)
    test_batch, test_batch_reversed, test_batch_mask, test_seq_lengths = poly.get_mini_batch(
        torch.arange(n_eval_samples * test_data_sequences.shape[0]), rep(test_data_sequences),
        test_seq_lengths, cuda=args.cuda)

    # instantiate the dmm
    dmm = DMM(rnn_dropout_rate=args.rnn_dropout_rate, num_iafs=args.num_iafs,
              iaf_dim=args.iaf_dim, use_cuda=args.cuda)

    # setup optimizer
    adam_params = {"lr": args.learning_rate, "betas": (args.beta1, args.beta2),
                   "clip_norm": args.clip_norm, "lrd": args.lr_decay,
                   "weight_decay": args.weight_decay}
    adam = ClippedAdam(adam_params)

    # setup inference algorithm
    elbo = JitTrace_ELBO() if args.jit else Trace_ELBO()
    svi = SVI(model, guide, adam, loss=elbo)

    # now we're going to define some functions we need to form the main training loop

    # saves the model and optimizer states to disk
    def save_checkpoint():
        log("saving model to %s..." % args.save_model)
        torch.save(dmm.state_dict(), args.save_model)
        log("saving optimizer states to %s..." % args.save_opt)
        adam.save(args.save_opt)
        log("done saving model and optimizer checkpoints to disk.")

    # loads the model and optimizer states from disk
    def load_checkpoint():
        assert exists(args.load_opt) and exists(args.load_model), \
            "--load-model and/or --load-opt misspecified"
        log("loading model from %s..." % args.load_model)
        dmm.load_state_dict(torch.load(args.load_model))
        log("loading optimizer states from %s..." % args.load_opt)
        adam.load(args.load_opt)
        log("done loading model and optimizer states.")

    # prepare a mini-batch and take a gradient step to minimize -elbo
    def process_minibatch(epoch, which_mini_batch, shuffled_indices):
        if args.annealing_epochs > 0 and epoch < args.annealing_epochs:
            # compute the KL annealing factor approriate for the current mini-batch in the current epoch
            min_af = args.minimum_annealing_factor
            annealing_factor = min_af + (1.0 - min_af) * \
                (float(which_mini_batch + epoch * N_mini_batches + 1) /
                 float(args.annealing_epochs * N_mini_batches))
        else:
            # by default the KL annealing factor is unity
            annealing_factor = 1.0

        # compute which sequences in the training set we should grab
        mini_batch_start = (which_mini_batch * args.mini_batch_size)
        mini_batch_end = np.min([(which_mini_batch + 1) * args.mini_batch_size, N_train_data])
        mini_batch_indices = shuffled_indices[mini_batch_start:mini_batch_end]
        # grab a fully prepped mini-batch using the helper function in the data loader
        if debug: print("===== process_minibatch:S =====")
        mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths \
            = poly.get_mini_batch(mini_batch_indices, training_data_sequences,
                                  training_seq_lengths, cuda=args.cuda)
        if debug: print("===== process_minibatch:E =====")
        # do an actual gradient step
        loss = svi.step(mini_batch, mini_batch_reversed, mini_batch_mask,
                        mini_batch_seq_lengths, annealing_factor)
        # keep track of the training loss
        return loss

    # helper function for doing evaluation
    def do_evaluation():
        # put the RNN into evaluation mode (i.e. turn off drop-out if applicable)
        rnn.eval()

        # compute the validation and test loss n_samples many times
        val_nll = svi.evaluate_loss(val_batch, val_batch_reversed, val_batch_mask,
                                    val_seq_lengths) / torch.sum(val_seq_lengths)
        test_nll = svi.evaluate_loss(test_batch, test_batch_reversed, test_batch_mask,
                                     test_seq_lengths) / torch.sum(test_seq_lengths)

        # put the RNN back into training mode (i.e. turn on drop-out if applicable)
        rnn.train()
        return val_nll, test_nll

    # if checkpoint files provided, load model and optimizer states from disk before we start training
    if args.load_opt != '' and args.load_model != '':
        load_checkpoint()

    #################
    # TRAINING LOOP #
    #################
    times = [time.time()]
    for epoch in range(args.num_epochs):
        # if specified, save model and optimizer states to disk every checkpoint_freq epochs
        if args.checkpoint_freq > 0 and epoch > 0 and epoch % args.checkpoint_freq == 0:
            save_checkpoint()

        # accumulator for our estimate of the negative log likelihood (or rather -elbo) for this epoch
        epoch_nll = 0.0
        # prepare mini-batch subsampling indices for this epoch
        shuffled_indices = torch.randperm(N_train_data)

        # process each mini-batch; this is where we take gradient steps
        for which_mini_batch in range(N_mini_batches):
            epoch_nll += process_minibatch(epoch, which_mini_batch, shuffled_indices)

        # report training diagnostics
        times.append(time.time())
        epoch_time = times[-1] - times[-2]
        log("[training epoch %04d]  %.4f \t\t\t\t(dt = %.3f sec)" %
            (epoch, epoch_nll / N_train_time_slices, epoch_time))

        # do evaluation on test and validation data and report results
        if val_test_frequency > 0 and epoch > 0 and epoch % val_test_frequency == 0:
            val_nll, test_nll = do_evaluation()
            log("[val/test epoch %04d]  %.4f  %.4f" % (epoch, val_nll, test_nll))


# parse command-line arguments and execute the main method
if __name__ == '__main__':
    assert pyro.__version__.startswith('0.3.1')

    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', type=int, default=5000)
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.0003)
    parser.add_argument('-b1', '--beta1', type=float, default=0.96)
    parser.add_argument('-b2', '--beta2', type=float, default=0.999)
    parser.add_argument('-cn', '--clip-norm', type=float, default=20.0)
    parser.add_argument('-lrd', '--lr-decay', type=float, default=0.99996)
    parser.add_argument('-wd', '--weight-decay', type=float, default=2.0)
    parser.add_argument('-mbs', '--mini-batch-size', type=int, default=20)
    parser.add_argument('-ae', '--annealing-epochs', type=int, default=1000)
    parser.add_argument('-maf', '--minimum-annealing-factor', type=float, default=0.1)
    parser.add_argument('-rdr', '--rnn-dropout-rate', type=float, default=0.1)
    parser.add_argument('-iafs', '--num-iafs', type=int, default=0)
    parser.add_argument('-id', '--iaf-dim', type=int, default=100)
    parser.add_argument('-cf', '--checkpoint-freq', type=int, default=0)
    parser.add_argument('-lopt', '--load-opt', type=str, default='')
    parser.add_argument('-lmod', '--load-model', type=str, default='')
    parser.add_argument('-sopt', '--save-opt', type=str, default='')
    parser.add_argument('-smod', '--save-model', type=str, default='')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--jit', action='store_true')
    parser.add_argument('-l', '--log', type=str, default='dmm.log')
    args = parser.parse_args()

    main(args)
