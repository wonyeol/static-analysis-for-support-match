from __future__ import absolute_import, division, print_function

import argparse
import logging

import torch
import torch.nn as nn
from torch.distributions import constraints

import utils.polyphonic_data_loader as poly
import pyro
import pyro.distributions as dist
from pyro.distributions import Dirichlet, Beta, Bernoulli, Categorical
from pyro import poutine
from pyro.contrib.autoguide import AutoDelta
from pyro.infer import SVI, JitTraceEnum_ELBO, TraceEnum_ELBO, JitTrace_ELBO, Trace_ELBO
from pyro.optim import Adam
from pyro.util import ignore_jit_warnings
logging.basicConfig(format='%(relativeCreated) 9d %(message)s', level=logging.INFO)

'''
num_sequences = 229
max_length = 129
data_dim = 51
hidden_dim = 16
batch_size= 8
'''
lengths = torch.tensor([129,  65,  49,  65, 114,  33,  57,  49,  64,  33, 108,  48,  49,  48,
         		61,  48,  65,  53,  41,  52,  33,  61,  41,  45,  69,  39,  57,  80,
         		86,  57,  61, 105,  68,  65,  48,  57,  57,  52,  48,  33,  93,  41,
         		65,  49,  73,  48,  33,  45,  65,  52,  49, 109,  49,  52,  65,  41,
        		49,  65,  77,  73,  57,  41,  65,  57,  44,  33,  85,  72,  60,  41,
        		49,  33,  56,  52,  56,  83,  57,  57,  52,  41,  53,  64,  61,  65,
         		65,  48,  25,  96,  41,  37,  76,  65,  65,  72,  41,  65,  41,  37,
         		77,  48,  85,  57,  72,  48,  45,  53,  49,  84,  68,  49,  73,  50,
         		96,  55,  66,  98,  37,  76,  65,  33,  49,  77,  76,  65, 128,  41,
         		45,  61,  33,  88,  61,  63,  65,  68,  60,  49,  48,  45,  65,  45,
         		33,  64,  64,  65,  49,  63,  61,  48,  45,  65,  84, 120,  41, 102,
         		49,  57,  69,  57,  82,  48,  76,  48,  52, 113,  97,  83,  33,  68,
         		41,  49,  65, 109, 108,  60,  65,  57,  49,  33,  57,  33,  61, 113,
         		58,  64,  65,  33,  57,  49,  64,  68,  60,  65,  41,  72,  53,  49,
         		57,  33,  49,  89,  65,  44,  49,  61,  65,  69,  52,  76,  33,  57,
         		76,  57,  33,  76, 101,  40,  41,  44,  76,  65,  49,  54,  49,  49,
         		64,  51, 109,  65,  65])

def model(sequences):
    with poutine.mask(mask=False):
        probs_x = pyro.sample("probs_x",
                              Dirichlet(0.9 * torch.eye(16) + 0.1)
                                  .to_event(1))
        probs_y = pyro.sample("probs_y",
                              Beta(0.1, 0.9)
                                  .expand([16, 51])
                                  .to_event(2))
    tones_plate = pyro.plate("tones", 51, dim=-1)
    for i in pyro.plate("sequences", len(sequences)):
        length = lengths[i]
        sequence = sequences[i, :length]
        x = 0
        for t in pyro.markov(range(length)):
            x = pyro.sample("x_{}_{}".format(i, t), Categorical(probs_x[x]), infer = {"enumerate":"parallel"})
            with tones_plate:
                pyro.sample("y_{}_{}".format(i, t), Bernoulli(probs_y[x.squeeze(-1)]),
                            obs=sequence[t])

def guide(sequences):
    theta = pyro.param("theta", torch.ones(16))
    alpha = pyro.param("alpha", torch.rand(1))
    beta = pyro.param("beta", torch.rand(1))
    p = pyro.param("p", torch.rand(1))
    q = pyro.param("q", torch.rand(1))
    w = p * torch.eye(16) + q
    with poutine.mask(mask=False):        
        probs_x = pyro.sample("probs_x",
                              Dirichlet(w).to_event(1)
                                  )
        probs_y = pyro.sample("probs_y",
                              Beta(alpha, beta).expand([16,51])
                                  .to_event(2))
    
    for i in pyro.plate("sequences", len(sequences), 8):
        length = lengths[i]
        sequence = sequences[i, :length]
        x = 0
        for t in pyro.markov(range(length)):
            x = pyro.sample("x_{}_{}".format(i,t), Categorical(probs_x[x]))

def main(args):
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    logging.info('Loading data')
    data = poly.load_data(poly.JSB_CHORALES)

    logging.info('-' * 40)
    logging.info('Training {} on {} sequences'.format(
        model.__name__, len(data['train']['sequences'])))
    sequences = data['train']['sequences']
    lengths = data['train']['sequence_lengths']

    # find all the notes that are present at least once in the training set
    present_notes = ((sequences == 1).sum(0).sum(0) > 0)
    # remove notes that are never played (we remove 37/88 notes)
    sequences = sequences[..., present_notes]

    if args.truncate:
        lengths.clamp_(max=args.truncate)
        sequences = sequences[:, :args.truncate]
    num_observations = float(lengths.sum())
    pyro.set_rng_seed(0)
    pyro.clear_param_store()
    pyro.enable_validation(True)

    # We'll train using MAP Baum-Welch, i.e. MAP estimation while marginalizing
    # out the hidden state x. This is accomplished via an automatic guide that
    # learns point estimates of all of our conditional probability tables,
    # named probs_*.
    # guide = AutoDelta(poutine.block(model, expose_fn=lambda msg: msg["name"].startswith("probs_")))
    # To help debug our tensor shapes, let's print the shape of each site's
    # distribution, value, and log_prob tensor. Note this information is
    # automatically printed on most errors inside SVI.
    if args.print_shapes:
        first_available_dim = -2 
        guide_trace = poutine.trace(guide).get_trace(
            sequences)
        model_trace = poutine.trace(
            poutine.replay(poutine.enum(model, first_available_dim), guide_trace)).get_trace(
            sequences)
        logging.info(model_trace.format_shapes())

    # Enumeration requires a TraceEnum elbo and declaring the max_plate_nesting.
    # All of our models have two plates: "data" and "tones".
    # Elbo = JitTraceEnum_ELBO if args.jit else TraceEnum_ELBO
    Elbo = Trace_ELBO
    elbo = Elbo(max_plate_nesting=1)
    optim = Adam({'lr': args.learning_rate})
    svi = SVI(model, guide, optim, loss=elbo)

    # We'll train on small minibatches.
    logging.info('Step\tLoss')
    for step in range(args.num_steps):
        loss = svi.step(sequences)
        logging.info('{: >5d}\t{}'.format(step, loss / num_observations))

    # We evaluate on the entire training dataset,
    # excluding the prior term so our results are comparable across models.
    train_loss = elbo.loss(model, guide, sequences)
    logging.info('training loss = {}'.format(train_loss / num_observations))

    # Finally we evaluate on the test dataset.
    logging.info('-' * 40)
    logging.info('Evaluating on {} test sequences'.format(len(data['test']['sequences'])))
    sequences = data['test']['sequences'][..., present_notes]
    lengths = data['test']['sequence_lengths']
    if args.truncate:
        lengths.clamp_(max=args.truncate)
    num_observations = float(lengths.sum())

    # note that since we removed unseen notes above (to make the problem a bit easier and for
    # numerical stability) this test loss may not be directly comparable to numbers
    # reported on this dataset elsewhere.
    test_loss = elbo.loss(model, guide, sequences)
    logging.info('test loss = {}'.format(test_loss / num_observations))

    # We expect models with higher capacity to perform better,
    # but eventually overfit to the training set.
    capacity = sum(value.reshape(-1).size(0)
                   for value in pyro.get_param_store().values())
    logging.info('{} capacity = {} parameters'.format(model.__name__, capacity))


if __name__ == '__main__':
    assert pyro.__version__.startswith('0.3.1')
    parser = argparse.ArgumentParser(description="MAP Baum-Welch learning Bach Chorales")
    # parser.add_argument("-m", "--model", default="1", type=str,
    #                     help="one of: {}".format(", ".join(sorted(models.keys()))))
    parser.add_argument("-n", "--num-steps", default=50, type=int)
    parser.add_argument("-b", "--batch-size", default=8, type=int)
    parser.add_argument("-d", "--hidden-dim", default=16, type=int)
    parser.add_argument("-nn", "--nn-dim", default=48, type=int)
    parser.add_argument("-nc", "--nn-channels", default=2, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.05, type=float)
    parser.add_argument("-t", "--truncate", type=int)
    parser.add_argument("-p", "--print-shapes", action="store_true")
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--jit', action='store_true')
    parser.add_argument('-rp', '--raftery-parameterization', action='store_true')
    args = parser.parse_args()
    main(args)
