"""
Source: /examples/pyro/lda/lda_simp_2_hg.py
"""

# source: http://pyro.ai/examples/lda.html
# replaced some args.* variables into default value.
#
# This is a variant of the simplified lda example. 
# Here we sample cartegorical random variables z in guide, using the exact posterior probability of z defined by model.
# Expected learning behavior of this code is exactly same as the original code http://pyro.ai/examples/lda.html, but I guess there are some bugs.


from __future__ import absolute_import, division, print_function

import argparse
import functools
import logging

import torch
from torch import nn
from torch.distributions import constraints

import pyro
from pyro.distributions import Gamma, Dirichlet, Categorical, Delta 
import pyro.distributions as dist
from pyro.infer import SVI, JitTraceEnum_ELBO, TraceEnum_ELBO, TraceGraph_ELBO
from pyro.optim import Adam

logging.basicConfig(format='%(relativeCreated) 9d %(message)s', level=logging.INFO)
debug = True

#########
# model #
#########
# This is a fully generative model of a batch of documents.
# data is a [num_words_per_doc, num_documents] shaped array of word ids
# (specifically it is not a histogram). We assume in this simple example
# that all documents have the same number of words.
def model(data=None, args=None, batch_size=None):
    if debug: print("model:")
    data = torch.reshape(data, [64, 1000])
    
    # Globals.
    with pyro.plate("topics", 8):
        # shape = [8] + []
        topic_weights = pyro.sample("topic_weights", Gamma(1. / 8, 1.))
        # shape = [8] + [1024]
        topic_words = pyro.sample("topic_words", Dirichlet(torch.ones(1024) / 1024))
        if debug:
            print("topic_weights\t: shape={}, sum={}".
                  format(topic_weights.shape, torch.sum(topic_weights)))
            print("topic_words\t: shape={}".format(topic_words.shape))

    # Locals.
    # with pyro.plate("documents", 1000) as ind:
    with pyro.plate("documents", 1000, 32, dim=-1) as ind:
        # if data is not None:
        #     data = data[:, ind]
        # shape = [64, 32]
        data = data[:, ind]
        # shape = [32] + [8]
        doc_topics = pyro.sample("doc_topics", Dirichlet(topic_weights))
        if debug:
            print("data\t\t: shape={}".format(data.shape))
            print("doc_topics\t: shape={}, [0].sum={}".
                  format(doc_topics.shape, torch.sum(doc_topics[0])))

        # with pyro.plate("words", 64):
        with pyro.plate("words", 64, dim=-2):
            # The word_topics variable is marginalized out during inference,
            # achieved by specifying infer={"enumerate": "parallel"} and using
            # TraceEnum_ELBO for inference. Thus we can ignore this variable in
            # the guide.
            # shape = [64, 32] + []
            word_topics =\
                pyro.sample("word_topics", Categorical(doc_topics),
                            infer={"enumerate": "parallel"})
                # pyro.sample("word_topics", Categorical(doc_topics))
            # shape = [64, 32] + []
            data =\
                pyro.sample("doc_words", Categorical(topic_words[word_topics]),
                            obs=data)
            if debug:
                print("word_topics\t: shape={}".format(word_topics.shape))
                print("data\t\t: shape={}".format(data.shape))
            
    return topic_weights, topic_words, data

#########
# guide #
#########
# nn
layer1 = nn.Linear(1024,100)
layer2 = nn.Linear(100,100)
layer3 = nn.Linear(100,8)
sigmoid = nn.Sigmoid()

layer1.weight.data.normal_(0, 0.001)
layer1.bias.data.normal_(0, 0.001)
layer2.weight.data.normal_(0, 0.001)
layer2.bias.data.normal_(0, 0.001)
layer3.weight.data.normal_(0, 0.001)
layer3.bias.data.normal_(0, 0.001)

# guide
def guide(data, args, batch_size=None):
    if debug: print("guide:")
    data = torch.reshape(data, [64, 1000])
    
    pyro.module("layer1", layer1)
    pyro.module("layer2", layer2)
    pyro.module("layer3", layer3)

    # Use a conjugate guide for global variables.
    topic_weights_posterior = pyro.param(
        "topic_weights_posterior",
        torch.ones(8) / 8,
        constraint=constraints.positive)
    topic_words_posterior = pyro.param(
        "topic_words_posterior",
        torch.ones(8, 1024) / 1024,
        constraint=constraints.positive)
    
    with pyro.plate("topics", 8):
        # shape = [8] + []
        topic_weights = pyro.sample("topic_weights", Gamma(topic_weights_posterior, 1.))
        # shape = [8] + [1024]
        topic_words = pyro.sample("topic_words", Dirichlet(topic_words_posterior))
        if debug:
            print("topic_weights\t: shape={}, sum={}".
                  format(topic_weights.shape, torch.sum(topic_weights)))
            print("topic_words\t: shape={}".format(topic_words.shape))

    # Use an amortized guide for local variables.
    with pyro.plate("documents", 1000, 32, dim=-1) as ind:
        # shape =  [64, 32]
        data = data[:, ind]
        # The neural network will operate on histograms rather than word
        # index vectors, so we'll convert the raw data to a histogram.
        counts = torch.zeros(1024, 32)
        counts = torch.Tensor.scatter_add_\
            (counts, 0, data,
             torch.Tensor.expand(torch.tensor(1.), [1024, 32]))
        if debug:
            print("counts.shape={}, counts_trans.shape={}".
                  format(counts.shape, torch.transpose(counts, 0, 1).shape))
        h1 = sigmoid(layer1(torch.transpose(counts, 0, 1)))
        h2 = sigmoid(layer2(h1))
        # shape = [32, 8]
        doc_topics = sigmoid(layer3(h2))
        if debug: 
            print("doc_topics(nn result)\t: shape={}, [0].sum={}".
                format(doc_topics.shape, torch.sum(doc_topics[0])))
            d = Dirichlet(doc_topics)
            print("Dirichlet(doc_topics)\t: batch_shape={}, event_shape={}".
                    format(d.batch_shape, d.event_shape))
        
        """
        NOTE: There are three options we can take:
        1. sample doc_topics (from *DELTA*) *ONLY*. (original code)
        2. sample doc_topics (from *DIRICHLET*) *ONLY*.
        3. sample doc_topics (from *DIRICHLET*), word_topics (from Categorical) *BOTH*.

        Expected results:
          | Trace_ELBO | TraceEnum_ELBO 
        -------------------------------
        1 | FAIL       | FAIL
        2 | FAIL       | PASS
        3 | PASS       | FAIL
        """
        # shape = [32] + [8]
        # doc_topics = pyro.sample("doc_topics", Delta(doc_topics, event_dim=1))
        # doc_topics = pyro.sample("doc_topics", Delta(doc_topics).to_event(1))
        doc_topics = pyro.sample("doc_topics", Dirichlet(doc_topics))
        if debug:
            print("doc_topics(sampled)\t: shape={}, [0].sum = {}".
                format(doc_topics.shape, torch.sum(doc_topics[0])))
        with pyro.plate("words", 64, dim =-2):
            # shape = [64, 32, 8]
            word_topics_posterior = doc_topics * topic_words.transpose(0, 1)[data, :]
            word_topics_posterior = word_topics_posterior /(word_topics_posterior.sum(dim=-1, keepdim=True))
            word_topics =\
                pyro.sample("word_topics", Categorical(word_topics_posterior))            
            if debug:
                print("word_tpics_posterior\t: shape={}".
                    format(word_topics_posterior.shape))
                print("word_topics(sampled)\t: shape={}".
                    format(word_topics.shape))
                d = Categorical(word_topics_posterior)
                print("Categorical(word_topics_posterior)\t: batch_shape={}, event_shape={}".
                    format(d.batch_shape, d.event_shape))
                
#======================== original
# This is a fully generative model of a batch of documents.
# data is a [num_words_per_doc, num_documents] shaped array of word ids
# (specifically it is not a histogram). We assume in this simple example
# that all documents have the same number of words.
def model_original(data=None, args=None, batch_size=None):
    # Globals.
    with pyro.plate("topics", 8):
        topic_weights = pyro.sample("topic_weights", Gamma(1. / 8, 1.))
        topic_words = pyro.sample("topic_words", Dirichlet(torch.ones(1024) / 1024))
    # Locals.
    with pyro.plate("documents", 1000) as ind:
        if data is not None:
            data = data[:, ind]
        doc_topics = pyro.sample("doc_topics", Dirichlet(topic_weights))
        with pyro.plate("words", 64):
            # The word_topics variable is marginalized out during inference,
            # achieved by specifying infer={"enumerate": "parallel"} and using
            # TraceEnum_ELBO for inference. Thus we can ignore this variable in
            # the guide.
            word_topics =\
                pyro.sample("word_topics", Categorical(doc_topics),
                            infer={"enumerate": "parallel"})
            data =\
                pyro.sample("doc_words", Categorical(topic_words[word_topics]),
                            obs=data)
    return topic_weights, topic_words, data
#======================== original

def main(args):
    # pyro.enable_validation(True)
    
    logging.info('Generating data')
    pyro.set_rng_seed(0)
    # We can generate synthetic data directly by calling the model.
    true_topic_weights, true_topic_words, data = model_original(args=args)

    # We'll train using SVI.
    logging.info('-' * 40)
    logging.info('Training on {} documents'.format(args.num_docs))
    # wy: currently don't do enumeration.
    # # Elbo = JitTraceEnum_ELBO if args.jit else TraceEnum_ELBO
    # Elbo = TraceEnum_ELBO
    Elbo = TraceGraph_ELBO
    elbo = Elbo(max_plate_nesting=2)
    optim = Adam({'lr': args.learning_rate})
    svi = SVI(model, guide, optim, elbo)
    logging.info('Step\tLoss')
    for step in range(args.num_steps):
        loss = svi.step(data, args=args, batch_size=args.batch_size)
        if step % 10 == 0:
            logging.info('{: >5d}\t{}'.format(step, loss))
    loss = elbo.loss(model, guide, data, args=args)
    logging.info('final loss = {}'.format(loss))


if __name__ == '__main__':
    # assert pyro.__version__.startswith('0.3.0')
    parser = argparse.ArgumentParser(description="Amortized Latent Dirichlet Allocation")
    parser.add_argument("-t", "--num-topics", default=8, type=int)
    parser.add_argument("-w", "--num-words", default=1024, type=int)
    parser.add_argument("-d", "--num-docs", default=1000, type=int)
    parser.add_argument("-wd", "--num-words-per-doc", default=64, type=int)
    parser.add_argument("-n", "--num-steps", default=1000, type=int)
    # parser.add_argument("-n", "--num-steps", default=100, type=int)
    parser.add_argument("-l", "--layer-sizes", default="100-100")
    parser.add_argument("-lr", "--learning-rate", default=0.001, type=float)
    parser.add_argument("-b", "--batch-size", default=32, type=int)
    # parser.add_argument('--jit', action='store_true')
    args = parser.parse_args()
    main(args)
