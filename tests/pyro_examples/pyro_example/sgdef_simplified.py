"""
Source: /examples/pyro/sparse_gamma_def/sparse_gamma_def_simp.py
"""

from __future__ import absolute_import, division, print_function

import argparse
import errno
import os

import numpy as np
import torch
import torch.nn as nn

import pyro
import pyro.optim as optim
import wget

from pyro.contrib.examples.util import get_data_directory
from pyro.distributions import Gamma, Poisson
from pyro.infer import SVI, TraceMeanField_ELBO
from pyro.contrib.autoguide import AutoDiagonalNormal

torch.set_default_tensor_type('torch.FloatTensor')
pyro.enable_validation(True)
pyro.util.set_rng_seed(0)

#=================================================== simplified code (by wy)

#########
# model #
#########
# hyperparams
alpha_z = torch.tensor(0.1)
beta_z = torch.tensor(0.1)
alpha_w = torch.tensor(0.1)
beta_w = torch.tensor(0.3)

# model
def model(x):
    x = torch.reshape(x, [320, 4096])
    
    with pyro.plate("w_top_plate", 4000):
        w_top = pyro.sample("w_top", Gamma(alpha_w, beta_w))
    with pyro.plate("w_mid_plate", 600):
        w_mid = pyro.sample("w_mid", Gamma(alpha_w, beta_w))
    with pyro.plate("w_bottom_plate", 61440):
        w_bottom = pyro.sample("w_bottom", Gamma(alpha_w, beta_w))

    with pyro.plate("data", 320):
        z_top = pyro.sample("z_top", Gamma(alpha_z, beta_z).expand_by([100]).to_event(1))

        w_top = torch.reshape(w_top, [100, 40])
        mean_mid = torch.matmul(z_top, w_top)
        z_mid = pyro.sample("z_mid", Gamma(alpha_z, beta_z / mean_mid).to_event(1))

        w_mid = torch.reshape(w_mid, [40, 15])
        mean_bottom = torch.matmul(z_mid, w_mid)
        z_bottom = pyro.sample("z_bottom", Gamma(alpha_z, beta_z / mean_bottom).to_event(1))

        w_bottom = torch.reshape(w_bottom, [15, 4096])
        mean_obs = torch.matmul(z_bottom, w_bottom)

        pyro.sample('obs', Poisson(mean_obs).to_event(1), obs=x)

#########
# guide #
#########
# init params
alpha_init = 0.5
mean_init = 0.0
sigma_init = 0.1
softplus = nn.Softplus()

# guide
def guide(x):
    x = torch.reshape(x, [320, 4096])
    
    with pyro.plate("w_top_plate", 4000):
        #============ sample_ws
        alpha_w_q =\
            pyro.param("log_alpha_w_q_top",
                       alpha_init * torch.ones(4000) +
                       sigma_init * torch.randn(4000))
        mean_w_q =\
            pyro.param("log_mean_w_q_top",
                       mean_init * torch.ones(4000) +
                       sigma_init * torch.randn(4000)) 
        alpha_w_q = softplus(alpha_w_q)
        mean_w_q  = softplus(mean_w_q)
        pyro.sample("w_top", Gamma(alpha_w_q, alpha_w_q / mean_w_q))
        #============ sample_ws

    with pyro.plate("w_mid_plate", 600):
        #============ sample_ws
        alpha_w_q =\
            pyro.param("log_alpha_w_q_mid",
                       alpha_init * torch.ones(600) +
                       sigma_init * torch.randn(600)) 
        mean_w_q =\
            pyro.param("log_mean_w_q_mid",
                       mean_init * torch.ones(600) +
                       sigma_init * torch.randn(600)) 
        alpha_w_q = softplus(alpha_w_q)
        mean_w_q  = softplus(mean_w_q)
        pyro.sample("w_mid", Gamma(alpha_w_q, alpha_w_q / mean_w_q))
        #============ sample_ws

    with pyro.plate("w_bottom_plate", 61440):
        #============ sample_ws
        alpha_w_q =\
            pyro.param("log_alpha_w_q_bottom",
                       alpha_init * torch.ones(61440) +
                       sigma_init * torch.randn(61440)) 
        mean_w_q =\
            pyro.param("log_mean_w_q_bottom",
                       mean_init * torch.ones(61440) +
                       sigma_init * torch.randn(61440)) 
        alpha_w_q = softplus(alpha_w_q)
        mean_w_q  = softplus(mean_w_q)
        pyro.sample("w_bottom", Gamma(alpha_w_q, alpha_w_q / mean_w_q))
        #============ sample_ws

    with pyro.plate("data", 320):
        #============ sample_zs
        alpha_z_q =\
            pyro.param("log_alpha_z_q_top",
                       alpha_init * torch.ones(320, 100) +
                       sigma_init * torch.randn(320, 100)) 
        mean_z_q =\
            pyro.param("log_mean_z_q_top",
                       mean_init * torch.ones(320, 100) +
                       sigma_init * torch.randn(320, 100))
        alpha_z_q = softplus(alpha_z_q)
        mean_z_q  = softplus(mean_z_q)
        pyro.sample("z_top", Gamma(alpha_z_q, alpha_z_q / mean_z_q).to_event(1))
        #============ sample_zs
        #============ sample_zs
        alpha_z_q =\
            pyro.param("log_alpha_z_q_mid",
                       alpha_init * torch.ones(320, 40) +
                       sigma_init * torch.randn(320, 40)) 
        mean_z_q =\
            pyro.param("log_mean_z_q_mid",
                       mean_init * torch.ones(320, 40) +
                       sigma_init * torch.randn(320, 40))
        alpha_z_q = softplus(alpha_z_q)
        mean_z_q  = softplus(mean_z_q)
        pyro.sample("z_mid", Gamma(alpha_z_q, alpha_z_q / mean_z_q).to_event(1))
        #============ sample_zs
        #============ sample_zs
        alpha_z_q =\
            pyro.param("log_alpha_z_q_bottom",
                       alpha_init * torch.ones(320, 15) +
                       sigma_init * torch.randn(320, 15)) 
        mean_z_q =\
            pyro.param("log_mean_z_q_bottom",
                       mean_init * torch.ones(320, 15) +
                       sigma_init * torch.randn(320, 15))
        alpha_z_q = softplus(alpha_z_q)
        mean_z_q  = softplus(mean_z_q)
        pyro.sample("z_bottom", Gamma(alpha_z_q, alpha_z_q / mean_z_q).to_event(1))
        #============ sample_zs

#=================================================== original code (model, main)

# define the sizes of the layers in the deep exponential family
top_width = 100
mid_width = 40
bottom_width = 15
image_size = 64 * 64

def model_original(x):
    x_size = x.size(0)

    # sample the global weights
    with pyro.plate("w_top_plate", top_width * mid_width):
        w_top = pyro.sample("w_top", Gamma(alpha_w, beta_w))
    with pyro.plate("w_mid_plate", mid_width * bottom_width):
        w_mid = pyro.sample("w_mid", Gamma(alpha_w, beta_w))
    with pyro.plate("w_bottom_plate", bottom_width * image_size):
        w_bottom = pyro.sample("w_bottom", Gamma(alpha_w, beta_w))

    # sample the local latent random variables
    # (the plate encodes the fact that the z's for different datapoints are conditionally independent)
    with pyro.plate("data", x_size):
        z_top = pyro.sample("z_top", Gamma(alpha_z, beta_z).expand([top_width]).to_event(1))
        # note that we need to use matmul (batch matrix multiplication) as well as appropriate reshaping
        # to make sure our code is fully vectorized
        w_top = w_top.reshape(top_width, mid_width) if w_top.dim() == 1 else \
                w_top.reshape(-1, top_width, mid_width)
        mean_mid = torch.matmul(z_top, w_top)
        z_mid = pyro.sample("z_mid", Gamma(alpha_z, beta_z / mean_mid).to_event(1))

        w_mid = w_mid.reshape(mid_width, bottom_width) if w_mid.dim() == 1 else \
            w_mid.reshape(-1, mid_width, bottom_width)
        mean_bottom = torch.matmul(z_mid, w_mid)
        z_bottom = pyro.sample("z_bottom", Gamma(alpha_z, beta_z / mean_bottom).to_event(1))

        w_bottom = w_bottom.reshape(bottom_width, image_size) if w_bottom.dim() == 1 else \
            w_bottom.reshape(-1, bottom_width, image_size)
        mean_obs = torch.matmul(z_bottom, w_bottom)

        # observe the data using a poisson likelihood
        pyro.sample('obs', Poisson(mean_obs).to_event(1), obs=x)

# define a helper function to clip parameters defining the custom guide.
# (this is to avoid regions of the gamma distributions with extremely small means)
def clip_params():
    for param, clip in zip(("log_alpha", "log_mean"), (-2.5, -4.5)):
        for layer in ["top", "mid", "bottom"]:
            for wz in ["_w_q_", "_z_q_"]:
                pyro.param(param + wz + layer).data.clamp_(min=clip)

def main(args):
    # load data
    print('loading training data...')
    dataset_directory = get_data_directory(__file__)
    dataset_path = os.path.join(dataset_directory, 'faces_training.csv')
    if not os.path.exists(dataset_path):
        try:
            os.makedirs(dataset_directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
            pass
        wget.download('https://d2fefpcigoriu7.cloudfront.net/datasets/faces_training.csv', dataset_path)
    data = torch.tensor(np.loadtxt(dataset_path, delimiter=',')).float()

    learning_rate = 4.5
    momentum = 0.1
    opt = optim.AdagradRMSProp({"eta": learning_rate, "t": momentum})

    # this is the svi object we use during training; we use TraceMeanField_ELBO to
    # get analytic KL divergences
    svi = SVI(model, guide, opt, loss=TraceMeanField_ELBO())

    # we use svi_eval during evaluation; since we took care to write down our model in
    # a fully vectorized way, this computation can be done efficiently with large tensor ops
    svi_eval = SVI(model_original, guide, opt,
                   loss=TraceMeanField_ELBO(num_particles=args.eval_particles,
                                            vectorize_particles=True))

    guide_description = 'custom'
    print('\nbeginning training with %s guide...' % guide_description)

    # the training loop
    for k in range(args.num_epochs):
        loss = svi.step(data)
        clip_params()
        
        if k % args.eval_frequency == 0 and k > 0 or k == args.num_epochs - 1:
            loss = svi_eval.evaluate_loss(data)
            print("[epoch %04d] training elbo: %.4g" % (k, -loss))


if __name__ == '__main__':
    assert pyro.__version__.startswith('0.3.1')
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=1000, type=int, help='number of training epochs')
    parser.add_argument('-ef', '--eval-frequency', default=25, type=int,
                        help='how often to evaluate elbo (number of epochs)')
    parser.add_argument('-ep', '--eval-particles', default=20, type=int,
                        help='number of samples/particles to use during evaluation')

    args = parser.parse_args()
    main(args)
