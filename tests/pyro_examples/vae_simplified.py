import torch
import torchvision.datasets as dset
import torch.nn as nn
import torchvision.transforms as transforms

import pyro
import pyro.distributions as dist
from pyro.distributions import Normal, Bernoulli
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

pyro.set_rng_seed(0)

def setup_data_loaders(batch_size=128):
    root = './data'
    download = True
    trans = transforms.ToTensor()
    train_set = dset.MNIST(root=root, train=True, transform=trans,
                           download=download)
    test_set = dset.MNIST(root=root, train=False, transform=trans)
    kwargs = {'num_workers': 1}
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=batch_size,
                                               shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=batch_size,
                                              shuffle=False, **kwargs)
    return train_loader, test_loader

def train(svi, train_loader):
    epoch_loss = 0.
    for x, _ in train_loader:
        if x.shape[0] == 256:
            epoch_loss += svi.step(x)

    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train

def evaluate(svi, test_loader):
    test_loss = 0.
    for x, _ in test_loader:
        if x.shape[0] == 256:
            test_loss += svi.evaluate_loss(x)

    normalizer_test = len(test_loader.dataset)
    total_epoch_loss_test = test_loss / normalizer_test
    return total_epoch_loss_test

LEARNING_RATE = 1.0e-3
NUM_EPOCHS = 100
TEST_FREQUENCY = 5

train_loader, test_loader = setup_data_loaders(batch_size=256)

# We replaced z_dim by 50. All 50's in this code are z_dim.
z_dim=50 
# We replaced hidden_dim by 400. All 400's in this code are hidden_dim.
hidden_dim=400

m_fc1 = nn.Linear(50, 400)
m_fc21 = nn.Linear(400, 784)
softplus = nn.Softplus()
sigmoid = nn.Sigmoid()

def model(x):
    pyro.module("decoder_fc1", m_fc1)
    pyro.module("decoder_fc21", m_fc21)
    with pyro.plate("data", 256):
        z_loc = torch.zeros([256, 50])
        z_scale = torch.ones([256, 50])
        z = pyro.sample("latent", Normal(z_loc, z_scale).to_event(1))
        hidden = softplus(m_fc1(z))
        loc_img = sigmoid(m_fc21(hidden))
        pyro.sample("obs", Bernoulli(loc_img).to_event(1), obs=torch.reshape(x, [256, 784]))
        return loc_img

g_fc1 = nn.Linear(784, 400)
g_fc21 = nn.Linear(400, 50)
g_fc22 = nn.Linear(400, 50)
softplus = nn.Softplus()

def guide(x):
    pyro.module("encoder_fc1", g_fc1)
    pyro.module("encoder_fc21", g_fc21)
    pyro.module("encoder_fc22", g_fc22)
    with pyro.plate("data", 256):
        x = torch.reshape(x, [256, 784])
        hidden = softplus(g_fc1(x))
        z_loc = g_fc21(hidden)
        z_scale = torch.exp(g_fc22(hidden))
        pyro.sample("latent", Normal(z_loc, z_scale).to_event(1))

pyro.clear_param_store()

adam_args = {"lr": LEARNING_RATE}
optimizer = Adam(adam_args)

svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

train_elbo = []
test_elbo = []

for epoch in range(NUM_EPOCHS):
    total_epoch_loss_train = train(svi, train_loader)
    train_elbo.append(-total_epoch_loss_train)
    print("[epoch %03d] averatge training loss: %.4f" % (epoch, total_epoch_loss_train))
    
    if epoch % TEST_FREQUENCY == 0:
        total_epoch_loss_test = evaluate(svi, test_loader)
        test_elbo.append(-total_epoch_loss_test)
        print("[epoch %03d] average test loss: %.4f" % (epoch, total_epoch_loss_test))
