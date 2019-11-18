########## wy: from csis_original.py ##########
def model(prior_mean, observations={"x1": 0, "x2": 0}):
    x = pyro.sample("z", dist.Normal(prior_mean, torch.tensor(5**0.5)))
    y1 = pyro.sample("x1", dist.Normal(x, torch.tensor(2**0.5)), obs=observations["x1"])
    y2 = pyro.sample("x2", dist.Normal(x, torch.tensor(2**0.5)), obs=observations["x2"])
    return x

class Guide(nn.Module):
    def __init__(self):
        super(Guide, self).__init__()
        self.neural_net = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2))

    def forward(self, prior_mean, observations={"x1": 0, "x2": 0}):
        pyro.module("guide", self)
        x1 = observations["x1"]
        x2 = observations["x2"]
        v = torch.cat((x1.view(1, 1), x2.view(1, 1)), 1)
        v = self.neural_net(v)
        mean = v[0, 0]
        std = v[0, 1].exp()
        pyro.sample("z", dist.Normal(mean, std))
