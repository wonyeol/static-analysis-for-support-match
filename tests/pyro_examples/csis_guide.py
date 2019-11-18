#########
# guide #
#########
first  = nn.Linear(2, 10)
second = nn.Linear(10, 20)
third  = nn.Linear(20, 10)
fourth = nn.Linear(10, 5)
fifth  = nn.Linear(5, 2)
relu = nn.ReLU()

pyro.module("first", first)
pyro.module("second", second)
pyro.module("third", third)
pyro.module("fourth", fourth)
pyro.module("fifth", fifth)

obs = torch.tensor([float(observations['x1']),
                    float(observations['x2'])])
x1 = obs[0]
x2 = obs[1]
v = torch.cat((torch.Tensor.view(x1, [1, 1]),
               torch.Tensor.view(x2, [1, 1])), 1)

h1  = relu(first(v))
h2  = relu(second(h1))
h3  = relu(third(h2))
h4  = relu(fourth(h3))
out = fifth(h4)

mean = out[0, 0]
std = torch.exp(out[0, 1])
pyro.sample("z", Normal(mean, std))
