"""
Source: ./lda_simp_2.py  
PASS when Trace_ELBO is used for inference.
FAIL when TraceEnum_ELBO is used for inference.
"""

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
data = torch.reshape(data, [64, 1000])

pyro.module("layer1", layer1)
pyro.module("layer2", layer2)
pyro.module("layer3", layer3)

# Use a conjugate guide for global variables.
topic_weights_posterior = pyro.param(
    "topic_weights_posterior",
    # lambda: torch.ones(8) / 8,
    torch.ones(8) / 8,
    constraint=constraints.positive)
topic_words_posterior = pyro.param(
    "topic_words_posterior",
    # lambda: torch.ones(8, 1024) / 1024,
    torch.ones(8, 1024) / 1024,
    constraint=constraints.positive)

with pyro.plate("topics", 8):
    # shape = [8] + []
    topic_weights = pyro.sample("topic_weights", Gamma(topic_weights_posterior, 1.))
    # shape = [8] + [1024]
    topic_words = pyro.sample("topic_words", Dirichlet(topic_words_posterior))

# Use an amortized guide for local variables.
with pyro.plate("documents", 1000, 32) as ind:
    # shape =  [64, 32]
    data = data[:, ind]
    # The neural network will operate on histograms rather than word
    # index vectors, so we'll convert the raw data to a histogram.
    counts = torch.zeros(1024, 32)
    counts = torch.Tensor.scatter_add_\
        (counts, 0, data,
         torch.Tensor.expand(torch.tensor(1.), [1024, 32]))
    h1 = sigmoid(layer1(torch.transpose(counts, 0, 1)))
    h2 = sigmoid(layer2(h1))
    # shape = [32, 8]
    doc_topics_w = sigmoid(layer3(h2))
    # shape = [32] + [8]
    # # doc_topics = pyro.sample("doc_topics", Delta(doc_topics_w, event_dim=1))
    # doc_topics = pyro.sample("doc_topics", Delta(doc_topics_w).to_event(1))
    doc_topics = pyro.sample("doc_topics", Dirichlet(doc_topics_w))

    # wy: sample from exact posterior of word_topics
    with pyro.plate("words", 64):
        # ks : [K, D] = [8, 32]
        # ks = torch.arange(0,8).expand(32,8).transpose(0,1)
        ks = torch.arange(0, 8)
        ks = torch.Tensor.expand(ks, 32, 8)
        ks = torch.Tensor.transpose(ks, 0, 1)
        # logprob1 : [N, D, K] = [32, 8]
        # logprob1 = Categorical(doc_topics).log_prob(ks).transpose(0,1).expand(64,32,8)
        logprob1 = Categorical.log_prob(Categorical(doc_topics), ks)
        logprob1 = torch.Tensor.transpose(logprob1, 0, 1)
        logprob1 = torch.Tensor.expand(logprob1, 64, 32, 8)
        # data2 : [N, D, K] = [64, 32, 8]
        # data2 = data.expand(8,64,32).transpose(0,1).transpose(1,2)
        data2 = torch.Tensor.expand(data, 8, 64, 32)
        data2 = torch.Tensor.transpose(data2, 0, 1)
        data2 = torch.Tensor.transpose(data2, 1, 2)
        # logprob2 : [N, D, K] = [64, 32, 8]
        # logprob2 = Categorical(topic_words).log_prob(data2)
        logprob2 = Categorical.log_prob(Categorical(topic_words), data2)
        # prob : [N, D, K] = [64, 32, 8]
        prob = torch.exp(logprob1 + logprob2)
        # word_topics : [N, D] = [64, 32]
        word_topics = pyro.sample("word_topics", Categorical(prob))
