#########
# model #
#########
# This is a fully generative model of a batch of documents.
# data is a [num_words_per_doc, num_documents] shaped array of word ids
# (specifically it is not a histogram). We assume in this simple example
# that all documents have the same number of words.
data = torch.reshape(data, [64, 1000])

# Globals.
with pyro.plate("topics", 8):
    # shape = [8] + []
    topic_weights = pyro.sample("topic_weights", Gamma(1. / 8, 1.))
    # shape = [8] + [1024]
    topic_words = pyro.sample("topic_words", Dirichlet(torch.ones(1024) / 1024))

# Locals.
# with pyro.plate("documents", 1000) as ind:
with pyro.plate("documents", 1000, 32, dim=-1) as ind:
    # if data is not None:
    #     data = data[:, ind]
    # shape = [64, 32]
    data = data[:, ind]
    # shape = [32] + [8]
    doc_topics = pyro.sample("doc_topics", Dirichlet(topic_weights))

    with pyro.plate("words", 64, dim=-2):
        # The word_topics variable is marginalized out during inference,
        # achieved by specifying infer={"enumerate": "parallel"} and using
        # TraceEnum_ELBO for inference. Thus we can ignore this variable in
        # the guide.
        # shape = [64, 32] + []
        word_topics =\
            pyro.sample("word_topics", Categorical(doc_topics),
                        infer={"enumerate": "parallel"})
        # shape = [64, 32] + []
        data =\
            pyro.sample("doc_words", Categorical(topic_words[word_topics]),
                        obs=data)
