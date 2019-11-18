########## wy: from utils/custom_mlp.py ##########
class Exp(nn.Module):
    """
    a custom module for exponentiation of tensors
    """
    def __init__(self):
        super(Exp, self).__init__()

    def forward(self, val):
        return torch.exp(val)


class ConcatModule(nn.Module):
    """
    a custom module for concatenation of tensors
    """
    def __init__(self, allow_broadcast=False):
        self.allow_broadcast = allow_broadcast
        super(ConcatModule, self).__init__()

    def forward(self, *input_args):
        # we have a single object
        if len(input_args) == 1:
            # regardless of type,
            # we don't care about single objects
            # we just index into the object
            input_args = input_args[0]

        # don't concat things that are just single objects
        if torch.is_tensor(input_args):
            return input_args
        else:
            if self.allow_broadcast:
                shape = broadcast_shape(*[s.shape[:-1] for s in input_args]) + (-1,)
                input_args = [s.expand(shape) for s in input_args]
            return torch.cat(input_args, dim=-1)


class ListOutModule(nn.ModuleList):
    """
    a custom module for outputting a list of tensors from a list of nn modules
    """
    def __init__(self, modules):
        super(ListOutModule, self).__init__(modules)

    def forward(self, *args, **kwargs):
        # loop over modules in self, apply same args
        return [mm.forward(*args, **kwargs) for mm in self]


def call_nn_op(op):
    """
    a helper function that adds appropriate parameters when calling
    an nn module representing an operation like Softmax

    :param op: the nn.Module operation to instantiate
    :return: instantiation of the op module with appropriate parameters
    """
    if op in [nn.Softmax, nn.LogSoftmax]:
        return op(dim=1)
    else:
        return op()


class MLP(nn.Module):

    def __init__(self, mlp_sizes, activation=nn.ReLU, output_activation=None,
                 post_layer_fct=lambda layer_ix, total_layers, layer: None,
                 post_act_fct=lambda layer_ix, total_layers, layer: None,
                 allow_broadcast=False, use_cuda=False):
        # init the module object
        super(MLP, self).__init__()

        assert len(mlp_sizes) >= 2, "Must have input and output layer sizes defined"

        # get our inputs, outputs, and hidden
        input_size, hidden_sizes, output_size = mlp_sizes[0], mlp_sizes[1:-1], mlp_sizes[-1]

        # assume int or list
        assert isinstance(input_size, (int, list, tuple)), "input_size must be int, list, tuple"

        # everything in MLP will be concatted if it's multiple arguments
        last_layer_size = input_size if type(input_size) == int else sum(input_size)

        # everything sent in will be concatted together by default
        all_modules = [ConcatModule(allow_broadcast)]

        # loop over l
        for layer_ix, layer_size in enumerate(hidden_sizes):
            assert type(layer_size) == int, "Hidden layer sizes must be ints"

            # get our nn layer module (in this case nn.Linear by default)
            cur_linear_layer = nn.Linear(last_layer_size, layer_size)

            # for numerical stability -- initialize the layer properly
            cur_linear_layer.weight.data.normal_(0, 0.001)
            cur_linear_layer.bias.data.normal_(0, 0.001)

            # use GPUs to share data during training (if available)
            if use_cuda:
                cur_linear_layer = nn.DataParallel(cur_linear_layer)

            # add our linear layer
            all_modules.append(cur_linear_layer)

            # handle post_linear
            post_linear = post_layer_fct(layer_ix + 1, len(hidden_sizes), all_modules[-1])

            # if we send something back, add it to sequential
            # here we could return a batch norm for example
            if post_linear is not None:
                all_modules.append(post_linear)

            # handle activation (assumed no params -- deal with that later)
            all_modules.append(activation())

            # now handle after activation
            post_activation = post_act_fct(layer_ix + 1, len(hidden_sizes), all_modules[-1])

            # handle post_activation if not null
            # could add batch norm for example
            if post_activation is not None:
                all_modules.append(post_activation)

            # save the layer size we just created
            last_layer_size = layer_size

        # now we have all of our hidden layers
        # we handle outputs
        assert isinstance(output_size, (int, list, tuple)), "output_size must be int, list, tuple"

        if type(output_size) == int:
            all_modules.append(nn.Linear(last_layer_size, output_size))
            if output_activation is not None:
                all_modules.append(call_nn_op(output_activation)
                                   if isclass(output_activation) else output_activation)
        else:

            # we're going to have a bunch of separate layers we can spit out (a tuple of outputs)
            out_layers = []

            # multiple outputs? handle separately
            for out_ix, out_size in enumerate(output_size):

                # for a single output object, we create a linear layer and some weights
                split_layer = []

                # we have an activation function
                split_layer.append(nn.Linear(last_layer_size, out_size))

                # then we get our output activation (either we repeat all or we index into a same sized array)
                act_out_fct = output_activation if not isinstance(output_activation, (list, tuple)) \
                    else output_activation[out_ix]

                if(act_out_fct):
                    # we check if it's a class. if so, instantiate the object
                    # otherwise, use the object directly (e.g. pre-instaniated)
                    split_layer.append(call_nn_op(act_out_fct)
                                       if isclass(act_out_fct) else act_out_fct)

                # our outputs is just a sequential of the two
                out_layers.append(nn.Sequential(*split_layer))

            all_modules.append(ListOutModule(out_layers))

        # now we have all of our modules, we're ready to build our sequential!
        # process mlps in order, pretty standard here
        self.sequential_mlp = nn.Sequential(*all_modules)

    # pass through our sequential for the output!
    def forward(self, *args, **kwargs):
        return self.sequential_mlp.forward(*args, **kwargs)

########## wy: from ssvae.py ##########
class SSVAE(nn.Module):
    """
    This class encapsulates the parameters (neural networks) and models & guides needed to train a
    semi-supervised variational auto-encoder on the MNIST image dataset
    :param output_size: size of the tensor representing the class label (10 for MNIST since
                        we represent the class labels as a one-hot vector with 10 components)
    :param input_size: size of the tensor representing the image (28*28 = 784 for our MNIST dataset
                       since we flatten the images and scale the pixels to be in [0,1])
    :param z_dim: size of the tensor representing the latent random variable z
                  (handwriting style for our MNIST dataset)
    :param hidden_layers: a tuple (or list) of MLP layers to be used in the neural networks
                          representing the parameters of the distributions in our model
    :param use_cuda: use GPUs for faster training
    :param aux_loss_multiplier: the multiplier to use with the auxiliary loss
    """
    def __init__(self, output_size=10, input_size=784, z_dim=50, hidden_layers=(500,),
                 config_enum=None, use_cuda=False, aux_loss_multiplier=None):

        super(SSVAE, self).__init__()

        # initialize the class with all arguments provided to the constructor
        self.output_size = output_size
        self.input_size = input_size
        self.z_dim = z_dim
        self.hidden_layers = hidden_layers
        self.allow_broadcast = config_enum == 'parallel'
        self.use_cuda = use_cuda
        self.aux_loss_multiplier = aux_loss_multiplier

        # define and instantiate the neural networks representing
        # the paramters of various distributions in the model
        self.setup_networks()

    def setup_networks(self):

        z_dim = self.z_dim
        hidden_sizes = self.hidden_layers

        # define the neural networks used later in the model and the guide.
        # these networks are MLPs (multi-layered perceptrons or simple feed-forward networks)
        # where the provided activation parameter is used on every linear layer except
        # for the output layer where we use the provided output_activation parameter
        self.encoder_y = MLP([self.input_size] + hidden_sizes + [self.output_size],
                             activation=nn.Softplus,
                             output_activation=nn.Softmax,
                             allow_broadcast=self.allow_broadcast,
                             use_cuda=self.use_cuda)

        # a split in the final layer's size is used for multiple outputs
        # and potentially applying separate activation functions on them
        # e.g. in this network the final output is of size [z_dim,z_dim]
        # to produce loc and scale, and apply different activations [None,Exp] on them
        self.encoder_z = MLP([self.input_size + self.output_size] +
                             hidden_sizes + [[z_dim, z_dim]],
                             activation=nn.Softplus,
                             output_activation=[None, Exp],
                             allow_broadcast=self.allow_broadcast,
                             use_cuda=self.use_cuda)

        self.decoder = MLP([z_dim + self.output_size] +
                           hidden_sizes + [self.input_size],
                           activation=nn.Softplus,
                           output_activation=nn.Sigmoid,
                           allow_broadcast=self.allow_broadcast,
                           use_cuda=self.use_cuda)

        # using GPUs for faster training of the networks
        if self.use_cuda:
            self.cuda()

    def model(self, xs, ys=None):
        """
        The model corresponds to the following generative process:
        p(z) = normal(0,I)              # handwriting style (latent)
        p(y|x) = categorical(I/10.)     # which digit (semi-supervised)
        p(x|y,z) = bernoulli(loc(y,z))   # an image
        loc is given by a neural network  `decoder`
        :param xs: a batch of scaled vectors of pixels from an image
        :param ys: (optional) a batch of the class labels i.e.
                   the digit corresponding to the image(s)
        :return: None
        """
        # register this pytorch module and all of its sub-modules with pyro
        pyro.module("ss_vae", self)

        batch_size = xs.size(0)
        with pyro.plate("data"):

            # sample the handwriting style from the constant prior distribution
            prior_loc = xs.new_zeros([batch_size, self.z_dim])
            prior_scale = xs.new_ones([batch_size, self.z_dim])
            zs = pyro.sample("z", dist.Normal(prior_loc, prior_scale).to_event(1))

            # if the label y (which digit to write) is supervised, sample from the
            # constant prior, otherwise, observe the value (i.e. score it against the constant prior)
            alpha_prior = xs.new_ones([batch_size, self.output_size]) / (1.0 * self.output_size)
            ys = pyro.sample("y", dist.OneHotCategorical(alpha_prior), obs=ys)

            # finally, score the image (x) using the handwriting style (z) and
            # the class label y (which digit to write) against the
            # parametrized distribution p(x|y,z) = bernoulli(decoder(y,z))
            # where `decoder` is a neural network
            loc = self.decoder.forward([zs, ys])
            pyro.sample("x", dist.Bernoulli(loc).to_event(1), obs=xs)
            # return the loc so we can visualize it later
            return loc

    def guide(self, xs, ys=None):
        """
        The guide corresponds to the following:
        q(y|x) = categorical(alpha(x))              # infer digit from an image
        q(z|x,y) = normal(loc(x,y),scale(x,y))       # infer handwriting style from an image and the digit
        loc, scale are given by a neural network `encoder_z`
        alpha is given by a neural network `encoder_y`
        :param xs: a batch of scaled vectors of pixels from an image
        :param ys: (optional) a batch of the class labels i.e.
                   the digit corresponding to the image(s)
        :return: None
        """
        # inform Pyro that the variables in the batch of xs, ys are conditionally independent
        with pyro.plate("data"):

            # if the class label (the digit) is not supervised, sample
            # (and score) the digit with the variational distribution
            # q(y|x) = categorical(alpha(x))
            if ys is None:
                alpha = self.encoder_y.forward(xs)
                ys = pyro.sample("y", dist.OneHotCategorical(alpha))

            # sample (and score) the latent handwriting-style with the variational
            # distribution q(z|x,y) = normal(loc(x,y),scale(x,y))
            loc, scale = self.encoder_z.forward([xs, ys])
            pyro.sample("z", dist.Normal(loc, scale).to_event(1))

    def classifier(self, xs):
        """
        classify an image (or a batch of images)
        :param xs: a batch of scaled vectors of pixels from an image
        :return: a batch of the corresponding class labels (as one-hots)
        """
        # use the trained model q(y|x) = categorical(alpha(x))
        # compute all class probabilities for the image(s)
        alpha = self.encoder_y.forward(xs)

        # get the index (digit) that corresponds to
        # the maximum predicted class probability
        res, ind = torch.topk(alpha, 1)

        # convert the digit(s) to one-hot tensor(s)
        ys = xs.new_zeros(alpha.size())
        ys = ys.scatter_(1, ind, 1.0)
        return ys

    def model_classify(self, xs, ys=None):
        """
        this model is used to add an auxiliary (supervised) loss as described in the
        Kingma et al., "Semi-Supervised Learning with Deep Generative Models".
        """
        # register all pytorch (sub)modules with pyro
        pyro.module("ss_vae", self)

        # inform Pyro that the variables in the batch of xs, ys are conditionally independent
        with pyro.plate("data"):
            # this here is the extra term to yield an auxiliary loss that we do gradient descent on
            if ys is not None:
                alpha = self.encoder_y.forward(xs)
                with pyro.poutine.scale(scale=self.aux_loss_multiplier):
                    pyro.sample("y_aux", dist.OneHotCategorical(alpha), obs=ys)

    def guide_classify(self, xs, ys=None):
        """
        dummy guide function to accompany model_classify in inference
        """
        pass
