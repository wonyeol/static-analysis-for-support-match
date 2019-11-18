########## wy: from modules.py ##########
# Takes pixel intensities of the attention window to parameters (mean,
# standard deviation) of the distribution over the latent code,
# z_what.
class Encoder(nn.Module):
    def __init__(self, x_size, h_sizes, z_size, non_linear_layer):
        super(Encoder, self).__init__()
        self.z_size = z_size
        output_size = 2 * z_size
        self.mlp = MLP(x_size, h_sizes + [output_size], non_linear_layer)

    def forward(self, x):
        a = self.mlp(x)
        return a[:, 0:self.z_size], softplus(a[:, self.z_size:])


# Takes a latent code, z_what, to pixel intensities.
class Decoder(nn.Module):
    def __init__(self, x_size, h_sizes, z_size, bias, use_sigmoid, non_linear_layer):
        super(Decoder, self).__init__()
        self.bias = bias
        self.use_sigmoid = use_sigmoid
        self.mlp = MLP(z_size, h_sizes + [x_size], non_linear_layer)

    def forward(self, z):
        a = self.mlp(z)
        if self.bias is not None:
            a = a + self.bias
        return torch.sigmoid(a) if self.use_sigmoid else a


# A general purpose module to construct networks that look like:
# [Linear (256 -> 1)]
# [Linear (256 -> 256), ReLU (), Linear (256 -> 1)]
# [Linear (256 -> 256), ReLU (), Linear (256 -> 1), ReLU ()]
# etc.
class MLP(nn.Module):
    def __init__(self, in_size, out_sizes, non_linear_layer, output_non_linearity=False):
        super(MLP, self).__init__()
        assert len(out_sizes) >= 1
        layers = []
        in_sizes = [in_size] + out_sizes[0:-1]
        sizes = list(zip(in_sizes, out_sizes))
        for (i, o) in sizes[0:-1]:
            layers.append(nn.Linear(i, o))
            layers.append(non_linear_layer())
        layers.append(nn.Linear(sizes[-1][0], sizes[-1][1]))
        if output_non_linearity:
            layers.append(non_linear_layer())
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


# Takes the guide RNN hidden state to parameters of the guide
# distributions over z_where and z_pres.
class Predict(nn.Module):
    def __init__(self, input_size, h_sizes, z_pres_size, z_where_size, non_linear_layer):
        super(Predict, self).__init__()
        self.z_pres_size = z_pres_size
        self.z_where_size = z_where_size
        output_size = z_pres_size + 2 * z_where_size
        self.mlp = MLP(input_size, h_sizes + [output_size], non_linear_layer)

    def forward(self, h):
        out = self.mlp(h)
        z_pres_p = torch.sigmoid(out[:, 0:self.z_pres_size])
        z_where_loc = out[:, self.z_pres_size:self.z_pres_size + self.z_where_size]
        z_where_scale = softplus(out[:, (self.z_pres_size + self.z_where_size):])
        return z_pres_p, z_where_loc, z_where_scale


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


########## wy: from air.py ##########
# Default prior success probability for z_pres.
def default_z_pres_prior_p(t):
    return 0.5


ModelState = namedtuple('ModelState', ['x', 'z_pres', 'z_where'])
GuideState = namedtuple('GuideState', ['h', 'c', 'bl_h', 'bl_c', 'z_pres', 'z_where', 'z_what'])


class AIR(nn.Module):
    def __init__(self,
                 num_steps,
                 x_size,
                 window_size,
                 z_what_size,
                 rnn_hidden_size,
                 encoder_net=[],
                 decoder_net=[],
                 predict_net=[],
                 embed_net=None,
                 bl_predict_net=[],
                 non_linearity='ReLU',
                 decoder_output_bias=None,
                 decoder_output_use_sigmoid=False,
                 use_masking=True,
                 use_baselines=True,
                 baseline_scalar=None,
                 scale_prior_mean=3.0,
                 scale_prior_sd=0.1,
                 pos_prior_mean=0.0,
                 pos_prior_sd=1.0,
                 likelihood_sd=0.3,
                 use_cuda=False):

        super(AIR, self).__init__()

        self.num_steps = num_steps
        self.x_size = x_size
        self.window_size = window_size
        self.z_what_size = z_what_size
        self.rnn_hidden_size = rnn_hidden_size
        self.use_masking = use_masking
        self.use_baselines = use_baselines
        self.baseline_scalar = baseline_scalar
        self.likelihood_sd = likelihood_sd
        self.use_cuda = use_cuda
        self.prototype = torch.tensor(0.).cuda() if use_cuda else torch.tensor(0.)

        self.z_pres_size = 1
        self.z_where_size = 3
        # By making these parameters they will be moved to the gpu
        # when necessary. (They are not registered with pyro for
        # optimization.)
        self.z_where_loc_prior = nn.Parameter(
            torch.FloatTensor([scale_prior_mean, pos_prior_mean, pos_prior_mean]),
            requires_grad=False)
        self.z_where_scale_prior = nn.Parameter(
            torch.FloatTensor([scale_prior_sd, pos_prior_sd, pos_prior_sd]),
            requires_grad=False)

        # Create nn modules.
        rnn_input_size = x_size ** 2 if embed_net is None else embed_net[-1]
        rnn_input_size += self.z_where_size + z_what_size + self.z_pres_size
        nl = getattr(nn, non_linearity)

        self.rnn = nn.LSTMCell(rnn_input_size, rnn_hidden_size)
        self.encode = Encoder(window_size ** 2, encoder_net, z_what_size, nl)
        self.decode = Decoder(window_size ** 2, decoder_net, z_what_size,
                              decoder_output_bias, decoder_output_use_sigmoid, nl)
        self.predict = Predict(rnn_hidden_size, predict_net, self.z_pres_size, self.z_where_size, nl)
        self.embed = Identity() if embed_net is None else MLP(x_size ** 2, embed_net, nl, True)

        self.bl_rnn = nn.LSTMCell(rnn_input_size, rnn_hidden_size)
        self.bl_predict = MLP(rnn_hidden_size, bl_predict_net + [1], nl)
        self.bl_embed = Identity() if embed_net is None else MLP(x_size ** 2, embed_net, nl, True)

        # Create parameters.
        self.h_init = nn.Parameter(torch.zeros(1, rnn_hidden_size))
        self.c_init = nn.Parameter(torch.zeros(1, rnn_hidden_size))
        self.bl_h_init = nn.Parameter(torch.zeros(1, rnn_hidden_size))
        self.bl_c_init = nn.Parameter(torch.zeros(1, rnn_hidden_size))
        self.z_where_init = nn.Parameter(torch.zeros(1, self.z_where_size))
        self.z_what_init = nn.Parameter(torch.zeros(1, self.z_what_size))

        if use_cuda:
            self.cuda()

    def prior(self, n, **kwargs):

        state = ModelState(
            x=self.prototype.new_zeros([n, self.x_size, self.x_size]),
            z_pres=self.prototype.new_ones([n, self.z_pres_size]),
            z_where=None)

        z_pres = []
        z_where = []

        for t in range(self.num_steps):
            state = self.prior_step(t, n, state, **kwargs)
            z_where.append(state.z_where)
            z_pres.append(state.z_pres)

        return (z_where, z_pres), state.x

    def prior_step(self, t, n, prev, z_pres_prior_p=default_z_pres_prior_p):

        # Sample presence indicators.
        z_pres = pyro.sample('z_pres_{}'.format(t),
                             dist.Bernoulli(z_pres_prior_p(t) * prev.z_pres)
                                 .to_event(1))

        # If zero is sampled for a data point, then no more objects
        # will be added to its output image. We can't
        # straight-forwardly avoid generating further objects, so
        # instead we zero out the log_prob_sum of future choices.
        sample_mask = z_pres if self.use_masking else torch.tensor(1.0)

        # Sample attention window position.
        z_where = pyro.sample('z_where_{}'.format(t),
                              dist.Normal(self.z_where_loc_prior.expand(n, self.z_where_size),
                                          self.z_where_scale_prior.expand(n, self.z_where_size))
                                  .mask(sample_mask)
                                  .to_event(1))

        # Sample latent code for contents of the attention window.
        z_what = pyro.sample('z_what_{}'.format(t),
                             dist.Normal(self.prototype.new_zeros([n, self.z_what_size]),
                                         self.prototype.new_ones([n, self.z_what_size]))
                                 .mask(sample_mask)
                                 .to_event(1))

        # Map latent code to pixel space.
        y_att = self.decode(z_what)

        # Position/scale attention window within larger image.
        y = window_to_image(z_where, self.window_size, self.x_size, y_att)

        # Combine the image generated at this step with the image so far.
        # (Note that there's no notion of occlusion here. Overlapping
        # objects can create pixel intensities > 1.)
        x = prev.x + (y * z_pres.view(-1, 1, 1))

        return ModelState(x=x, z_pres=z_pres, z_where=z_where)

    def model(self, data, batch_size, **kwargs):
        pyro.module("decode", self.decode)
        with pyro.plate('data', data.size(0), device=data.device) as ix:
            batch = data[ix]
            n = batch.size(0)
            (z_where, z_pres), x = self.prior(n, **kwargs)
            pyro.sample('obs',
                        dist.Normal(x.view(n, -1),
                                    (self.likelihood_sd * self.prototype.new_ones(n, self.x_size ** 2)))
                            .to_event(1),
                        obs=batch.view(n, -1))

    def guide(self, data, batch_size, **kwargs):
        pyro.module('rnn', self.rnn),
        pyro.module('predict', self.predict),
        pyro.module('encode', self.encode),
        pyro.module('embed', self.embed),
        pyro.module('bl_rnn', self.bl_rnn),
        pyro.module('bl_predict', self.bl_predict),
        pyro.module('bl_embed', self.bl_embed)

        pyro.param('h_init', self.h_init)
        pyro.param('c_init', self.c_init)
        pyro.param('z_where_init', self.z_where_init)
        pyro.param('z_what_init', self.z_what_init)
        pyro.param('bl_h_init', self.bl_h_init)
        pyro.param('bl_c_init', self.bl_c_init)

        with pyro.plate('data', data.size(0), subsample_size=batch_size, device=data.device) as ix:
            batch = data[ix]
            n = batch.size(0)

            # Embed inputs.
            flattened_batch = batch.view(n, -1)
            inputs = {
                'raw': batch,
                'embed': self.embed(flattened_batch),
                'bl_embed': self.bl_embed(flattened_batch)
            }

            # Initial state.
            state = GuideState(
                h=batch_expand(self.h_init, n),
                c=batch_expand(self.c_init, n),
                bl_h=batch_expand(self.bl_h_init, n),
                bl_c=batch_expand(self.bl_c_init, n),
                z_pres=self.prototype.new_ones(n, self.z_pres_size),
                z_where=batch_expand(self.z_where_init, n),
                z_what=batch_expand(self.z_what_init, n))

            z_pres = []
            z_where = []

            for t in range(self.num_steps):
                state = self.guide_step(t, n, state, inputs)
                z_where.append(state.z_where)
                z_pres.append(state.z_pres)

            return z_where, z_pres

    def guide_step(self, t, n, prev, inputs):

        rnn_input = torch.cat((inputs['embed'], prev.z_where, prev.z_what, prev.z_pres), 1)
        h, c = self.rnn(rnn_input, (prev.h, prev.c))
        z_pres_p, z_where_loc, z_where_scale = self.predict(h)

        # Compute baseline estimates for discrete choice z_pres.
        infer_dict, bl_h, bl_c = self.baseline_step(prev, inputs)

        # Sample presence.
        z_pres = pyro.sample('z_pres_{}'.format(t),
                             dist.Bernoulli(z_pres_p * prev.z_pres).to_event(1),
                             infer=infer_dict)

        sample_mask = z_pres if self.use_masking else torch.tensor(1.0)

        z_where = pyro.sample('z_where_{}'.format(t),
                              dist.Normal(z_where_loc + self.z_where_loc_prior,
                                          z_where_scale * self.z_where_scale_prior)
                                  .mask(sample_mask)
                                  .to_event(1))

        # Figure 2 of [1] shows x_att depending on z_where and h,
        # rather than z_where and x as here, but I think this is
        # correct.
        x_att = image_to_window(z_where, self.window_size, self.x_size, inputs['raw'])

        # Encode attention windows.
        z_what_loc, z_what_scale = self.encode(x_att)

        z_what = pyro.sample('z_what_{}'.format(t),
                             dist.Normal(z_what_loc, z_what_scale)
                                 .mask(sample_mask)
                                 .to_event(1))
        return GuideState(h=h, c=c, bl_h=bl_h, bl_c=bl_c, z_pres=z_pres, z_where=z_where, z_what=z_what)

    def baseline_step(self, prev, inputs):
        if not self.use_baselines:
            return dict(), None, None

        # Prevent gradients flowing back from baseline loss to
        # inference net by detaching from graph here.
        rnn_input = torch.cat((inputs['bl_embed'],
                               prev.z_where.detach(),
                               prev.z_what.detach(),
                               prev.z_pres.detach()), 1)
        bl_h, bl_c = self.bl_rnn(rnn_input, (prev.bl_h, prev.bl_c))
        bl_value = self.bl_predict(bl_h)

        # Zero out values for finished data points. This avoids adding
        # superfluous terms to the loss.
        if self.use_masking:
            bl_value = bl_value * prev.z_pres

        # The value that the baseline net is estimating can be very
        # large. An option to scale the nets output is provided
        # to make it easier for the net to output values of this
        # scale.
        if self.baseline_scalar is not None:
            bl_value = bl_value * self.baseline_scalar

        infer_dict = dict(baseline=dict(baseline_value=bl_value.squeeze(-1)))
        return infer_dict, bl_h, bl_c


# Spatial transformer helpers.

expansion_indices = torch.LongTensor([1, 0, 2, 0, 1, 3])


def expand_z_where(z_where):
    # Take a batch of three-vectors, and massages them into a batch of
    # 2x3 matrices with elements like so:
    # [s,x,y] -> [[s,0,x],
    #             [0,s,y]]
    n = z_where.size(0)
    out = torch.cat((z_where.new_zeros(n, 1), z_where), 1)
    ix = expansion_indices
    if z_where.is_cuda:
        ix = ix.cuda()
    out = torch.index_select(out, 1, ix)
    out = out.view(n, 2, 3)
    return out


# Scaling by `1/scale` here is unsatisfactory, as `scale` could be
# zero.
def z_where_inv(z_where):
    # Take a batch of z_where vectors, and compute their "inverse".
    # That is, for each row compute:
    # [s,x,y] -> [1/s,-x/s,-y/s]
    # These are the parameters required to perform the inverse of the
    # spatial transform performed in the generative model.
    n = z_where.size(0)
    out = torch.cat((z_where.new_ones(n, 1), -z_where[:, 1:]), 1)
    # Divide all entries by the scale.
    out = out / z_where[:, 0:1]
    return out


def window_to_image(z_where, window_size, image_size, windows):
    n = windows.size(0)
    assert windows.size(1) == window_size ** 2, 'Size mismatch.'
    theta = expand_z_where(z_where)
    grid = F.affine_grid(theta, torch.Size((n, 1, image_size, image_size)))
    out = F.grid_sample(windows.view(n, 1, window_size, window_size), grid)
    return out.view(n, image_size, image_size)


def image_to_window(z_where, window_size, image_size, images):
    n = images.size(0)
    assert images.size(1) == images.size(2) == image_size, 'Size mismatch.'
    theta_inv = expand_z_where(z_where_inv(z_where))
    grid = F.affine_grid(theta_inv, torch.Size((n, 1, window_size, window_size)))
    out = F.grid_sample(images.view(n, 1, image_size, image_size), grid)
    return out.view(n, -1)


# Helper to expand parameters to the size of the mini-batch. I would
# like to remove this and just write `t.expand(n, -1)` inline, but the
# `-1` argument of `expand` doesn't seem to work with PyTorch 0.2.0.
def batch_expand(t, n):
    return t.expand(n, t.size(1))
