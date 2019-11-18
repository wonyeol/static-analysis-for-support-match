###################################
# vars used in both model & guide #
###################################
expansion_indices = torch.LongTensor([1, 0, 2, 0, 1, 3])
z_where_loc_prior =\
    nn.Parameter(torch.FloatTensor([3.0, 0.0, 0.0]), requires_grad=False)
z_where_scale_prior =\
    nn.Parameter(torch.FloatTensor([0.2, 1.0, 1.0]), requires_grad=False)

#########
# model #
#########
# nn's
decode_l1 = nn.Linear(50, 200)
decode_l2 = nn.Linear(200, 784)

# trial_probs (from z_pres_prior_p)
z_pres_prior = 0.01
#===== make_prior
k = z_pres_prior
u = 1 / (1 + k + (k*k) + (k*k*k))
p0 = 1 - u
p1 = 1 - (k * u) / p0
p2 = 1 - ((k*k) * u) / (p0 * p1)
trial_probs = torch.tensor([p0, p1, p2])
#===== make_prior

# model()
data = torch.reshape(data, [60000, 50, 50])

pyro.module("decode_l1", decode_l1)
pyro.module("decode_l2", decode_l2)

with pyro.plate('data', 60000, 64) as ix:
    # size = [64, 50, 50]
    batch = data[ix] 

    #================= prior
    state_x = torch.zeros([64, 50, 50])
    state_z_pres = torch.ones([64, 1])
    state_z_where = None

    z_pres = []
    z_where = []

    for t in range(3):
        #==================== prior_step
        # size = [64, 50, 50]
        prev_x = state_x
        # size = [64, 1]
        prev_z_pres = state_z_pres
        # size = None or [64, 3]
        prev_z_where = state_z_where

        # size = [64, 1]
        cur_z_pres =\
            pyro.sample('z_pres_{}'.format(t),
                        Bernoulli(trial_probs[t] * prev_z_pres)
                        .to_event(1))

        sample_mask = cur_z_pres
        # size = [64, 3]
        cur_z_where =\
            pyro.sample('z_where_{}'.format(t),
                        Normal(torch.Tensor.expand(z_where_loc_prior, [64, 3]),
                               torch.Tensor.expand(z_where_scale_prior, [64, 3]))
                        .mask(sample_mask)
                        .to_event(1))

        # size = [64, 50]
        cur_z_what =\
            pyro.sample('z_what_{}'.format(t),
                        Normal(torch.zeros([64, 50]),
                               torch.ones([64, 50]))
                        .mask(sample_mask)
                        .to_event(1))

        #===== decode
        # size = [64, 784]
        y_att = torch.sigmoid(decode_l2(F.relu(decode_l1(cur_z_what))) - 2.0)
        #===== decode

        #===== window_to_image
        windows = y_att

        #===== expand_z_where
        # size = [64, 4]
        out = torch.cat((torch.zeros(64, 1), cur_z_where), 1)
        # size = [64, 6]
        out = torch.index_select(out, 1, expansion_indices)
        # size = [64, 2, 3]
        out = torch.Tensor.view(out, [64, 2, 3])
        theta = out
        #===== expand_z_where
        # size = [64, 50, 50, 2]
        grid = F.affine_grid(theta, [64, 1, 50, 50])
        # size = [64, 1, 50, 50]
        out = F.grid_sample(torch.Tensor.view(windows, [64, 1, 28, 28]), grid)

        y = torch.Tensor.view(out, [64, 50, 50])
        #===== window_to_image

        # size = [64, 50, 50]
        cur_x = prev_x + (y * torch.Tensor.view(cur_z_pres, [64, 1, 1]))

        state_x = cur_x
        state_z_pres = cur_z_pres
        state_z_where = cur_z_where
        #==================== prior_step

        z_where.append(state_z_where)
        z_pres.append(state_z_pres)

    # size = [64, 50, 50]
    x = state_x
    #================== prior

    pyro.sample('obs',
                Normal(torch.Tensor.view(x, [64, 2500]),
                       (0.3 * torch.ones(64, 2500)))
                .to_event(1),
                obs=torch.Tensor.view(batch, [64, 2500]))
