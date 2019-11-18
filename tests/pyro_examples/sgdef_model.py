#########
# model #
#########
# hyperparams
alpha_z = torch.tensor(0.1)
beta_z = torch.tensor(0.1)
alpha_w = torch.tensor(0.1)
beta_w = torch.tensor(0.3)

# model
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
