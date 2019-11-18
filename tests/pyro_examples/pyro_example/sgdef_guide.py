#########
# guide #
#########
# init params
alpha_init = 0.5
mean_init = 0.0
sigma_init = 0.1
softplus = nn.Softplus()

# guide
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
