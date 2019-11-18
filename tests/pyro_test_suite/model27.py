# test_nested_plate_plate_dim_error_3
# shape: [1]
p = torch.tensor([0.5], requires_grad=True)
with pyro.plate("plate_outer", 10, 5) as ind_outer:
    # shape: [5,1]+[1] = [5,1,1] --bcast--> [5,1,5]
    # pyro.sample("x", Bernoulli(p).expand_by([len(ind_outer), 1]))
    pyro.sample("x", Bernoulli(p).expand_by([5, 1]))
    with pyro.plate("plate_inner", 11, 6) as ind_inner:
        # shape: [6]+[1] = [6,1] --bcast--> [6,5]
        # pyro.sample("y", Bernoulli(p).expand_by([len(ind_inner)]))
        # shape: [6,1]+[1] = [6,1,1] --bcast--> [6,6,5]
        # pyro.sample("z", Bernoulli(p).expand_by([len(ind_inner), 1]))  # error here
        pyro.sample("y", Bernoulli(p).expand_by([6]))
        pyro.sample("z", Bernoulli(p).expand_by([6, 1]))  # error here <-- wy: WRONG! No error here!
