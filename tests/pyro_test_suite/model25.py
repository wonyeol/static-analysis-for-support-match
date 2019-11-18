# test_nested_plate_plate_dim_error_1
# shape: [1]
p = torch.tensor([0.5], requires_grad=True)
with pyro.plate("plate_outer", 10, 5) as ind_outer:
    # shape: [5]+[1] = [5,1] --bcast--> [5,5]
    # pyro.sample("x", Bernoulli(p).expand_by([len(ind_outer)]))  # error here
    pyro.sample("x", Bernoulli(p).expand_by([5]))  # error here <-- wy: WRONG! No error here!
    with pyro.plate("plate_inner", 11, 6) as ind_inner:
        # shape: [6]+[1] = [6,1] --bcast--> [6,5]
        # pyro.sample("y", Bernoulli(p).expand_by([len(ind_inner)]))
        # shape: [5,6]+[1] = [5,6,1] --bcast--> [5,6,5]
        # pyro.sample("z", Bernoulli(p).expand_by([len(ind_outer), len(ind_inner)]))
        pyro.sample("y", Bernoulli(p).expand_by([6]))
        pyro.sample("z", Bernoulli(p).expand_by([5, 6]))
