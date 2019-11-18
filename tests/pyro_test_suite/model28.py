# test_nested_plate_plate_dim_error_4
# shape: [1]
p = torch.tensor([0.5], requires_grad=True)
with pyro.plate("plate_outer", 10, 5) as ind_outer:
    # shape: [5,1]+[1] = [5,1,1] --bcast--> [5,1,5]
    # pyro.sample("x", Bernoulli(p).expand_by([len(ind_outer), 1]))
    pyro.sample("x", Bernoulli(p).expand_by([5, 1]))
    with pyro.plate("plate_inner", 11, 6) as ind_inner:
        # shape: [6]+[1] = [6,1] --bcast--> [6,5]
        # pyro.sample("y", Bernoulli(p).expand_by([len(ind_inner)]))
        # shape: [5,5]+[1] = [5,5,1] --bcast--> ERROR because of mismatching sizes in dims=-2: 5 vs 6.
        # pyro.sample("z", Bernoulli(p).expand_by([len(ind_outer), len(ind_outer)]))  # error here
        pyro.sample("y", Bernoulli(p).expand_by([6]))
        pyro.sample("z", Bernoulli(p).expand_by([5, 5]))  # error here
