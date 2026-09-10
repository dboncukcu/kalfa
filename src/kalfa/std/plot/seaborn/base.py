def sampled(table, sample, seed=0):
    if sample and len(table) > int(sample):
        return table.sample(int(sample), random_state=int(seed))
    return table
