import torch

from kalfa.std.objective.base import input_of


def ntxent(models, batch, model, temperature=0.5):
    net = models[model]
    views = input_of(net, batch)
    if views.ndim < 5:
        raise ValueError("ntxent needs two views per image: put two_views in the field's chain (sets: [train])")
    count = views.shape[0]
    flat = views.flatten(0, 1)
    z = torch.nn.functional.normalize(net(flat).float(), dim=1)
    similarity = z @ z.t() / float(temperature)
    similarity = similarity.masked_fill(torch.eye(2 * count, device=z.device, dtype=torch.bool), float("-inf"))
    positions = torch.arange(2 * count, device=z.device)
    partner = positions ^ 1
    return torch.nn.functional.cross_entropy(similarity, partner)
