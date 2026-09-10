import torch


def input_of(model, batch):
    wires = list(getattr(model, "inputs", []))
    if not wires or wires[0] not in batch:
        raise KeyError(f"the objective needs the model's first input wire in the batch; wires {wires}, "
                       f"batch {sorted(batch)}")
    return batch[wires[0]]


def latent_noise(count, latent, like, rng):
    return torch.randn((count, int(latent)), generator=rng, device=like.device, dtype=like.dtype) if rng is not None \
        else torch.randn((count, int(latent)), device=like.device, dtype=like.dtype)


def labels_of(batch, conditional):
    if not conditional:
        return None
    if "label" not in batch:
        raise KeyError("a conditional GAN reads the label field of the batch")
    return batch["label"]


def call_with(model, *parts):
    return model(*[part for part in parts if part is not None])
