import torch

from kalfa.registration import lego
from kalfa.std.generate.base import pick_model


def context_of(net, context):
    """The window of tokens the model sees: ``context`` when given, else the seq_len a module of the model declares."""
    if context is not None:
        return int(context)
    for module in net.modules():
        if hasattr(module, "seq_len"):
            return int(module.seq_len)
    return None


@lego("/generate/kalfa/lm_sampler", partial=True, refs={"model": "model"}, alias="lm_sampler",
      description="Autoregressive text from a prompt with the record's tokenizer; temperature scales the "
                  "logits, the window is context or the model's seq_len; the model's last layer has one "
                  "logit per vocabulary entry (vocab_size)")
def lm_sampler(models, prep, rng, model, prompt, max_new_tokens=100, temperature=1.0, context=None):
    net = pick_model(models, model)
    net.eval()
    tokenizer = prep.tokenizer() if prep is not None else None
    if tokenizer is None:
        raise ValueError("lm_sampler needs a fitted tokenizer among the preprocessors")
    device = next(iter(net.parameters()), torch.zeros(1)).device
    ids = torch.as_tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).reshape(1, -1)
    limit = context_of(net, context)
    with torch.no_grad():
        for _ in range(int(max_new_tokens)):
            window = ids[:, -int(limit):] if limit else ids
            logits = net(window)[:, -1, :].float() / max(float(temperature), 1e-6)
            probabilities = torch.softmax(logits, dim=-1)
            pick = torch.multinomial(probabilities, 1, generator=rng) if rng is not None \
                else torch.multinomial(probabilities, 1)
            ids = torch.cat([ids, pick], dim=1)
    return tokenizer.decode(ids[0].cpu().numpy())
