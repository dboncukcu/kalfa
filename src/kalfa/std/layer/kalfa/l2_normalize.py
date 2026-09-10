from torch import nn

from kalfa.registration import lego


class L2Normalize(nn.Module):
    def __init__(self, eps):
        super().__init__()
        self.eps = float(eps)

    def forward(self, values):
        flat = values.reshape(values.shape[0], -1)
        return flat / flat.norm(dim=1, keepdim=True).clamp_min(self.eps)


@lego("/layer/kalfa/l2_normalize", alias="l2_normalize",
      description="Divide every sample by the L2 norm of its own feature vector (sklearn's Normalizer as a "
                  "layer: it reads the whole vector, so it belongs to the model, not to a column chain)")
def l2_normalize(eps=1e-12):
    return L2Normalize(eps)
