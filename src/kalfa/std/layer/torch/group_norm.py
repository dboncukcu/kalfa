from torch import nn

from kalfa.registration import lego


@lego("/layer/torch/group_norm", alias="group_norm",
      description="torch.nn.GroupNorm: num_groups groups over num_channels channels")
def group_norm(num_groups, num_channels, eps=1e-5, affine=True):
    return nn.GroupNorm(int(num_groups), int(num_channels), eps=float(eps), affine=bool(affine))
