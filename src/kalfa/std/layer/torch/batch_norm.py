from torch import nn

from kalfa.registration import lego


@lego("/layer/torch/batch_norm", alias="batch_norm",
      description="torch.nn.BatchNorm over the feature axis, lazy in the number of features: dims 1 for (batch, "
                  "features) and sequences, 2 for images, 3 for volumes")
def batch_norm(dims=1, eps=1e-5, momentum=0.1, affine=True):
    kinds = {1: nn.LazyBatchNorm1d, 2: nn.LazyBatchNorm2d, 3: nn.LazyBatchNorm3d}
    if dims not in kinds:
        raise ValueError(f"batch_norm.dims must be 1, 2 or 3, got {dims!r}")
    return kinds[dims](eps=float(eps), momentum=float(momentum), affine=bool(affine))
