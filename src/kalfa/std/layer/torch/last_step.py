from torch import nn

from kalfa.registration import lego


@lego("/layer/torch/last_step", alias="last_step", description="The last step of a sequence")
class LastStep(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, value):
        return value[:, -1, :]
