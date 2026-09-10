from torch import nn

from kalfa.registration import lego


class LastStep(nn.Module):
    def forward(self, value):
        return value[:, -1, :]


@lego("/layer/torch/last_step", alias="last_step", description="The last step of a sequence")
def last_step():
    return LastStep()
