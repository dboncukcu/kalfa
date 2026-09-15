from kalfa.registration import pack

lego = pack(__name__)


lego("/device/kalfa/auto", "devices:auto", alias="auto", description="The first available device of cuda, mps, cpu")
lego("/device/kalfa/cpu", "devices:cpu", alias="cpu", description="The CPU")
lego("/device/kalfa/cuda", "devices:cuda", alias="cuda",
     description="The cuda device with the given index; an error when cuda or that index is not available")
lego("/device/kalfa/mps", "devices:mps", alias="mps",
     description="The Apple mps device; an error when it is not available")
