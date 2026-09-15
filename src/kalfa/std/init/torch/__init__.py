from kalfa.registration import pack


lego = pack(__name__)


lego("/init/torch/zeros", "initializers:zeros", alias="zeros", description="Zero initialization")
lego("/init/torch/normal", "initializers:normal", alias="normal", description="Normal initialization with std and mean")
lego("/init/torch/xavier", "initializers:xavier", alias="xavier", description="Xavier uniform initialization")
lego("/init/torch/kaiming", "initializers:kaiming", alias="kaiming", description="Kaiming normal initialization")
lego("/init/torch/kaiming_uniform", "initializers:kaiming_uniform", alias="kaiming_uniform",
     description="Kaiming uniform initialization")
