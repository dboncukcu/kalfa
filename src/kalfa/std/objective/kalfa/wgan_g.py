from kalfa.registration import lego
from kalfa.std.objective.base import call_with, input_of, labels_of, latent_noise


@lego("/objective/kalfa/wgan_g", partial=True, refs={"generator": "model", "critic": "model"},
      alias="wgan_g", description="WGAN generator loss: minus the critic's mean score of generated samples")
def wgan_g(models, batch, generator, critic, latent, conditional=False, rng=None):
    real = input_of(models[critic], batch)
    labels = labels_of(batch, conditional)
    noise = latent_noise(real.shape[0], latent, real, rng)
    fake = call_with(models[generator], noise, labels)
    return -call_with(models[critic], fake, labels).mean()
