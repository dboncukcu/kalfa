from kalfa.registration import pack

lego = pack(__name__)


lego("/objective/kalfa/wgan_g", "adversarial:wgan_g", partial=True, refs={"generator": "model", "critic": "model"},
     alias="wgan_g", description="WGAN generator loss: minus the critic's mean score of generated samples")
lego("/objective/kalfa/wgan_gp_d", "adversarial:wgan_gp_d", partial=True, needs_grad=True,
     refs={"generator": "model", "critic": "model"}, alias="wgan_gp_d",
     description="WGAN critic loss with a gradient penalty on interpolates; the label conditions both models")

lego("/objective/kalfa/ddpm", "ddpm:ddpm", partial=True, refs={"model": "model", "schedule": "schedule"}, alias="ddpm",
     description="DDPM noise prediction loss: a random time step and noise per sample (rng), the model predicts the "
                 "noise of the noised input")

lego("/objective/kalfa/distill", "distill:distill", partial=True, refs={"student": "model", "teacher": "model"},
     alias="distill",
     description="Knowledge distillation: alpha * KL(teacher || student) at temperature T (times T squared) plus (1 - "
                 "alpha) * cross entropy of the student; returns loss, ce and kl")

lego("/objective/kalfa/ntxent", "ntxent:ntxent", partial=True, refs={"model": "model"}, alias="ntxent",
     description="NT-Xent contrastive loss over the two views of every image in the batch")

lego("/objective/kalfa/vae", "vae:vae", partial=True,
     refs={"encoder": "model", "decoder": "model", "recon": "criterion", "kl_schedule": "schedule"}, alias="vae",
     description="VAE loss: w_rec * recon(decoder(z), x) + kl_schedule(step) * KL, z sampled from the encoder's mu "
                 "and logvar; returns loss, recon, kl and w_kl")

lego("/objective/kalfa/weighted_sum", "weighted_sum:weighted_sum", partial=True, refs={"terms": "loss"},
     alias="weighted_sum",
     description="The weighted sum of other losses definitions on the same batch: terms maps a losses name to its "
                 "weight; returns loss and every term")
