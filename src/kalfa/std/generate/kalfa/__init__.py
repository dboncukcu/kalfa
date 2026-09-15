from kalfa.registration import pack

lego = pack(__name__)


lego("/generate/kalfa/gan_sampler", "samplers:gan_sampler", partial=True, refs={"model": "model"}, alias="gan_sampler",
     description="n samples of a generator from latent noise; conditional samples cycle through n_classes")
lego("/generate/kalfa/ddpm_sampler", "samplers:ddpm_sampler", partial=True,
     refs={"model": "model", "schedule": "schedule"}, alias="ddpm_sampler",
     description="n samples by the reverse diffusion of the noise schedule from pure noise")
lego("/generate/kalfa/lm_sampler", "samplers:lm_sampler", partial=True, refs={"model": "model"}, alias="lm_sampler",
     description="Autoregressive text from a prompt with the record's tokenizer; temperature scales the logits, the "
                 "window is context or the model's seq_len; the model's last layer has one logit per vocabulary entry "
                 "(vocab_size)")
