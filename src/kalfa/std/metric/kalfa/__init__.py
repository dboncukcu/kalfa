from kalfa.registration import pack

lego = pack(__name__)


lego("/metric/kalfa/fid", "fid:Fid", state=True, refs={"model": "model"}, uses=["models", "batch"], alias="fid",
     description="Fréchet inception distance of n samples of the model against n real images of the set; conditional "
                 "samples use the batch labels")

lego("/metric/kalfa/perplexity", "perplexity:Perplexity", state=True, alias="perplexity",
     description="exp of the mean token cross entropy of the logits against the targets")

lego("/metric/kalfa/recon_error", "recon_error:ReconError", state=True, alias="recon_error",
     description="Mean per sample squared reconstruction error of the output against the target")

lego("/metric/kalfa/rmse", "rmse:Rmse", state=True, alias="rmse", description="Root mean squared error")

lego("/metric/kalfa/sample_writer", "sample_writer:SampleWriter", state=True, alias="sample_writer",
     refs={"sampler": "generate"}, uses=["models"],
     description="A metric that writes n samples per pass under samples/turn_<n> (png and pt, or txt) from the "
                 "sampler (sampler: generate takes the generate section) or the predicts model; it reports no value, "
                 "use every and sets to pace it")
