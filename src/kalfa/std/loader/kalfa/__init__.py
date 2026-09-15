from kalfa.registration import pack

lego = pack(__name__)


lego("/loader/kalfa/torch", "torch:torch_loader",
     description="torch DataLoader over a dataset: size batches shuffled for the train set, eval_size batches in "
                 "order for the other sets, the whole set as one batch without a size; drop_last auto drops the last "
                 "train batch only when it would hold one row; balanced puts a class balancing sampler over the "
                 "single target field; every worker is seeded from the torch seed on its own; a stream dataset "
                 "shuffles through buffer rows and takes no sampler or workers")
