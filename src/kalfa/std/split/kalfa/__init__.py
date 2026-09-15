from kalfa.registration import pack


lego = pack(__name__)


lego("/split/kalfa/random", "splits:random_split", returns=["train", "valid", "test"], alias="random_split",
     sizes="/lego/kalfa/ratio_sizes", needs_table=True,
     description="Shuffle the rows with a seed and cut them by ratios into train, valid and test; the short form of a "
                 "split without a uri")
lego("/split/kalfa/sequential", "splits:sequential", returns=["train", "valid", "test"], refs={"group": "column"},
     alias="sequential", sizes="/lego/kalfa/ratio_sizes",
     description="Cut the rows in their order by ratios; with a group column every group is cut on its own")
lego("/split/kalfa/kfold", "splits:kfold", returns=["train", "valid", "test"], alias="kfold",
     sizes="/lego/kalfa/kfold_sizes", needs_table=True,
     description="k folds of a seeded permutation: the held out fold is the test set, val carves the valid set from "
                 "the rest; without val there is no valid set")
lego("/split/kalfa/given", "splits:given", returns=["train", "valid", "test"], alias="given",
     sizes="/lego/kalfa/given_sizes",
     description="The source is the train set; valid and test come from the given paths, read like the source (a "
                 "missing path means no set)")
lego("/split/kalfa/prepared", "splits:prepared_split", returns=["train", "valid", "test"],
     sizes="/lego/kalfa/prepared_sizes",
     description="The split kalfa prepare recorded: the sets a prepared frame is marked with, or the positions of a "
                 "Dataset source's items per set")
