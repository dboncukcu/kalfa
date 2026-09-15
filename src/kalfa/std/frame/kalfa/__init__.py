from kalfa.registration import pack

lego = pack(__name__)


lego("/frame/kalfa/group_statistic", "statistics:GroupStatistic", alias="group_statistic", needs_table=True,
     refs={"by": "column"},
     description="A statistic of a column per group, learned on the train set and mapped onto every set as a new "
                 "column (name, or <column>_<statistic>_by_<by>); a group the train set never saw takes the statistic "
                 "over the whole train set")
lego("/frame/kalfa/target_encoding", "statistics:TargetEncoding", alias="target_encoding", needs_table=True,
     refs={"column": "column"},
     description="The train mean of the target per category of a column, smoothed toward the overall mean by "
                 "smoothing pseudo counts, as a new column (name, or <column>_target); a category the train set never "
                 "saw takes the overall mean")
