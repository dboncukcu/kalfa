from kalfa.registration import pack

lego = pack(__name__)


lego("/plot/torchview/architecture", "architecture:architecture", partial=True, alias="torchview", requires="torchview",
     description="torchview's drawing of every report model the batch feeds, under plots/<name>_<model>.png; it needs "
                 "the graphviz dot binary and runs on the device of the run, so a composite keeps its referenced "
                 "models with it")
