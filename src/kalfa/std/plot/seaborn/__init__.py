from kalfa.registration import pack

lego = pack(__name__)


lego("/plot/seaborn/kde", "distributions:kde", partial=True, needs=["train_loader"], alias="kde", requires="seaborn",
     refs={"x": "column", "y": "column", "hue": "column"},
     description="seaborn's kernel density of one column of a set, or of two as contours; skipped with a warning when "
                 "seaborn is not installed")
lego("/plot/seaborn/pairplot", "distributions:pairplot", partial=True, needs=["train_loader"], alias="pairplot",
     requires="seaborn",
     description="seaborn's pairwise grid of a few columns of a set, hue colouring the points by a column; skipped "
                 "with a warning when seaborn is not installed")
lego("/plot/seaborn/violin", "distributions:violin", partial=True, needs=["train_loader"], alias="violin",
     requires="seaborn", refs={"value": "column", "group": "column"},
     description="seaborn's violin of one column of a set, split by a grouping column when one is named; skipped with "
                 "a warning when seaborn is not installed")
