from kalfa.registration import pack

lego = pack(__name__)


lego("/transform/kalfa/filter", "table:filter_rows", alias="filter", partial=True,
     description="Keep the rows a pandas query selects; a bare string in data.transform is this call; a stream "
                 "applies it chunk by chunk and a Dataset source takes field equality queries")
lego("/transform/kalfa/derive", "table:derive", alias="derive", partial=True, needs_table=True,
     description="A new column from a pandas eval expression over the frame (log10(x), a > b, a + b)")
lego("/transform/kalfa/rename", "table:rename", alias="rename", partial=True, needs_table=True,
     description="Rename the columns a regular expression matches, pattern to the replacement, backreferences allowed "
                 "(cms_(.*)_Z_score to z_\\1)")
lego("/transform/kalfa/astype", "table:astype", alias="astype", partial=True, needs_table=True,
     description="Cast columns to dtypes, {column: dtype}")
lego("/transform/kalfa/drop", "table:drop", alias="drop", partial=True, needs_table=True,
     description="Drop columns from the frame before anything reads them")
