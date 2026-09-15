from kalfa.registration import pack


lego = pack(__name__)


lego("/source/kalfa/prepared", "prepared:prepared", returns="df", header="/lego/kalfa/prepared_header",
     description="The data kalfa prepare wrote: the sets of a table read back into one frame marked by set, or the "
                 "items of a Dataset source read from where they are with the split kept as positions")

lego("/source/kalfa/image_folder", "samples:image_folder", returns="df", alias="image_folder",
     header="/lego/kalfa/image_folder_header", samples=True,
     description="Images under root/<class>/ as a Dataset with fields image and label")
lego("/source/kalfa/text_lines", "samples:text_lines", returns="df", alias="text_lines",
     header="/lego/kalfa/text_lines_header", samples=True,
     description="The lines of a text file as a Dataset with the field text")

lego("/source/kalfa/parquet", "tables:parquet", returns="df", alias="parquet", header="/lego/kalfa/parquet_header",
     description="Read a parquet file into a DataFrame; columns lists the ones to read, the others stay on disk")
lego("/source/kalfa/csv", "tables:csv", returns="df", alias="csv", header="/lego/kalfa/csv_header",
     description="Read a CSV file into a DataFrame; columns lists the ones to read, the others stay on disk")
lego("/source/kalfa/parquet_stream", "tables:parquet_stream", returns="df", header="/lego/kalfa/parquet_header",
     stream=True,
     description="Read a parquet file in chunks (the lazy set): a stream the data legos filter, cut and fit without "
                 "loading the table")
lego("/source/kalfa/csv_stream", "tables:csv_stream", returns="df", header="/lego/kalfa/csv_header", stream=True,
     description="Read a CSV file in chunks (the lazy set)")
