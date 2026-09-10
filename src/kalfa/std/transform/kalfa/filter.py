from kalfa.registration import lego


@lego("/transform/kalfa/filter", alias="filter", partial=True,
      description="Keep the rows a pandas query selects; a bare string in data.transform is this call; a stream "
                  "applies it chunk by chunk and a Dataset source takes field equality queries")
def filter_rows(df, query):
    return df.query(query)
