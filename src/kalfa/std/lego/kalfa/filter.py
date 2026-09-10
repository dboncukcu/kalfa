from kalfa.registration import lego


@lego("/lego/kalfa/filter",
      description="Keep the rows a pandas query selects; a Dataset source takes field equality queries")
def filter(df, query):
    return df.query(query)
