from kalfa.registration import pack

lego = pack(__name__)


lego("/feed/kalfa/next_token", "next_token:next_token", alias="next_token",
     description="input_ids and targets windows of seq_len tokens over the set's token stream")

lego("/feed/kalfa/table", "table:table", alias="table",
     description="Feature columns as one tensor x and target fields by name; Dataset fields by name")

lego("/feed/kalfa/window", "window:window", alias="window", refs={"group": "column"}, needs_table=True,
     description="Windows of size steps and the next horizon steps of the targets; context takes the tail of the "
                 "previous set at the split boundary, group keeps series apart")
