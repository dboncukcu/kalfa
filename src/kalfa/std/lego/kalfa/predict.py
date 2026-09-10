from pathlib import Path

from kalfa.registration import lego
from kalfa.std.common.log import logger_for
from kalfa.std.common.prediction import prediction_table
from kalfa.std.common.runtime import resolve_model
from kalfa.std.feed.base import sized


logger_after = logger_for("after")


@lego("/lego/kalfa/predict", returns="predictions", bus=["record", "device"],
      description="Predict the test set with the report model, invert the target chain, "
                  "write predictions.parquet")
def predict(models, composites, loader, prep, predicts, set, target_map=None, record=None, device=None):
    import pandas

    if loader is None or sized(loader.dataset) == 0 or predicts is None:
        return pandas.DataFrame()
    model = resolve_model(predicts, models, composites)
    table = prediction_table(model, loader, prep, loader.dataset, device, target_map)
    if len(table) == 0:
        return pandas.DataFrame()
    if record is not None:
        target = Path(record)
        target.mkdir(parents=True, exist_ok=True)
        table.to_parquet(target / "predictions.parquet", index=False)
        logger_after.info(f"predictions.parquet: {len(table)} rows")
    return table
