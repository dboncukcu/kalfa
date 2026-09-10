from pathlib import Path

import pandas

from kalfa.registration import lego
from kalfa.std.common.files import atomic
from kalfa.std.common.log import logger_for
from kalfa.std.common.prediction import prediction_table
from kalfa.std.common.runtime import resolve_model


logger_after = logger_for("after")


@lego("/lego/kalfa/predict", returns="predictions", bus=["record", "device"],
      description="Predict a set with the report model, invert the target chain, apply the fitted "
                  "calibrations, write predictions.parquet for the test set and predictions_<set>.parquet for "
                  "another")
def predict(models, composites, loader, prep, predicts, set, target_map=None, calibrations=None, record=None,
            device=None):
    if loader is None or loader.dataset.size() == 0 or predicts is None:
        return pandas.DataFrame()
    model = resolve_model(predicts, models, composites)
    table = prediction_table(model, loader, prep, loader.dataset, device, target_map)
    for item in (calibrations or {}).values():
        table = item.apply(table)
    if len(table) == 0:
        return pandas.DataFrame()
    if record is not None:
        target = Path(record)
        target.mkdir(parents=True, exist_ok=True)
        name = "predictions.parquet" if set == "test" else f"predictions_{set}.parquet"
        with atomic(target / name) as temporary:
            table.to_parquet(temporary, index=False)
        logger_after.info(f"{name}: {len(table)} rows")
    return table
