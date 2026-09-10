from kalfa.registration import lego
from kalfa.std.calibrate.base import write_calibrations
from kalfa.std.common.device import Device
from kalfa.std.common.log import logger_for


logger_after = logger_for("after.calibrate")


@lego("/lego/kalfa/calibrate", returns="calibrations", bus=["record", "device"],
      description="Fit every calibration of the calibrate section on the report models and the sets, in order, "
                  "and keep them in the record under fitted/calibrate; predict applies them to its table")
def calibrate(models, composites, prep, loaders, calibrations, predicts, record=None, device=None):
    device = device or Device.cpu()
    everything = {**dict(composites or {}), **dict(models or {})}
    fitted = {}
    for name, item in (calibrations or {}).items():
        item.fit(everything, loaders, prep, device, predicts)
        fitted[name] = item
        logger_after.info(f"{name}: {item.note()}")
    if record is not None:
        write_calibrations(fitted, record)
    return fitted
