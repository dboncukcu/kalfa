import numpy
import torch

from kalfa.std.calibrate.base import Calibration
from kalfa.std.common.runtime import call_model, named_outputs, resolve_model


class Threshold(Calibration):
    def __init__(self, set="valid", quantile=0.95, output=None):
        if not 0.0 <= float(quantile) <= 1.0:
            raise ValueError(f"threshold.quantile must be between 0 and 1, got {quantile!r}")
        self.set = set
        self.quantile = float(quantile)
        self.output = output
        self.threshold = None
        self.wire = None

    def fit(self, models, loaders, prep, device, predicts):
        loader = (loaders or {}).get(self.set)
        if loader is None or loader.dataset.size() == 0:
            raise ValueError(f"threshold reads the {self.set!r} set, which the run does not have")
        model = resolve_model(predicts, models)
        model.eval()
        scores = []
        with torch.no_grad():
            for batch in loader:
                outputs = named_outputs(model, call_model(model, device.move(batch)))
                self.wire = self.output or next(iter(outputs))
                if self.wire not in outputs:
                    raise KeyError(f"threshold names output {self.wire!r}; the model has {sorted(outputs)}")
                values = outputs[self.wire].detach().cpu().numpy()
                scores.append(values.reshape(len(values), -1)[:, 0])
        self.threshold = float(numpy.quantile(numpy.concatenate(scores), self.quantile))

    def apply(self, table):
        column = f"raw_{self.wire}"
        if self.threshold is None or column not in table.columns:
            return table
        return table.assign(**{f"flag_{self.wire}": table[column].to_numpy() > self.threshold})

    def note(self):
        return {"set": self.set, "quantile": self.quantile, "output": self.wire, "threshold": self.threshold}
