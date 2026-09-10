import numpy
import torch

from kalfa.registration import lego
from kalfa.std.common.figure import Figure
from kalfa.std.common.runtime import named_outputs, resolve_model
from kalfa.std.plot.base import bars, first_set


def feature_batch(loader, model, prep, sample):
    wires = list(model.inputs)
    device = next(iter(model.parameters()), torch.zeros(1)).device
    features, targets, taken = [], [], 0
    names = list(prep.targets) if prep is not None else []
    for batch in loader:
        if wires[0] not in batch or not names or names[0] not in batch:
            return None, None
        features.append(batch[wires[0]])
        targets.append(batch[names[0]])
        taken += len(features[-1])
        if taken >= int(sample):
            break
    if not features:
        return None, None
    matrix = torch.cat(features)[:int(sample)].to(device)
    truth = torch.cat(targets)[:int(sample)].to(device).reshape(len(matrix), -1)
    if matrix.ndim != 2:
        return None, None
    return matrix, truth


def scored(model, matrix, truth, wire):
    with torch.no_grad():
        outputs = named_outputs(model, model(matrix))
    guess = outputs[wire] if wire in outputs else next(iter(outputs.values()))
    guess = guess.reshape(len(matrix), -1)[:, :truth.shape[1]]
    spread = float(((truth - truth.mean(dim=0)) ** 2).sum())
    if spread <= 0:
        return float("nan")
    return 1.0 - float(((guess - truth) ** 2).sum()) / spread


def importances(model, matrix, truth, wire, base, repeats, seed):
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    means, deviations = [], []
    for column in range(matrix.shape[1]):
        scores = []
        for _ in range(int(repeats)):
            shuffled = matrix.clone()
            order = torch.randperm(len(matrix), generator=generator).to(matrix.device)
            shuffled[:, column] = matrix[order, column]
            scores.append(base - scored(model, shuffled, truth, wire))
        means.append(float(numpy.mean(scores)))
        deviations.append(float(numpy.std(scores)))
    return means, deviations


@lego("/plot/kalfa/permutation_importance", partial=True, alias="permutation_importance",
      description="The drop in R2 when one feature column is shuffled, the largest first; the model runs "
                  "again for every feature and every repeat, so sample bounds the cost")
def permutation_importance(predictions, history, models, record, loaders=None, prep=None, predicts=None, sets=None,
                           repeats=3, sample=20000, top=25, output=None, groups=None, seed=0, name=None, figures=None):
    figures = figures or Figure()
    loader = (loaders or {}).get(first_set(sets, "test"))
    if loader is None or prep is None or predicts is None:
        return None
    model = resolve_model(predicts, models)
    model.eval()
    matrix, truth = feature_batch(loader, model, prep, sample)
    if matrix is None or matrix.shape[1] != len(prep.features):
        return None
    wire = output or ""
    base = scored(model, matrix, truth, wire)
    if not numpy.isfinite(base):
        return None
    means, deviations = importances(model, matrix, truth, wire, base, repeats, seed)
    order = numpy.argsort(means)[::-1][:int(top)][::-1]
    names = [prep.features[position] for position in order]
    drawing, axes = figures.sized(figures.width_of(9.5), 0.34 * len(names) + 2.0)
    axis = axes[0][0]
    bars(figures, axis, names, [means[position] for position in order], groups,
         [deviations[position] for position in order])
    figures.label(axis, "Permutation importance", "drop in R2 when the feature is shuffled", None,
                 note=f"R2 = {base:.4f} on {len(matrix):,} points, {int(repeats)} repeats")
    figures.save(drawing, record, name or "permutation_importance")
    return None
