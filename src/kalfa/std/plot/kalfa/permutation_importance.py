import numpy

from kalfa.registration import lego
from kalfa.std.common import figure
from kalfa.std.plot.base import bars, first_set


def feature_batch(loader, model, prep, sample):
    """One matrix of the model's first input wire and the target tensor of the pass, up to sample rows."""
    import torch

    from kalfa.std.common.runtime import model_inputs, named_outputs

    wires = model_inputs(model)
    device = next(iter(model.parameters()), torch.zeros(1)).device
    features, targets, taken = [], [], 0
    names = list(prep.targets) if prep is not None else []
    for batch in loader:
        if wires[0] not in batch or not names or names[0] not in batch:
            return None, None, None
        features.append(batch[wires[0]])
        targets.append(batch[names[0]])
        taken += len(features[-1])
        if taken >= int(sample):
            break
    if not features:
        return None, None, None
    matrix = torch.cat(features)[:int(sample)].to(device)
    truth = torch.cat(targets)[:int(sample)].to(device).reshape(len(matrix), -1)
    if matrix.ndim != 2:
        return None, None, None
    return matrix, truth, named_outputs


def scored(model, matrix, truth, wire, named_outputs):
    import torch

    with torch.no_grad():
        outputs = named_outputs(model, model(matrix))
    guess = outputs[wire] if wire in outputs else next(iter(outputs.values()))
    guess = guess.reshape(len(matrix), -1)[:, :truth.shape[1]]
    spread = float(((truth - truth.mean(dim=0)) ** 2).sum())
    if spread <= 0:
        return float("nan")
    return 1.0 - float(((guess - truth) ** 2).sum()) / spread


@lego("/plot/kalfa/permutation_importance", partial=True, alias="permutation_importance",
      description="The drop in R2 when one feature column is shuffled, the largest first; the model runs "
                  "again for every feature and every repeat, so sample bounds the cost")
def permutation_importance(predictions, history, models, record, loaders=None, prep=None, predicts=None, sets=None,
                           repeats=3, sample=20000, top=25, output=None, groups=None, seed=0, name=None):
    import torch

    from kalfa.std.common.runtime import resolve_model

    loader = (loaders or {}).get(first_set(sets, "test"))
    if loader is None or prep is None or predicts is None:
        return None
    model = resolve_model(predicts, models)
    model.eval()
    matrix, truth, tools = feature_batch(loader, model, prep, sample)
    if matrix is None or matrix.shape[1] != len(prep.features):
        return None
    wire = output or ""
    base = scored(model, matrix, truth, wire, tools)
    if not numpy.isfinite(base):
        return None
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    means, deviations = [], []
    for column in range(matrix.shape[1]):
        scores = []
        for _ in range(int(repeats)):
            shuffled = matrix.clone()
            order = torch.randperm(len(matrix), generator=generator).to(matrix.device)
            shuffled[:, column] = matrix[order, column]
            scores.append(base - scored(model, shuffled, truth, wire, tools))
        means.append(float(numpy.mean(scores)))
        deviations.append(float(numpy.std(scores)))
    order = numpy.argsort(means)[::-1][:int(top)][::-1]
    names = [prep.features[position] for position in order]
    values = [means[position] for position in order]
    spread = [deviations[position] for position in order]
    drawing, axes = figure.sized(figure.width_of(9.5), 0.34 * len(names) + 2.0)
    axis = axes[0][0]
    bars(axis, names, values, groups, spread)
    figure.label(axis, "Permutation importance", "drop in R2 when the feature is shuffled", None,
                 note=f"R2 = {base:.4f} on {len(matrix):,} points, {int(repeats)} repeats")
    figure.save(drawing, record, name or "permutation_importance")
    return None
