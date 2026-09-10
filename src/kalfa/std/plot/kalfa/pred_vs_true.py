from kalfa.registration import lego
from kalfa.std.common import figure
from kalfa.std.plot.base import panel_title, prediction_pairs, r2_of


@lego("/plot/kalfa/pred_vs_true", partial=True, alias="pred_vs_true",
      description="Predicted against true values of the test set, one panel per predicted field with its R2, "
                  "as a hexbin density over many points and a scatter over few; the panel is titled with the "
                  "field name, plus the output wire when two outputs predict the same field")
def pred_vs_true(predictions, history, models, record, name=None, columns=4, kind="auto", gridsize=70):
    pairs = prediction_pairs(predictions)
    if not pairs:
        return None
    width = max(1, min(int(columns or 4), len(pairs)))
    rows = -(-len(pairs) // width)
    drawing, axes = figure.grid(rows, width, width=5.4, height=4.6)
    panels = [axis for row in axes for axis in row]
    paired = [field for _, field in pairs]
    for axis, (pred, target) in zip(panels, pairs):
        true, guess = figure.finite(predictions[target].to_numpy(), predictions[pred].to_numpy())
        if not len(true):
            axis.axis("off")
            continue
        dense = kind == "hexbin" or (kind == "auto" and len(true) >= 2000)
        if dense:
            figure.density(drawing, axis, true, guess, gridsize)
        else:
            axis.scatter(true, guess, s=12, alpha=0.5, color=figure.CATEGORICAL[0], edgecolors="none")
        low = float(min(true.min(), guess.min()))
        high = float(max(true.max(), guess.max()))
        axis.plot([low, high], [low, high], color=figure.CATEGORICAL[1], linewidth=2, label="perfect")
        axis.legend(loc="upper left")
        figure.label(axis, panel_title(pred, target, paired), "true", "predicted",
                     note=f"R2 = {r2_of(true, guess):.4f} on {len(true):,} points")
    for axis in panels[len(pairs):]:
        axis.axis("off")
    figure.save(drawing, record, name or "pred_vs_true")
    return None
