from kalfa.records import stats


def read_predictions(directory):
    frame = stats.read_predictions(directory)
    return frame if frame is not None and len(frame) else None


def pred_column(output):
    return output if str(output).startswith("pred_") else f"pred_{output}"


def prediction_pairs(frame, plan, pairs=None):
    columns = [str(column) for column in frame.columns]
    if pairs:
        chosen = [(pred_column(output), str(target)) for output, target in dict(pairs).items()]
        return [(pred, target) for pred, target in chosen if pred in columns and target in columns]
    found = [(pair["pred"], pair["target"]) for pair in stats.prediction_pairs(frame, plan)]
    paired = {pred for pred, _ in found}
    preds = [column for column in columns if column.startswith("pred_") and column not in paired
             and stats.spread_member(column, columns) is None]
    targets = [name for name in stats.plan_targets(plan) if name in columns]
    if not targets:
        targets = [column for column in columns
                   if not column.startswith(("pred_", "raw_", "prob_")) and column not in ("row", "set")]
    if len(targets) == 1:
        found.extend((pred, targets[0]) for pred in preds)
        return found
    for pred in preds:
        output = pred[len("pred_"):]
        matches = [target for target in targets if output == target or output.endswith(f"_{target}")]
        if matches:
            found.append((pred, max(matches, key=len)))
    return found


def pick_pair(frame, plan, pred=None, target=None):
    column = None if pred is None else pred_column(pred)
    if column is not None and target is not None:
        columns = [str(name) for name in frame.columns]
        return (column, str(target)) if column in columns and str(target) in columns else None
    for found_pred, found_target in prediction_pairs(frame, plan):
        if column is not None and found_pred != column:
            continue
        if target is not None and found_target != str(target):
            continue
        return found_pred, found_target
    return None


def chosen_pair(frame, plan, fitted, pred=None, target=None, prefer=None):
    columns = [str(column) for column in frame.columns]
    column = None if pred is None else pred_column(pred)
    if column is not None and target is not None:
        target = str(target)
        if column not in columns or target not in columns:
            return None
        output = column[len("pred_"):]
        scores = [name for name in columns if name.startswith(f"prob_{output}_")]
        raw = f"raw_{output}" if f"raw_{output}" in columns else None
        kind = stats.kind_of(frame, target, stats.classes_for(plan, fitted, target), scores)
        return {"output": output, "pred": column, "target": target, "kind": kind, "raw": raw, "scores": scores}
    found = [pair for pair in stats.prediction_pairs(frame, plan, fitted)
             if (column is None or pair["pred"] == column) and (target is None or pair["target"] == str(target))]
    preferred = [pair for pair in found if pair["kind"] == prefer]
    return (preferred or found or [None])[0]
