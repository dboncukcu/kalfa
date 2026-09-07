"""Row filters applied before and after the split."""


from ..registration import lego
from .samples import is_samples


def _query(df, text):
    if is_samples(df):
        return df.query(text)
    return df.query(text)


@lego("/lego/kalfa/filter",
            description="Keep the rows a pandas query selects; a Dataset source takes field equality queries")
def filter(df, query):
    return _query(df, query)


@lego("/lego/kalfa/filter_set",
            description="Apply the {query, sets} filters that name this set; the frame passes untouched otherwise")
def filter_set(df, set, filters):
    out = df
    for entry in filters or []:
        if set in (entry.get("sets") or []):
            out = _query(out, entry["query"])
    return out


@lego("/data/kalfa/class_weights", alias="class_weights",
            description="Inverse frequency class weights of the train set's target field, mean one; built once the "
                        "train loader exists")
def class_weights(loader, target=None):
    import torch

    dataset = loader.dataset
    targets = list(getattr(dataset, "targets", []))
    if target is None:
        if len(targets) != 1:
            raise ValueError(f"class_weights needs one target field or a target name; the dataset has {targets}")
        target = targets[0]
    labels = dataset.labels(target).reshape(-1).long()
    counts = torch.bincount(labels).float()
    weights = counts.sum() / (len(counts) * counts.clamp_min(1.0))
    return weights


@lego("/data/kalfa/vocab_size", alias="vocab_size",
            description="The vocabulary size of the fitted tokenizer among the preprocessors; built once prep exists")
def vocab_size(prep):
    tokenizer = prep.tokenizer() if prep is not None else None
    if tokenizer is None or not hasattr(tokenizer, "size"):
        raise ValueError("vocab_size needs a fitted tokenizer among the preprocessors (char_tokenizer)")
    return int(tokenizer.size)
