import torch
from torch import nn

from kalfa.core.aliases import register
from kalfa.core.errors import RunError
from kalfa.core.protocols import Objective
from kalfa.core.wrap import loss_scalar
from kalfa.std.common import classes_last_of


def name_of(of):
    return getattr(of, "name", "") or getattr(of, "__name__", "") or type(of).__name__


def label_of(of, term):
    name = name_of(of)
    return name if term is None else f"{name}.{term}"


def term_of(value, term, name):
    if term is None:
        return loss_scalar(value, name)
    if isinstance(value, dict) and term in value:
        return value[term]
    have = f"its terms are {sorted(value)}" if isinstance(value, dict) else "it returns one value"
    raise RunError(f"loss {name!r} has no term {term!r}; {have}")


def items_of(items, kind, owner):
    if isinstance(items, (dict, kind)):
        items = [items]
    out = []
    for item in items or ():
        if isinstance(item, kind):
            out.append(item)
        elif isinstance(item, dict):
            out.append(kind(**item))
        else:
            raise TypeError(f"{owner} takes a list of {{of: ..., ...}} mappings, got {type(item).__name__}")
    return out


class Term:
    def __init__(self, of, weight=1.0, term=None):
        self.of = of
        self.weight = float(weight)
        self.term = term

    def label(self):
        return label_of(self.of, self.term)


class WeightedSum(Objective):
    def __init__(self, terms):
        super().__init__()
        self.terms = items_of(terms, Term, "weighted_sum")
        if not self.terms:
            raise ValueError('weighted_sum needs terms: a list of {of: "@losses.<name>", weight: <value>}')

    def forward(self, outputs, batch, ctx):
        out = {"loss": None}
        total = None
        for term in self.terms:
            value = term_of(term.of(outputs, batch, ctx), term.term, name_of(term.of))
            out[term.label()] = value
            total = term.weight * value if total is None else total + term.weight * value
        out["loss"] = total
        return out


class Constraint:
    def __init__(self, of, epsilon, term=None, scale=1.0, damping=1.0, lmbda_init=0.0):
        self.of = of
        self.epsilon = float(epsilon)
        self.term = term
        self.scale = float(scale)
        self.damping = float(damping)
        self.lmbda_init = float(lmbda_init)

    def label(self):
        return label_of(self.of, self.term)


class Mdmm(Objective):
    def __init__(self, objective, constraints, term=None):
        super().__init__()
        self.objective = objective
        self.constraints = items_of(constraints, Constraint, "mdmm")
        self.term = term
        if not self.constraints:
            raise ValueError('mdmm needs constraints: a list of {of: "@losses.<name>", epsilon: <value>}')
        self.lmbda = nn.Parameter(torch.tensor([entry.lmbda_init for entry in self.constraints],
                                               dtype=torch.float32))

    def forward(self, outputs, batch, ctx):
        value = term_of(self.objective(outputs, batch, ctx), self.term, name_of(self.objective))
        out = {"loss": None, "objective": value}
        for position, constraint in enumerate(self.constraints):
            held = term_of(constraint.of(outputs, batch, ctx), constraint.term, name_of(constraint.of))
            infeasibility = constraint.epsilon - held
            value = value + constraint.scale * (self.lmbda[position] * infeasibility
                                                + constraint.damping * infeasibility ** 2 / 2)
            label = constraint.label()
            out[f"lambda/{label}"] = self.lmbda[position]
            out[f"inf/{label}"] = infeasibility
        out["loss"] = value
        return out


class CrossEntropy(nn.CrossEntropyLoss):
    def __init__(self, weight=None, label_smoothing=0.0, classes_last=None):
        super().__init__(weight=weight, label_smoothing=label_smoothing)
        self.classes_last = classes_last

    def forward(self, pred, target):
        if classes_last_of(pred, target, self.classes_last, self, "cross_entropy") and pred.ndim >= 3:
            pred = pred.movedim(-1, 1)
        return super().forward(pred, target)


def cross_entropy(weight=None, label_smoothing=0.0, classes_last=None):
    if weight is not None:
        weight = torch.as_tensor(weight, dtype=torch.float32)
    return CrossEntropy(weight=weight, label_smoothing=float(label_smoothing), classes_last=classes_last)


LOSSES = (
    ("mse", "torch.nn.MSELoss", "mean squared error"),
    ("l1", "torch.nn.L1Loss", "mean absolute error (the short name mae is the metric; write torch.nn.L1Loss or l1 "
                              "for the loss)"),
    ("huber", "torch.nn.HuberLoss", "huber loss, quadratic below delta and linear above"),
    ("bce_logits", "torch.nn.BCEWithLogitsLoss", "binary cross entropy over logits"),
    ("poisson_nll", "torch.nn.PoissonNLLLoss", "poisson negative log likelihood"),
    ("cross_entropy", "kalfa.std.losses.cross_entropy", "cross entropy over logits (classes in dim 1 or last, as a "
                                                        "language model gives them) with optional class weights "
                                                        "(for example @data.class_weights) and label smoothing; "
                                                        "classes_last: true or false states the layout when the "
                                                        "tokens and the vocabulary have the same size"),
    ("weighted_sum", "kalfa.std.losses.WeightedSum", "a weighted sum of other losses: terms: [{of: @losses.a, "
                                                     "weight: 1.0, term: <key of a dict result>}]"),
    ("mdmm", "kalfa.std.losses.Mdmm", "modified differential method of multipliers: an objective plus constraints "
                                      "[{of: @losses.b, epsilon, scale, damping, lmbda_init, term}] with a "
                                      "lambda parameter per constraint; give lmbda to an optimizer group with a "
                                      "negative lr"),
)

OPTIMIZERS = (
    ("adam", "torch.optim.Adam", "adam"),
    ("adamw", "torch.optim.AdamW", "adam with decoupled weight decay"),
    ("sgd", "torch.optim.SGD", "stochastic gradient descent with optional momentum"),
)


def register_names():
    for name, target, doc in LOSSES:
        register(name, target, slot="loss", doc=doc)
    for name, target, doc in OPTIMIZERS:
        register(name, target, slot="optimizer", doc=doc)


register_names()
