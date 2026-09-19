import pytest
import torch
from cirak import Deferred
from cirak.build import Graph, GraphNode
from cirak.registry import registry
from torch import nn

from helpers import batch, build, frame, linear_graph, tiny_model
from kalfa.std import STD_URIS
from kalfa.std.builder.kalfa.module import Module
from kalfa.std.checkpoint.base import save
from kalfa.std.common.deferred import DeferredLayer
from kalfa.std.common.rng import derived_seed
from kalfa.std.layer.torch.recurrent import Recurrent
from kalfa.std.pre.base import Field, Prep


URI = "/builder/kalfa/module"


def node(name, obj, inputs, outputs, **extra):
    return GraphNode(name, obj, tuple(inputs), tuple(outputs), **extra)


def graph(*nodes, inputs=("x",), outputs=("y",)):
    return Graph(tuple(inputs), tuple(outputs), tuple(nodes))


def constant(value):
    def apply(tensor):
        tensor.fill_(value)
    return apply


def weight(model, name="layer"):
    return model.nodes[name].weight.detach().clone()


def deferred(uri, **params):
    return Deferred(uri, params, registry.resolve(uri))


def train_loader(rows=16):
    dataset = build("/feed/kalfa/table", frame=frame(rows=rows))
    return build("/loader/kalfa/torch", data=dataset, set="train", size=8)


def gru_graph():
    return graph(node("gru", build("/layer/torch/gru", hidden=4), ("x",), ("y",)))


def test_the_builder_declares_its_bus_and_roles():
    assert URI in STD_URIS
    facts = registry.facts(URI)
    assert facts.kind == "builder" and facts.alias == () and facts.partial is False
    assert facts.bus == {"prep": "prep", "train_loader": "train_loader"}
    assert facts.get("roles") == ["weights", "bias", "scale"]


def test_module_runs_the_graph_and_computes_its_outputs():
    model = build(URI, graph=linear_graph(3, 2), seed=1, name="m")
    assert isinstance(model, Module) and isinstance(model, nn.Module)
    assert model.inputs == ["x"] and model.outputs == ["y"] and model.name == "m"
    assert model.initialized and model.settled and model.trainable and model.training
    x = batch()["x"]
    layer = model.nodes["layer"]
    assert torch.equal(model(x), x @ layer.weight.T + layer.bias)
    assert list(model.state_dict()) == ["nodes.layer.weight", "nodes.layer.bias"]
    assert model.node_module(model.graph.nodes[0]) is layer


def test_module_hands_wires_between_nodes_and_returns_the_outputs_in_order():
    first = nn.Linear(3, 2)
    model = build(URI, graph=graph(node("blk.f", first, ("x",), ("h",)), node("act", nn.ReLU(), ("h",), ("y",)),
                                   node("same", nn.Identity(), ("h",), ("z",)), outputs=("y", "z")), seed=1)
    assert model.safe == {"blk.f": "blk__f", "act": "act", "same": "same"}
    assert list(model.nodes) == ["blk__f", "act", "same"] and model.nodes["blk__f"] is first
    assert model.node_module(model.graph.nodes[0]) is first
    x = batch()["x"]
    y, z = model(x)
    assert torch.equal(z, first(x)) and torch.equal(y, torch.relu(first(x)))
    assert list(model.state_dict()) == ["nodes.blk__f.weight", "nodes.blk__f.bias"]


def test_module_counts_its_inputs():
    model = tiny_model()
    with pytest.raises(ValueError) as caught:
        model(torch.ones(2, 3), torch.ones(2, 3))
    assert str(caught.value) == "model takes 1 inputs ['x'], got 2"


def test_unpack_spreads_a_tuple_over_the_node_outputs():
    cell = build("/layer/torch/lstm_cell", input_size=3, hidden=2)
    model = build(URI, graph=graph(node("cell", cell, ("x",), ("h", "c"), unpack=True), outputs=("h", "c")), seed=1)
    h, c = model(batch()["x"])
    expected_h, expected_c = cell(batch()["x"])
    assert torch.equal(h, expected_h) and torch.equal(c, expected_c)
    wrong = build(URI, graph=graph(node("cell", nn.Linear(3, 2), ("x",), ("h", "c"), unpack=True), outputs=("h",)),
                  seed=1)
    with pytest.raises(ValueError) as caught:
        wrong(batch()["x"])
    assert str(caught.value) == "node 'cell' declares outputs ['h', 'c'] but returned Tensor"


def test_derived_rng_seeds_the_build_by_name_whatever_the_index():
    rng = build("/rng/kalfa/derived")
    first = build(URI, graph=linear_graph(), rng=rng, seed=7, name="a", index=0)
    shifted = build(URI, graph=linear_graph(), rng=rng, seed=7, name="a", index=3)
    other = build(URI, graph=linear_graph(), rng=rng, seed=7, name="b", index=0)
    assert first.seed == derived_seed(7, "a") and shifted.seed == first.seed and other.seed == derived_seed(7, "b")
    assert torch.equal(weight(first), weight(shifted))
    assert not torch.equal(weight(first), weight(other))
    assert build(URI, graph=linear_graph(), rng=rng, seed=None, name="a").seed is None


def test_indexed_rng_seeds_the_build_by_position_whatever_the_name():
    rng = build("/rng/kalfa/indexed")
    first = build(URI, graph=linear_graph(), rng=rng, seed=7, name="a", index=1)
    renamed = build(URI, graph=linear_graph(), rng=rng, seed=7, name="b", index=1)
    moved = build(URI, graph=linear_graph(), rng=rng, seed=7, name="a", index=2)
    assert first.seed == derived_seed(7, 1) and renamed.seed == first.seed and moved.seed == derived_seed(7, 2)
    assert torch.equal(weight(first), weight(renamed))
    assert not torch.equal(weight(first), weight(moved))


def test_global_rng_leaves_the_stream_to_the_builds_in_order():
    rng = build("/rng/kalfa/global")
    torch.manual_seed(5)
    first = build(URI, graph=linear_graph(), rng=rng, seed=7, name="a")
    state = torch.get_rng_state()
    second = build(URI, graph=linear_graph(), rng=rng, seed=7, name="b")
    assert first.seed is None and second.seed is None
    assert not torch.equal(weight(first), weight(second))
    torch.manual_seed(5)
    again = build(URI, graph=linear_graph(), rng=rng, seed=7, name="zzz")
    assert torch.equal(weight(again), weight(first))
    torch.set_rng_state(state)
    assert torch.equal(weight(build(URI, graph=linear_graph(), rng=rng, seed=7, name="c")), weight(second))


def test_model_seed_without_an_rng_lego_derives_from_the_name():
    assert build(URI, graph=linear_graph(), seed=7, name="a").seed == derived_seed(7, "a")
    assert build(URI, graph=linear_graph(), seed=7, name="b").seed == derived_seed(7, "b")
    assert build(URI, graph=linear_graph(), seed=None, name="a").seed is None
    assert tiny_model(seed=7, index=0).seed == derived_seed(7, "model0")


def test_a_seeded_build_redraws_the_parameters_without_touching_the_global_stream():
    torch.manual_seed(1)
    one = linear_graph()
    torch.manual_seed(2)
    two = linear_graph()
    assert not torch.equal(one.nodes[0].obj.weight, two.nodes[0].obj.weight)
    torch.manual_seed(3)
    state = torch.get_rng_state()
    first = build(URI, graph=one, seed=3, name="m")
    assert torch.equal(torch.get_rng_state(), state)
    second = build(URI, graph=two, seed=3, name="m")
    assert torch.equal(weight(first), weight(second))
    assert not torch.equal(weight(first), weight(build(URI, graph=linear_graph(), seed=4, name="m")))


def test_an_unseeded_build_keeps_the_weights_the_layers_were_constructed_with():
    torch.manual_seed(1)
    layer = nn.Linear(3, 1)
    original = layer.weight.detach().clone()
    model = build(URI, graph=graph(node("layer", layer, ("x",), ("y",))), seed=None, name="m")
    assert torch.equal(weight(model), original) and model.seed is None and model.initialized


def test_lazy_layers_materialize_on_the_first_batch_under_the_model_seed():
    first = tiny_model(seed=3, lazy=True)
    second = tiny_model(seed=3, lazy=True)
    assert not first.initialized and not first.settled
    assert isinstance(first.nodes["layer"], nn.LazyLinear)
    torch.manual_seed(9)
    state = torch.get_rng_state()
    x = torch.ones(2, 5)
    out = first(x)
    assert torch.equal(torch.get_rng_state(), state)
    assert first.initialized and first.settled and tuple(out.shape) == (2, 1)
    assert type(first.nodes["layer"]) is nn.Linear and tuple(weight(first).shape) == (1, 5)
    second(x)
    assert torch.equal(weight(first), weight(second))
    assert torch.equal(first(x), second(x))
    third = tiny_model(seed=4, lazy=True)
    third(x)
    assert not torch.equal(weight(first), weight(third))


def test_kalfa_lazy_layers_count_as_lazy_and_build_their_core_on_the_first_batch():
    first = build(URI, graph=gru_graph(), seed=3, name="m")
    second = build(URI, graph=gru_graph(), seed=3, name="m")
    assert not first.initialized and isinstance(first.nodes["gru"], Recurrent) and first.nodes["gru"].core is None
    steps = torch.randn(2, 7, 5, generator=torch.Generator().manual_seed(0))
    assert tuple(first(steps).shape) == (2, 7, 4) and first.initialized
    second(steps)
    assert torch.equal(first.nodes["gru"].core.weight_ih_l0, second.nodes["gru"].core.weight_ih_l0)
    assert torch.equal(first(steps), second(steps))


def test_materialization_keeps_the_training_flag():
    model = tiny_model(seed=1, lazy=True)
    model.train()
    model(torch.ones(2, 3))
    assert model.training
    resting = tiny_model(seed=1, lazy=True)
    resting.eval()
    resting(torch.ones(2, 3))
    assert not resting.training


def test_init_roles_apply_by_the_role_of_every_parameter():
    zeros = build("/init/torch/zeros")
    layers = graph(node("a", nn.Linear(3, 3), ("x",), ("h",)), node("n", nn.LayerNorm(3), ("h",), ("g",)),
                   node("b", nn.Linear(3, 1), ("g",), ("y",)))
    model = build(URI, graph=layers, seed=1, init={"weights": zeros, "bias": constant(2.0), "scale": constant(5.0)})
    assert model.init == {"weights": zeros, "bias": model.init["bias"], "scale": model.init["scale"]}
    assert model.nodes["a"].weight.tolist() == [[0.0] * 3] * 3 and model.nodes["b"].weight.tolist() == [[0.0] * 3]
    assert model.nodes["a"].bias.tolist() == [2.0] * 3 and model.nodes["b"].bias.tolist() == [2.0]
    assert model.nodes["n"].weight.tolist() == [5.0] * 3 and model.nodes["n"].bias.tolist() == [2.0] * 3
    assert all(parameter.requires_grad for parameter in model.parameters())
    partial = build(URI, graph=linear_graph(), seed=1, init={"bias": constant(2.0)})
    assert partial.nodes["layer"].bias.tolist() == [2.0] and not torch.equal(weight(partial), torch.zeros(1, 3))


def test_init_patterns_match_the_full_parameter_name_after_the_roles_in_order():
    patterns = [{"match": "b.*", "weights": constant(3.0), "bias": constant(4.0)},
                {"match": "b.weight", "weights": constant(5.0)}, {"match": "*.bias", "scale": constant(9.0)}]
    layers = graph(node("a", nn.Linear(3, 3), ("x",), ("h",)), node("b", nn.Linear(3, 1), ("h",), ("y",)))
    model = build(URI, graph=layers, seed=1, init={"weights": constant(1.0), "patterns": patterns})
    assert model.nodes["a"].weight.tolist() == [[1.0] * 3] * 3
    assert model.nodes["b"].weight.tolist() == [[5.0] * 3] and model.nodes["b"].bias.tolist() == [4.0]
    assert model.nodes["a"].bias.tolist() != [4.0] * 3 and model.nodes["a"].bias.tolist() != [9.0] * 3
    dotted = build(URI, graph=graph(node("blk.f", nn.Linear(3, 1), ("x",), ("y",))), seed=1,
                   init={"patterns": [{"match": "blk__f.weight", "weights": constant(7.0)}]})
    assert dotted.nodes["blk__f"].weight.tolist() == [[7.0] * 3]
    assert [name for name, _ in dotted.named_parameters()] == ["nodes.blk__f.weight", "nodes.blk__f.bias"]


@pytest.mark.xfail(strict=True, reason="bug: apply_roles matches patterns against the safe node name blk__f.weight, "
                                       "so match: 'blk.f.*' as check advises for a template node applies nothing")
def test_init_pattern_written_with_the_dotted_node_name_reaches_its_parameters():
    dotted = build(URI, graph=graph(node("blk.f", nn.Linear(3, 1), ("x",), ("y",))), seed=1,
                   init={"patterns": [{"match": "blk.f.*", "weights": constant(8.0), "bias": constant(6.0)}]})
    assert dotted.nodes["blk__f"].weight.tolist() == [[8.0] * 3]
    assert dotted.nodes["blk__f"].bias.tolist() == [6.0]


def test_node_init_applies_after_the_model_init():
    spec = {"weights": constant(9.0), "patterns": [{"match": "bias", "bias": constant(8.0)}]}
    model = build(URI, graph=graph(node("a", nn.Linear(3, 3), ("x",), ("h",)),
                                   node("b", nn.Linear(3, 1), ("h",), ("y",), extra={"init": spec})),
                  seed=1, init={"weights": constant(1.0), "bias": constant(2.0)})
    assert model.node_init == {"b": spec}
    assert model.nodes["a"].weight.tolist() == [[1.0] * 3] * 3 and model.nodes["a"].bias.tolist() == [2.0] * 3
    assert model.nodes["b"].weight.tolist() == [[9.0] * 3] and model.nodes["b"].bias.tolist() == [8.0]
    only = node("b", nn.Linear(3, 1), ("x",), ("y",), extra={"init": {"weights": constant(9.0)}})
    alone = build(URI, graph=graph(only), seed=1)
    assert alone.nodes["b"].weight.tolist() == [[9.0] * 3]


def test_random_init_roles_run_under_the_model_seed_so_models_match():
    normal = build("/init/torch/normal", std=0.1)
    first = build(URI, graph=linear_graph(), seed=2, name="m", init={"weights": normal})
    second = build(URI, graph=linear_graph(), seed=2, name="m", init={"weights": normal})
    plain = build(URI, graph=linear_graph(), seed=2, name="m")
    assert torch.equal(weight(first), weight(second))
    assert not torch.equal(weight(first), weight(plain))
    assert torch.equal(first.nodes["layer"].bias, plain.nodes["layer"].bias)


def test_trainable_false_freezes_the_parameters_and_keeps_eval_mode():
    model = tiny_model(trainable=False)
    assert model.trainable is False and model.training is False
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert model.train() is model and model.training is False
    assert model.train(False) is model and model.training is False
    live = tiny_model()
    live.train()
    assert live.training and all(parameter.requires_grad for parameter in live.parameters())
    lazy = tiny_model(trainable=False, lazy=True)
    lazy.train()
    lazy(torch.ones(2, 3))
    assert lazy.training is False and all(not parameter.requires_grad for parameter in lazy.parameters())


def test_train_mode_reaches_the_referenced_models_of_a_composite():
    encoder = tiny_model(seed=1)
    frozen = tiny_model(seed=2, index=1, trainable=False)
    composite = build(URI, graph=graph(node("z", None, ("x",), ("z",), ref="encoder"),
                                       node("w", None, ("z",), ("y",), ref="frozen")),
                      models={"encoder": encoder, "frozen": frozen})
    encoder.eval()
    composite.train()
    assert composite.training and encoder.training and frozen.training is False
    composite.eval()
    assert not composite.training and not encoder.training


def test_reference_nodes_take_the_models_by_name():
    encoder = tiny_model(in_features=3, out_features=2, seed=1)
    composite = build(URI, graph=graph(node("z", None, ("x",), ("z",), ref="encoder"),
                                       node("s", nn.Identity(), ("z",), ("s",)), outputs=("s",)),
                      models={"encoder": encoder}, name="full")
    x = batch()["x"]
    assert torch.equal(composite(x), encoder(x)) and composite.refs == {"z": encoder}
    assert list(composite.parameters()) == [] and list(composite.nodes) == ["s"]
    assert composite.node_module(composite.graph.nodes[0]) is encoder
    assert composite.initialized and composite.seed is None


def test_a_reference_to_an_unbuilt_model_is_refused():
    definition = graph(node("z", None, ("x",), ("y",), ref="encoder"))
    for models in (None, {}, {"decoder": tiny_model()}):
        with pytest.raises(KeyError) as caught:
            build(URI, graph=definition, models=models)
        assert caught.value.args[0] == "node 'z' references model 'encoder', which is not built"


def test_weights_load_a_model_of_a_record_by_which(tmp_path):
    run = tmp_path / "run"
    states = {}
    for which, parts in (("best", ("checkpoints", "best.pt")), ("last", ("checkpoints", "last.pt")),
                         ("final", ("final", "state.pt"))):
        source = tiny_model(seed=len(which))
        states[which] = source.state_dict()
        save(run.joinpath(*parts), {"models": {"m": states[which], "other": tiny_model(seed=9).state_dict()}})
    for which in ("best", "last", "final"):
        spec = {"run": str(run), "model": "m", "which": which}
        model = build(URI, graph=linear_graph(), seed=1, name="m", weights=spec)
        assert torch.equal(weight(model), states[which]["nodes.layer.weight"])
        assert torch.equal(model.nodes["layer"].bias, states[which]["nodes.layer.bias"])
        assert model.weights == spec and model.initialized and model.settled and model.pending_state is None
    frozen = build(URI, graph=linear_graph(), seed=1, trainable=False,
                   weights={"run": str(run), "model": "other", "which": "best"})
    assert frozen.training is False and all(not parameter.requires_grad for parameter in frozen.parameters())
    assert torch.equal(weight(frozen), tiny_model(seed=9).nodes["layer"].weight)


def test_weights_name_the_missing_file_model_and_which(tmp_path):
    run = tmp_path / "run"
    with pytest.raises(FileNotFoundError) as caught:
        build(URI, graph=linear_graph(), seed=1, weights={"run": str(run), "model": "m", "which": "last"})
    assert str(caught.value) == f"weights: {run / 'checkpoints' / 'last.pt'} does not exist"
    save(run / "checkpoints" / "best.pt", {"models": {"m": tiny_model().state_dict(), "a": {}}})
    with pytest.raises(KeyError) as caught:
        build(URI, graph=linear_graph(), seed=1, weights={"run": str(run), "model": "zz", "which": "best"})
    assert caught.value.args[0] == f"weights: run {str(run)!r} has no model 'zz'; it has ['a', 'm']"
    with pytest.raises(ValueError) as caught:
        build(URI, graph=linear_graph(), seed=1, weights={"run": str(run), "model": "m", "which": "other"})
    assert str(caught.value) == "weights.which must be best, last or final, got 'other'"
    with pytest.raises(ValueError) as caught:
        build(URI, graph=linear_graph(5, 1), seed=1, weights={"run": str(run), "model": "m", "which": "best"})
    assert str(caught.value).startswith("the weights do not fit the model: ")


def test_weights_wait_for_a_lazy_model_to_materialize(tmp_path):
    run = tmp_path / "run"
    source = tiny_model(seed=6, in_features=4)
    save(run / "final" / "state.pt", {"models": {"m": source.state_dict()}})
    model = build(URI, graph=linear_graph(lazy=True), seed=1, name="m",
                  weights={"run": str(run), "model": "m", "which": "final"})
    assert not model.initialized and list(model.pending_state) == ["nodes.layer.weight", "nodes.layer.bias"]
    x = torch.ones(2, 4)
    assert torch.equal(model(x), source(x))
    assert model.initialized and model.pending_state is None and torch.equal(weight(model), weight(source))


def test_deferred_layer_params_are_built_from_the_train_loader_and_the_prep():
    pick = build("/layer/kalfa/select", index=deferred("/data/kalfa/feature_index", columns=["x2", "x0"]))
    norm = build("/layer/torch/layer_norm", normalized_shape=deferred("/data/kalfa/feature_width"))
    assert isinstance(pick, DeferredLayer) and isinstance(norm, DeferredLayer)
    model = build(URI, graph=graph(node("norm", norm, ("x",), ("h",)), node("pick", pick, ("h",), ("y",))), seed=1,
                  train_loader=train_loader())
    assert type(model.nodes["norm"]) is nn.LayerNorm and model.nodes["norm"].normalized_shape == (3,)
    assert model.nodes["pick"].index.tolist() == [2, 0]
    x = batch()["x"]
    assert torch.equal(model(x), model.nodes["norm"](x)[:, [2, 0]])
    tokenizer = build("/pre/kalfa/char_tokenizer")
    tokenizer.fit(["abc"])
    prep = Prep([Field("text", ["tok"], False, ["text"])], {"tok": {"text": tokenizer}}, {}, {}, [])
    embed = build("/layer/torch/embedding", num=deferred("/data/kalfa/vocab_size"), dim=2)
    words = build(URI, graph=graph(node("embed", embed, ("ids",), ("y",)), inputs=("ids",)), seed=1, prep=prep)
    assert type(words.nodes["embed"]) is nn.Embedding and tuple(words.nodes["embed"].weight.shape) == (4, 2)


def test_a_component_the_builder_cannot_supply_is_refused():
    pick = build("/layer/kalfa/select", index=deferred("/data/kalfa/target_weights"))
    with pytest.raises(ValueError) as caught:
        build(URI, graph=graph(node("pick", pick, ("x",), ("y",))), seed=1, train_loader=train_loader())
    assert str(caught.value) == ("/data/kalfa/target_weights needs ['weights'], which the model builder cannot supply "
                                 "(it has ['loader', 'prep', 'train_loader'])")


def test_state_dict_round_trip_waits_for_a_kalfa_lazy_model():
    source = build(URI, graph=gru_graph(), seed=2, name="m")
    steps = torch.randn(2, 7, 5, generator=torch.Generator().manual_seed(0))
    source(steps)
    target = build(URI, graph=gru_graph(), seed=9, name="m")
    assert target.load_state_dict(source.state_dict()) is None
    assert not target.initialized and list(target.pending_state) == list(source.state_dict())
    assert torch.equal(target(steps), source(steps))
    assert target.initialized and target.settled and target.pending_state is None
    assert torch.equal(target.nodes["gru"].core.weight_hh_l0, source.nodes["gru"].core.weight_hh_l0)


def test_state_dict_loads_straight_into_a_built_or_torch_lazy_model():
    source = tiny_model(seed=2, lazy=True)
    source(torch.ones(1, 4))
    lazy = tiny_model(seed=9, lazy=True)
    result = lazy.load_state_dict(source.state_dict())
    assert result.missing_keys == [] and result.unexpected_keys == []
    assert lazy.initialized and lazy.settled and lazy.pending_state is None
    assert torch.equal(lazy(torch.ones(2, 4)), source(torch.ones(2, 4)))
    built = tiny_model(seed=3, in_features=4)
    built.load_state_dict(source.state_dict())
    assert torch.equal(weight(built), weight(source))
    frozen = tiny_model(seed=4, in_features=4, trainable=False)
    frozen.train()
    frozen.load_state_dict(source.state_dict())
    assert frozen.training is False and all(not parameter.requires_grad for parameter in frozen.parameters())
