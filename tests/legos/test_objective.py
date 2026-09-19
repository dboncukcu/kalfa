import math

import pytest
import torch
from cirak.build import Graph, GraphNode
from cirak.registry import registry
from torch import nn

from helpers import batch, build, tiny_model
from kalfa.std import STD_URIS
from kalfa.std.builder.kalfa.module import Module


OBJECTIVES = sorted(uri for uri in STD_URIS if uri.startswith("/objective/"))


class Net(nn.Module):
    def __init__(self, function, inputs=("x",)):
        super().__init__()
        self.function = function
        self.inputs = list(inputs)
        self.outputs = ["y"]
        self.seen = None

    def forward(self, *arguments):
        self.seen = arguments
        return self.function(*arguments)


def scalar(value):
    return float(value.detach())


def constant(model, weight, bias=0.0):
    with torch.no_grad():
        model.nodes["layer"].weight.fill_(weight)
        model.nodes["layer"].bias.fill_(bias)
    return model


def lambdas_model(names):
    layer = build("/layer/kalfa/multipliers", names=names)
    graph = Graph(("x",), ("lmbda",), (GraphNode("m", layer, ("x",), ("lmbda",)),))
    return Module(graph, seed=1, name="lambdas")


def test_objective_scope_is_the_eight_kalfa_objectives():
    assert OBJECTIVES == ["/objective/kalfa/ddpm", "/objective/kalfa/distill", "/objective/kalfa/mdmm",
                          "/objective/kalfa/ntxent", "/objective/kalfa/vae", "/objective/kalfa/weighted_sum",
                          "/objective/kalfa/wgan_g", "/objective/kalfa/wgan_gp_d"]


def test_every_objective_is_partial_and_aliased_by_its_name():
    for uri in OBJECTIVES:
        assert registry.facts(uri).partial is True
        assert registry.aliases()[uri.rsplit("/", 1)[1]] == uri


def test_objective_refs_type_the_models_losses_and_schedules_they_name():
    assert registry.facts("/objective/kalfa/wgan_g").refs == {"generator": "model", "critic": "model"}
    assert registry.facts("/objective/kalfa/wgan_gp_d").get("needs_grad") is True
    assert registry.facts("/objective/kalfa/ddpm").refs == {"model": "model", "schedule": "schedule"}
    assert registry.facts("/objective/kalfa/distill").refs == {"student": "model", "teacher": "model"}
    assert registry.facts("/objective/kalfa/vae").refs == {"encoder": "model", "decoder": "model",
                                                          "recon": "criterion", "kl_schedule": "schedule"}
    assert registry.facts("/objective/kalfa/mdmm").refs == {"primary": "loss", "constraints": "loss",
                                                           "multipliers": "model"}
    assert registry.facts("/objective/kalfa/weighted_sum").refs == {"terms": "loss"}


def test_wgan_g_is_minus_the_mean_critic_score_of_the_generated_samples():
    models = {"g": constant(tiny_model(2, 3, index=0), 1.0), "d": constant(tiny_model(3, 1, index=1), 1.0)}
    objective = build("/objective/kalfa/wgan_g", generator="g", critic="d", latent=2)
    data = batch(rows=4)
    value = objective(models, data, rng=torch.Generator().manual_seed(3))
    noise = torch.randn((4, 2), generator=torch.Generator().manual_seed(3))
    assert scalar(value) == pytest.approx(-3.0 * float(noise.sum(dim=1).mean()))
    assert value.requires_grad


def test_wgan_g_reads_the_critic_input_wire_from_the_batch():
    models = {"g": tiny_model(2, 3, index=0), "d": tiny_model(3, 1, index=1)}
    objective = build("/objective/kalfa/wgan_g", generator="g", critic="d", latent=2)
    with pytest.raises(KeyError, match=r"the objective needs the model's first input wire in the batch"):
        objective(models, {"image": torch.zeros(2, 3)})


def test_wgan_g_conditional_needs_the_label_field():
    models = {"g": tiny_model(2, 3, index=0), "d": tiny_model(3, 1, index=1)}
    objective = build("/objective/kalfa/wgan_g", generator="g", critic="d", latent=2, conditional=True)
    with pytest.raises(KeyError, match=r"a conditional GAN reads the label field of the batch"):
        objective(models, batch(rows=4))


def test_wgan_g_conditional_hands_the_labels_to_both_models():
    generator = Net(lambda noise, labels: noise.sum(dim=1, keepdim=True) + labels[:, None].float(), ("z", "label"))
    critic = Net(lambda x, labels: x.sum(dim=1) + labels.float(), ("x", "label"))
    objective = build("/objective/kalfa/wgan_g", generator="g", critic="d", latent=2, conditional=True)
    data = {"x": torch.zeros(4, 1), "label": torch.tensor([0, 1, 2, 3])}
    value = objective({"g": generator, "d": critic}, data, rng=torch.Generator().manual_seed(3))
    noise = torch.randn((4, 2), generator=torch.Generator().manual_seed(3))
    assert scalar(value) == pytest.approx(-float((noise.sum(dim=1) + 2.0 * torch.arange(4)).mean()))


def test_wgan_gp_d_adds_the_unit_norm_gradient_penalty_to_the_score_gap():
    models = {"g": constant(tiny_model(2, 3, index=0), 1.0), "d": constant(tiny_model(3, 1, index=1), 1.0)}
    objective = build("/objective/kalfa/wgan_gp_d", generator="g", critic="d", latent=2, gp_weight=10.0)
    data = batch(rows=4)
    value = objective(models, data, rng=torch.Generator().manual_seed(3))
    noise = torch.randn((4, 2), generator=torch.Generator().manual_seed(3))
    gap = 3.0 * float(noise.sum(dim=1).mean()) - float(data["x"].sum(dim=1).mean())
    assert scalar(value) == pytest.approx(gap + 10.0 * (math.sqrt(3.0) - 1.0) ** 2)
    assert value.requires_grad


def test_wgan_gp_d_without_penalty_weight_is_the_score_gap():
    models = {"g": constant(tiny_model(2, 3, index=0), 1.0), "d": constant(tiny_model(3, 1, index=1), 1.0)}
    objective = build("/objective/kalfa/wgan_gp_d", generator="g", critic="d", latent=2, gp_weight=0.0)
    data = batch(rows=4)
    value = objective(models, data, rng=torch.Generator().manual_seed(3))
    noise = torch.randn((4, 2), generator=torch.Generator().manual_seed(3))
    assert scalar(value) == pytest.approx(3.0 * float(noise.sum(dim=1).mean()) - float(data["x"].sum(dim=1).mean()))


def test_ddpm_noises_the_input_by_the_schedule_and_scores_the_noise_prediction():
    net = Net(lambda noised, t: torch.zeros_like(noised))
    schedule = build("/schedule/kalfa/linear_betas", steps=4, start=0.1, end=0.4)
    objective = build("/objective/kalfa/ddpm", model="net", schedule=schedule)
    data = batch(rows=4)
    value = objective({"net": net}, data, rng=torch.Generator().manual_seed(5))
    rng = torch.Generator().manual_seed(5)
    t = torch.randint(0, 4, (4,), generator=rng)
    noise = torch.randn(data["x"].shape, generator=rng)
    assert scalar(value) == pytest.approx(float((noise ** 2).mean()))
    cumulative = torch.cumprod(1.0 - torch.tensor([0.1, 0.2, 0.3, 0.4]), dim=0)
    weight = cumulative[t].reshape(4, 1)
    noised, seen_t = net.seen
    assert torch.equal(seen_t, t)
    assert torch.allclose(noised, weight.sqrt() * data["x"] + (1.0 - weight).sqrt() * noise)


def test_ddpm_needs_a_schedule_lego_with_steps():
    objective = build("/objective/kalfa/ddpm", model="net", schedule=lambda step: 0.1)
    with pytest.raises(ValueError, match=r"the noise schedule must be a schedule lego with a steps param"):
        objective({"net": Net(lambda noised, t: noised)}, batch(rows=2))


def test_distill_returns_the_mixed_loss_with_its_ce_and_kl_terms():
    student = constant(tiny_model(3, 2, index=0), 0.0)
    teacher = constant(tiny_model(3, 2, index=1), 0.0)
    with torch.no_grad():
        teacher.nodes["layer"].bias.copy_(torch.tensor([math.log(3.0), 0.0]))
    objective = build("/objective/kalfa/distill", student="s", teacher="t", alpha=0.25)
    data = {"x": batch(rows=4)["x"], "label": torch.tensor([0, 1, 0, 1])}
    out = objective({"s": student, "t": teacher}, data)
    kl = 0.75 * math.log(0.75 / 0.5) + 0.25 * math.log(0.25 / 0.5)
    assert set(out) == {"loss", "ce", "kl"}
    assert scalar(out["ce"]) == pytest.approx(math.log(2.0))
    assert scalar(out["kl"]) == pytest.approx(kl)
    assert scalar(out["loss"]) == pytest.approx(0.25 * kl + 0.75 * math.log(2.0))
    assert out["loss"].requires_grad


def test_distill_softens_both_distributions_by_the_temperature_squared():
    student = constant(tiny_model(3, 2, index=0), 0.0)
    teacher = constant(tiny_model(3, 2, index=1), 0.0)
    with torch.no_grad():
        teacher.nodes["layer"].bias.copy_(torch.tensor([math.log(3.0), 0.0]))
    objective = build("/objective/kalfa/distill", student="s", teacher="t", temperature=2.0, alpha=1.0)
    data = {"x": batch(rows=2)["x"], "label": torch.tensor([0, 1])}
    out = objective({"s": student, "t": teacher}, data)
    top = math.sqrt(3.0) / (math.sqrt(3.0) + 1.0)
    kl = 4.0 * (top * math.log(top / 0.5) + (1.0 - top) * math.log((1.0 - top) / 0.5))
    assert scalar(out["kl"]) == pytest.approx(kl, rel=1e-5)
    assert scalar(out["loss"]) == pytest.approx(kl, rel=1e-5)


def test_distill_cannot_guess_the_target_among_several_fields():
    objective = build("/objective/kalfa/distill", student="s", teacher="t")
    data = {"x": torch.zeros(2, 3), "a": torch.tensor([0, 1]), "b": torch.tensor([1, 0])}
    with pytest.raises(ValueError, match=r"cannot tell the target field among \['a', 'b'\]; write target"):
        objective({"s": tiny_model(3, 2, index=0), "t": tiny_model(3, 2, index=1)}, data)
    named = build("/objective/kalfa/distill", student="s", teacher="t", target="b")
    out = named({"s": constant(tiny_model(3, 2, index=0), 0.0), "t": constant(tiny_model(3, 2, index=1), 0.0)}, data)
    assert scalar(out["ce"]) == pytest.approx(math.log(2.0))


def test_ntxent_scores_every_view_against_its_partner():
    net = Net(lambda views: views.flatten(1))
    views = torch.zeros(2, 2, 1, 2, 2)
    views[0, :, 0, 0, 0] = 1.0
    views[1, :, 0, 0, 1] = 1.0
    objective = build("/objective/kalfa/ntxent", model="net", temperature=0.5)
    value = objective({"net": net}, {"x": views})
    assert scalar(value) == pytest.approx(math.log(math.exp(2.0) + 2.0) - 2.0)
    warm = build("/objective/kalfa/ntxent", model="net", temperature=1.0)
    assert scalar(warm({"net": net}, {"x": views})) == pytest.approx(math.log(math.e + 2.0) - 1.0)


def test_ntxent_needs_two_views_per_image():
    objective = build("/objective/kalfa/ntxent", model="net")
    with pytest.raises(ValueError, match=r"ntxent needs two views per image: put two_views in the field's chain"):
        objective({"net": Net(lambda views: views.flatten(1))}, {"x": torch.zeros(2, 1, 2, 2)})


def test_vae_sums_the_reconstruction_and_the_kl_with_unit_weights():
    encoder = Net(lambda x: (torch.zeros(len(x), 2), torch.zeros(len(x), 2)))
    decoder = Net(lambda z: torch.zeros(len(z), 3))
    objective = build("/objective/kalfa/vae", encoder="e", decoder="d", recon=build("/criterion/kalfa/mse"))
    data = batch(rows=4)
    out = objective({"e": encoder, "d": decoder}, data, rng=torch.Generator().manual_seed(2))
    assert set(out) == {"loss", "recon", "kl", "w_kl"}
    assert scalar(out["recon"]) == pytest.approx(float((data["x"] ** 2).mean()))
    assert scalar(out["kl"]) == 0.0
    assert out["w_kl"] == 1.0
    assert scalar(out["loss"]) == pytest.approx(float((data["x"] ** 2).mean()))
    assert torch.equal(decoder.seen[0], torch.randn((4, 2), generator=torch.Generator().manual_seed(2)))


def test_vae_weights_the_kl_by_the_schedule_at_the_step():
    encoder = Net(lambda x: (torch.ones(len(x), 2), torch.zeros(len(x), 2)))
    decoder = Net(lambda z: torch.zeros(len(z), 3))
    warmup = build("/schedule/kalfa/linear_warmup", start=0.0, end=1.0, steps=10)
    objective = build("/objective/kalfa/vae", encoder="e", decoder="d", recon=build("/criterion/kalfa/mse"),
                      w_rec=2.0, kl_schedule=warmup)
    data = batch(rows=4)
    out = objective({"e": encoder, "d": decoder}, data, step=5, rng=torch.Generator().manual_seed(2))
    assert scalar(out["kl"]) == 1.0
    assert out["w_kl"] == 0.5
    assert scalar(out["loss"]) == pytest.approx(2.0 * float((data["x"] ** 2).mean()) + 0.5)


def test_mdmm_is_the_primary_plus_the_damped_multiplier_terms():
    constraints = {"mae": {"epsilon": 0.5, "lmbda_init": 2.0, "scale": 1.0, "damping": 1.0}, "ws.a": 0.3}
    models = {"lambdas": lambdas_model(constraints)}
    losses = {"ws": {"loss": torch.tensor(2.0), "a": torch.tensor(0.5)}, "mae": torch.tensor(1.5)}
    objective = build("/objective/kalfa/mdmm", primary="ws", multipliers="lambdas", constraints=constraints)
    out = objective(models, batch(rows=1), losses=losses)
    assert list(out) == ["loss", "primary", "lambda/mae", "inf/mae", "lambda/ws.a", "inf/ws.a"]
    assert scalar(out["primary"]) == 2.0
    assert scalar(out["lambda/mae"]) == 2.0
    assert scalar(out["inf/mae"]) == -1.0
    assert scalar(out["lambda/ws.a"]) == 0.0
    assert scalar(out["inf/ws.a"]) == pytest.approx(-0.2)
    assert scalar(out["loss"]) == pytest.approx(0.52)


def test_mdmm_scale_and_damping_shape_every_constraint_term():
    constraints = {"mae": {"epsilon": 0.5, "lmbda_init": 2.0, "scale": 2.0, "damping": 3.0}, "ws.a": 0.3}
    models = {"lambdas": lambdas_model(constraints)}
    losses = {"ws": {"loss": torch.tensor(2.0), "a": torch.tensor(0.5)}, "mae": torch.tensor(1.5)}
    objective = build("/objective/kalfa/mdmm", primary="ws", multipliers="lambdas", constraints=constraints)
    out = objective(models, batch(rows=1), losses=losses)
    assert scalar(out["loss"]) == pytest.approx(1.02)


def test_mdmm_gradient_on_a_multiplier_is_its_scaled_infeasibility():
    constraints = {"mae": {"epsilon": 0.5, "scale": 2.0}, "ws.a": 0.3}
    model = lambdas_model(constraints)
    losses = {"ws": {"loss": torch.tensor(2.0), "a": torch.tensor(0.5)}, "mae": torch.tensor(1.5)}
    objective = build("/objective/kalfa/mdmm", primary="ws", multipliers="lambdas", constraints=constraints)
    out = objective({"lambdas": model}, batch(rows=1), losses=losses)
    gradient = torch.autograd.grad(out["loss"], model.nodes["m"].lmbda)[0]
    assert torch.allclose(gradient, torch.tensor([-2.0, -0.2]))


def test_mdmm_refuses_constraints_that_are_not_a_mapping():
    objective = build("/objective/kalfa/mdmm", primary="ws", multipliers="lambdas", constraints={})
    with pytest.raises(ValueError, match=r"mdmm needs constraints: a mapping of a losses name \(or name.term\) to "
                                          r"\{epsilon, lmbda_init, scale, damping\}"):
        objective({}, batch(rows=1), losses={})


def test_mdmm_constraint_needs_epsilon():
    objective = build("/objective/kalfa/mdmm", primary="ws", multipliers="lambdas",
                      constraints={"mae": {"scale": 1.0}})
    with pytest.raises(ValueError, match=r"mdmm constraint 'mae' needs epsilon, the value the term is held at"):
        objective({}, batch(rows=1), losses={})


def test_mdmm_constraint_names_only_the_four_spec_keys():
    objective = build("/objective/kalfa/mdmm", primary="ws", multipliers="lambdas",
                      constraints={"mae": {"epsilon": 0.5, "weight": 2.0}})
    with pytest.raises(ValueError, match=r"mdmm constraint 'mae' has no \['weight'\]; a constraint writes "
                                          r"\['epsilon', 'lmbda_init', 'scale', 'damping'\]"):
        objective({}, batch(rows=1), losses={})


def test_mdmm_multipliers_must_name_a_model():
    objective = build("/objective/kalfa/mdmm", primary="ws", multipliers="lambdas", constraints={"mae": 0.5})
    with pytest.raises(KeyError, match=r"mdmm: multipliers names 'lambdas', which is no model; the models are "
                                        r"\['m'\]"):
        objective({"m": tiny_model()}, batch(rows=1), losses={})


def test_mdmm_multipliers_model_must_carry_the_same_names():
    objective = build("/objective/kalfa/mdmm", primary="ws", multipliers="lambdas", constraints={"ws.a": 0.3})
    with pytest.raises(ValueError, match=r"mdmm: model 'lambdas' holds multipliers for \['mae'\], the constraints "
                                          r"are \['ws.a'\]; write names: \$constraints\$ on the multipliers node"):
        objective({"lambdas": lambdas_model(["mae"])}, batch(rows=1), losses={})


def test_mdmm_multiplier_count_must_match_the_constraints():
    objective = build("/objective/kalfa/mdmm", primary="ws", multipliers="lambdas",
                      constraints={"mae": 0.5, "ws.a": 0.3})
    with pytest.raises(ValueError, match=r"mdmm: model 'lambdas' returns 3 multipliers for 2 constraints"):
        objective({"lambdas": constant(tiny_model(3, 3), 1.0)}, batch(rows=1), losses={})


def test_mdmm_names_a_missing_term_of_a_losses_definition():
    losses = {"ws": {"loss": torch.tensor(2.0), "a": torch.tensor(0.5)}, "mae": torch.tensor(1.5)}
    objective = build("/objective/kalfa/mdmm", primary="ws.zz", multipliers="lambdas", constraints={"mae": 0.5})
    with pytest.raises(KeyError, match=r"losses definition 'ws' has no term 'zz'; its terms are \['a', 'loss'\]"):
        objective({"lambdas": lambdas_model({"mae": 0.5})}, batch(rows=1), losses=losses)
    single = build("/objective/kalfa/mdmm", primary="ws", multipliers="lambdas", constraints={"mae.x": 0.5})
    with pytest.raises(KeyError, match=r"losses definition 'mae' has no term 'x'; it returns one value"):
        single({"lambdas": lambdas_model({"mae.x": 0.5})}, batch(rows=1), losses=losses)


def test_weighted_sum_returns_the_total_and_every_term():
    objective = build("/objective/kalfa/weighted_sum", terms={"a": 1.0, "b": 0.5})
    losses = {"a": torch.tensor(2.0), "b": {"loss": torch.tensor(4.0), "x": torch.tensor(1.0)}}
    out = objective({}, batch(rows=1), losses=losses)
    assert list(out) == ["a", "b", "loss"]
    assert scalar(out["a"]) == 2.0
    assert scalar(out["b"]) == 4.0
    assert scalar(out["loss"]) == 4.0


def test_weighted_sum_needs_terms():
    objective = build("/objective/kalfa/weighted_sum", terms={})
    with pytest.raises(ValueError, match=r"weighted_sum needs terms: a mapping of losses names to weights"):
        objective({}, batch(rows=1), losses={})
