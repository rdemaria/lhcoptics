import pytest


@pytest.fixture(params=("optics_hl", "optics_lhc"))
def optics(request):
    return request.getfixturevalue(request.param)


def _shift(value, delta):
    return value + delta


def _first_model_key(names, model):
    return next(name for name in names if name in model)


def _optional_model_key(names, model):
    return next((name for name in names if name in model), None)


def test_section_update_methods_accept_objects_dicts_and_models(optics):
    optics = optics.copy()
    section = optics.ir1.copy()
    section.parent = optics
    source = section.copy()

    strength_name = _first_model_key(section.strengths, optics.model)
    param_name = next(iter(section.params))
    knob_name = _first_model_key(section.knobs, optics.model)

    source.strengths[strength_name] = _shift(section.strengths[strength_name], 1e-6)
    source.params[param_name] = _shift(section.params[param_name], 1e-3)
    source.knobs[knob_name].value = _shift(section.knobs[knob_name].value, 0.25)

    section.update(source)
    assert section.strengths[strength_name] == source.strengths[strength_name]
    assert section.params[param_name] == source.params[param_name]
    assert section.knobs[knob_name].value == source.knobs[knob_name].value

    source.strengths[strength_name] = _shift(source.strengths[strength_name], 1e-6)
    source.params[param_name] = _shift(source.params[param_name], 1e-3)
    source.knobs[knob_name].value = _shift(source.knobs[knob_name].value, 0.25)
    section.update(source.to_dict())
    assert section.strengths[strength_name] == source.strengths[strength_name]
    assert section.params[param_name] == source.params[param_name]
    assert section.knobs[knob_name].value == source.knobs[knob_name].value

    section.update_strengths({strength_name: 123.0})
    section.update_params({param_name: 456.0})
    source.knobs[knob_name].value = 789.0
    section.update_knobs({knob_name: source.knobs[knob_name]})
    assert section.strengths[strength_name] == 123.0
    assert section.params[param_name] == 456.0
    assert section.knobs[knob_name].value == 789.0

    section.update_strengths(optics.model)
    section.update_knobs(optics.model)
    assert section.strengths[strength_name] == optics.model[strength_name]
    assert section.knobs[knob_name].value == optics.model[knob_name]

    model_param_name = _optional_model_key(section.params, optics.model)
    if model_param_name is not None:
        section.update_params(optics.model)
        assert section.params[model_param_name] == optics.model[model_param_name]


def test_lhcoptics_update_accepts_optics_and_structured_dicts(optics):
    target = optics.copy()
    source = target.copy()

    param_name = _first_model_key(target.params, target.model)
    global_knob_name = next(iter(target.knobs))
    section_strength = _first_model_key(target.ir1.strengths, target.model)
    section_param = next(iter(target.ir1.params))
    section_knob = next(iter(target.ir1.knobs))

    source.params[param_name] = _shift(target.params[param_name], 1e-4)
    source.knobs[global_knob_name].value = _shift(
        target.knobs[global_knob_name].value, 0.1
    )
    source.ir1.strengths[section_strength] = _shift(
        target.ir1.strengths[section_strength], 1e-6
    )
    source.ir1.params[section_param] = _shift(target.ir1.params[section_param], 1e-3)
    source.ir1.knobs[section_knob].value = _shift(
        target.ir1.knobs[section_knob].value, 0.2
    )

    target.update(source)
    assert target.params[param_name] == source.params[param_name]
    assert target.knobs[global_knob_name].value == source.knobs[global_knob_name].value
    assert (
        target.ir1.strengths[section_strength]
        == source.ir1.strengths[section_strength]
    )
    assert target.ir1.params[section_param] == source.ir1.params[section_param]
    assert target.ir1.knobs[section_knob].value == source.ir1.knobs[section_knob].value

    source.params[param_name] = _shift(source.params[param_name], 1e-4)
    source.knobs[global_knob_name].value = _shift(
        source.knobs[global_knob_name].value, 0.1
    )
    source.ir1.strengths[section_strength] = _shift(
        source.ir1.strengths[section_strength], 1e-6
    )
    source.ir1.params[section_param] = _shift(source.ir1.params[section_param], 1e-3)
    source.ir1.knobs[section_knob].value = _shift(
        source.ir1.knobs[section_knob].value, 0.2
    )

    target.update(source.to_dict())
    assert target.params[param_name] == source.params[param_name]
    assert target.knobs[global_knob_name].value == source.knobs[global_knob_name].value
    assert (
        target.ir1.strengths[section_strength]
        == source.ir1.strengths[section_strength]
    )
    assert target.ir1.params[section_param] == source.ir1.params[section_param]
    assert target.ir1.knobs[section_knob].value == source.ir1.knobs[section_knob].value

    target.update_params({"params": {param_name: 62.123}}, full=False)
    source.knobs[global_knob_name].value = 0.321
    target.update_knobs(
        {"knobs": {global_knob_name: source.knobs[global_knob_name]}}, full=False
    )
    assert target.params[param_name] == 62.123
    assert target.knobs[global_knob_name].value == 0.321


def test_update_model_accepts_structured_dict_sources(optics):
    target = optics.copy()
    source = target.copy()
    model = target.model

    param_name = _first_model_key(source.params, model)
    section_strength = _first_model_key(source.ir1.strengths, model)
    section_param = _optional_model_key(source.ir1.params, model)
    old_values = {
        param_name: model[param_name],
        section_strength: model[section_strength],
    }
    if section_param is not None:
        old_values[section_param] = model[section_param]

    source.params[param_name] = _shift(old_values[param_name], 1e-4)
    source.ir1.strengths[section_strength] = _shift(old_values[section_strength], 1e-6)
    if section_param is not None:
        source.ir1.params[section_param] = _shift(old_values[section_param], 1e-3)

    try:
        target.update_model(
            source.to_dict(),
            knobs=False,
            set_init=False,
        )
        assert model[param_name] == source.params[param_name]
        assert model[section_strength] == source.ir1.strengths[section_strength]
        if section_param is not None:
            assert model[section_param] == source.ir1.params[section_param]
    finally:
        model.update_vars(old_values)


def test_knob_update_methods_accept_model_like_sources(optics):
    knob = next(knob for knob in optics.knobs.values() if knob.weights).copy()
    weight_name = next(iter(knob.weights))
    model_values = {
        f"{name}_from_{knob.name}": value for name, value in knob.weights.items()
    }
    model_values[f"{weight_name}_from_{knob.name}"] = _shift(
        knob.weights[weight_name], 1e-6
    )

    knob.update(model_values)
    assert knob.weights[weight_name] == model_values[f"{weight_name}_from_{knob.name}"]

    class RecordingModel:
        def __init__(self):
            self.created = []
            self.updated = []

        def create_knob(self, knob, set_value=False, verbose=False):
            self.created.append((knob.name, set_value, verbose))

        def update_knob(self, knob, set_value=False, verbose=False):
            self.updated.append((knob.name, set_value, verbose))

    recorder = RecordingModel()
    knob.update_model(model=recorder, create=True, set_value=True, verbose=True)
    knob.update_model(model=recorder, create=False, set_value=False, verbose=False)

    assert recorder.created == [(knob.name, True, True)]
    assert recorder.updated == [(knob.name, False, False)]
