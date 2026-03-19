import pytest


@pytest.mark.parametrize("section_name", ["ir1", "a12"])
def test_section_copy_and_dict_preserve_variant(optics_hl, section_name):
    section = getattr(optics_hl, section_name)

    copied = section.copy()
    data = section.to_dict()

    assert copied.variant == "hl"
    assert data["variant"] == "hl"


@pytest.mark.parametrize("section_name", ["ir1", "a12"])
def test_section_check_params_on_hllhc_example(optics_hl, section_name):
    section = getattr(optics_hl, section_name)

    assert section.check_params(verbose=False)
