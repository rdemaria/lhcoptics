# Changes

## 0.0.7

### Highlights

- Refactored optics construction around model-driven loading:
  - Added and standardized `from_model`, `from_madx_scripts`, and `from_madx_optics` flows.
  - Added MAD-X optics loading through xSuite and MAD-X model wrappers.
  - Reduced reliance on external knob-structure files; knob names are now generated from optics and section variants.
- Reworked update APIs for `LHCOptics`, sections, and knobs:
  - Update methods now accept optics/section objects, structured dictionaries, and model-like sources more consistently.
  - Added section-source resolution for full optics updates, including verbose reporting.
- Refactored IR6 Twiss-derived extra parameters into `get_extra_params_from_twiss()` and model update hook `add_extra_params_from_twiss()`.
- Consolidated IR1/IR5 implementation into `ir15.py`; public imports still expose `LHCIR1` and `LHCIR5`.
- Added broader xSuite model support for MAD-X sequence conversion, MAD-X optics loading, knob creation/update, expression cleanup, copying, and model configuration.
- Added `sort_module_names.py` and normalized source ordering across `src/lhcoptics`.

### Added

- New docs/artifacts:
  - `architecture.md`
  - `class_diagram.mermaid`
  - `testing.md`
- New examples for model/optics creation and loading:
  - `examples/job_create_lhcoptics.py`
  - `examples/job_create_models.py`
  - `examples/job_load_hllhc_optics.py`
  - `examples/job_load_run3_optics.py`
  - `examples/job_params.py`
- New optics data:
  - `examples/data/hllhc/optics.madx`
  - `examples/data/lhc/optics.madx`
- New tests:
  - `tests/test_updates.py`
  - `tests/test_xsuite_model.py`
  - Session fixtures for `xsuite_model_hl`, `xsuite_model_lhc`, `optics_hl`, and `optics_lhc`.

### Changed

- Renamed active MAD-X loading convention from `from_madxfile` style to:
  - `from_madx_scripts(...)`
  - `from_madx_optics(...)`
- Added/updated `append_from_madx_optics` on optics table flows, with `append_from_madxfile` retained as compatibility forwarding in current code.
- `LHCOptics.gen_knob_names(full=True)` now supports global-only or full section-derived knob names.
- `update_knobs(..., full=False)` support was added for global-only updates.
- `LHCCircuits.to_json()` now uses shared `write_json`.
- Test configuration now supports `--run-long` and marks long tests as excluded by default.
- Examples were reorganized; older scripts moved under `examples/archive`.

### Removed

- Removed active `src/lhcoptics/ir1.py` and `src/lhcoptics/ir5.py` modules in favor of `ir15.py`.
- Removed `examples/data/hllhc/knobs.yaml`.
- Removed old generated JSON examples such as `examples/inj.json` and `examples/ramp/ramp_*.json`.
- Removed or disabled several older tests that no longer match the refactored construction flow.

### Compatibility Notes

- Code using old `LHCOptics.from_madxfile(...)` entry points may need migration to `from_madx_scripts(...)` or `from_madx_optics(...)`.
- Public package imports still expose `LHCIR1` and `LHCIR5`, but they now come from `ir15.py`.

### Verification

- `python3 sort_module_names.py -c src/lhcoptics/*.py`
- `python3 -m py_compile sort_module_names.py src/lhcoptics/*.py`
- `python3 -m pyflakes sort_module_names.py src/lhcoptics`
- `git diff --check -- sort_module_names.py src/lhcoptics`
- `pytest -q`

Result: `14 passed, 1 deselected, 1 warning`.
