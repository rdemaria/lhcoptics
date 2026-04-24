# Architecture

## Introduction

The library revolves around the `LHCOptics` class. An `LHCOptics`
instance represents a complete optics configuration and contains:

- `params`: global scalar parameters such as energy, tunes, chromaticities,
  and derived optics parameters.
- `knobs`: global `Knob` objects, often specialized as IP, tune, chroma,
  coupling, crab, or dispersion knobs.
- `irs`: the eight concrete IR section instances, `LHCIR1` through `LHCIR8`.
- `arcs`: the eight concrete arc section instances, `LHCA12` through `LHCA81`.

IRs and arcs are concrete subclasses of `LHCSection`. A section contains:

- `strengths`: MAD-X/xSuite strength variables.
- `params`: local scalar parameters, typically measured from variables or Twiss
  tables.
- `knobs`: local `Knob` objects.

`LHCOptics` may also reference:

- `model`: an `LHCXsuiteModel` or `LHCMadxModel` used to Twiss, match, create
  knobs, extract knobs, and update variables.
- `circuits`: an `LHCCircuits` instance connecting strength variables to power
  converter circuits, transfer functions, and limits.

Models and circuits are optional data sources for an optics object, but they are
needed for most matching and consistency checks.

Additional high-level components are:

- `LHCOpticsTable`, `LHCIRTable`, and `LHCArcTable`: sequences of optics or
  sections, typically representing process steps.
- `LHCDev` and `LHCRepo`: local access to acc-models-lhc branches/tags and the
  repository data model.
- `LHCCycle`, `LHCProcess`, `LHCOpticsSet`, and `LHCOpticsDef`: repository
  objects for cycles, beam processes, optics sets, and individual optics files.
- LSA/NXCALS helper classes in `lsa_util.py` and `nxcals_util.py`.

See `class_diagram.mermaid` for the current class hierarchy.

## Loading And Saving

The main loading entry points for `LHCOptics` are:

- `LHCOptics.from_model(model)`: extract optics data from an `LHCXsuiteModel`
  or `LHCMadxModel`.
- `LHCOptics.from_madx_optics(filename)`: load a MAD-X optics file containing
  variable assignments.
- `LHCOptics.from_madx_scripts(filename1, ...)`: run one or more MAD-X scripts
  and extract optics data from the resulting model.
- `LHCOptics.from_cpymad(madx)`: extract optics data from a `cpymad.Madx`
  instance.
- `LHCOptics.from_xsuite(env)`: extract optics data from an xSuite
  environment.
- `LHCOptics.from_json(jsonfile)` / `LHCOptics.from_dict(data)`: load the native
  serialized representation.

Concrete section classes such as `LHCIR1` and `LHCA12` expose similar
constructors for extracting one section. The base classes `LHCSection`, `LHCIR`,
and `LHCArc` provide shared implementation; the concrete IR/arc classes should
be used for real optics data.

The main save/export entry points are:

- `to_json(jsonfile)`: write the native JSON representation.
- `to_dict()`: return the native Python dictionary representation.
- `to_madx([output])`: write or return MAD-X code in variuos formats

Other model and data loading helpers include:

- `LHCXsuiteModel(env)`: wrap an xSuite environment.
- `LHCXsuiteModel.from_json(jsonfile)`: wrap a serialized xSuite environment.
- `LHCXsuiteModel.from_madx_sequence(sequencefile)`: build an xSuite model from
  a MAD-X sequence such as `lhc.seq`.
- `LHCXsuiteModel.from_madx_scripts(filename1, ...)`: run MAD-X scripts and
  convert the resulting cpymad model to xSuite.
- `LHCXsuiteModel.from_cpymad(madx)`: convert a `cpymad.Madx` instance.
- `LHCMadxModel(madx)`: wrap a `cpymad.Madx` instance.
- `LHCMadxModel.from_madx_scripts(filename1, ...)`: run MAD-X scripts and keep
  the cpymad-backed model.
- `LHCCircuits.from_lsa()`, `LHCCircuits.from_json(jsonfile)`, and
  `LHCCircuits.to_json(jsonfile)`.

## Updating, Diffing, Copying, And Plotting

`LHCOptics` and concrete sections are structured so data can be combined from
models, optics objects, sections, and dictionaries.

Common update methods are:

- `update(src=None, strengths=True, knobs=True, params=True, ...)`: update local
  optics or section data. If `src` is omitted, values are taken from the
  attached model where possible.
- `update_strengths(src=None)`: update existing strengths from `src.strengths`,
  a plain dictionary, or the model.
- `update_params(src=None)`: update existing parameters from `src.params`, a
  plain dictionary, or parameters measured from the model.
- `update_knobs(src=None, full=True)`: update existing knobs from `src.knobs`,
  a plain dictionary, or model-extracted knobs. On `LHCOptics`, `full=False`
  limits the update to global knobs.
- `update_model(src=None, full=True, knobs="create", params=True,
  strengths=True)`: push local optics data, or data from `src`, back to the
  attached model.

`src` can be another optics/section object, a structured dictionary, a
model-like object, or a filename for JSON-backed updates in the relevant
methods.

Other common helpers are:

- `diff(other)`, `diff_strengths(other)`, `diff_params(other)`, and
  `diff_knobs(other)`: print differences.
- `copy()`: copy optics or section objects while preserving concrete types and
  variant-specific data.
- `plot(...)` and `plot_<something>(...)`: plot Twiss, aperture, survey, or
  section-specific views, depending on the class.

## Matching

Matching is driven by optics and section parameters. Many parameters correspond
to observables and are derived from Twiss tables; others are model variables.

The main parameter measurement methods are:

- `get_params_from_variables(...)`: extract parameter values from model
  variables.
- `get_params_from_twiss(...)`: compute parameters from Twiss output.
- `set_params(mode="from_variables", ...)`: update local parameters from the
  selected source.

During matching, input parameters and strength variables are copied into the
model. The matched variables remain in the model until `update(...)` is called
to copy the result back into the optics object.

`LHCOptics`, concrete IR/arc sections, and specialized `Knob` classes provide
matching methods. Section and knob matching methods generally return an xSuite
match/optimize object or update the local/model state directly, depending on the
method and `solve` option.

Important matching entry points include:

- `LHCOptics.match(verbose=False)`: rematch the full optics.
- `LHCArc.match(..., solve=True, fail=True)`: match an arc.
- `LHCIR.match(..., solve=True, fail=True)`: match an IR.
- `Knob.match(..., solve=True, fail=True)`: match a specialized knob where
  implemented.
