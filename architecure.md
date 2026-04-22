Architecture
====================

Introduction
----------------

The library revolves around the `LHCOptics` class.

`LHCOptics` contains:
- `params`: dictionary of floats (cannot be string for the time being to be compatible with MAD-X storage)
- `knobs`: dictionary of `Knob` objects (typically specialized)
- `irs`: `IR1(...)`, `IR2(...)`, ..., `IR8(...)` instances
- `arcs`: `Arc("a12", ...)`, `Arc("a23", ...)` instances

IRs and Arcs are Sections and contains:
- `params`: as before
- `strengths`: dictionary of floats representing circuits
- `knobs`: as before

`LHCOptics` may contain:
- `model`: can be `LHCXsuiteModel` or `LHCMadxModel` which are able to Twiss, create, or extract knobs, do matching
- `circuits`: an instance of `LHCCircuits` that connects strength variables to their respective circuits, provide transfer functions, limits for matching

Model and circuits allow `LHCOptics` to match optics.

In additions there are:
- `LHCOpticsTable`, `LHCIRTable`, `LHCArcTable`: represent a sequence of optics in a process
- `LHCDev`: manages acc-models repository in https://gitlab.cern.ch/acc-models/acc-models-lhc
- `LHC`: interact with data https://acc-models.web.cern.ch/acc-models/lhc/


Loading and Saving
---------------------

`LHCOptics`, `LHCIR1`,...,`LHCA12`,... can be loaded from:
- `<cls>.from_model(model)`: load from a model among `LHCXsuiteModel` or `LHCMadxModel`
- `<cls>.from_madx_optics(filename)`: load from a full MAD-X optics file (not a scripts)
- `<cls>.from_madx_scripts(filename1, ...)`: load from a MAD-X scripts
- `<cls>.from_cpymad(madx,...)`: load from `cpymad` instance
- `<cls>.from_xsuite(env,...)`: load from an Xsuite environment
- `<cls>.from_json(jsonfile)`: load from a native JSON file
- `<cls>.from_acc_models(path,...)`: load from acc-model website e.g: `path="2025/pp/ramp/0"`
- `<cls>.from_dict(dct)`: from Python dictionary

`LHCOptics`, `LHCIR1`,...,`LHCA12`,... can be saved to:
- `.to_json(jsonfile)`: native json
- `.to_dict()`: to Python dictionary
- `.to_madx(madxfile)`: madx format

Other loading/saving functions
- `LHCXsuiteModel(env)`: from Xsuite environment
- `LHCXsuiteModel.from_madx_sequence(sequencefile)`: from MAD-X sequence such as `"lhc.seq"`
- `LHCXsuiteModel.from_cpymad(madx)`: from `cpypmad` `Madx` instance
- `LHCMadxModel(madx)`: from `cpymad` `Madx` instance
- `LHCMadxModel.from_madx_scripts(filename1, ...)`: from MAD-X script
- `LHCCircuits.from_lsa()`: from LSA system
- `LHCCircuits.from_json(jsonfile)`: from native JSON file
- `LHCCircuits.to_json(jsonfile)`: to native JSON file


Updating, diffing, copying, plotting
----------------------------------------

`LHCOptics` has been structured to facilitate combining section from different optics and models

The following methods are for `LHCOptics`, `LHCIR1`,...,`LHCA12`, ...:
- `.update([src],strenghts=True|False, knobs=True|False, params=True|False)`: update default is from model, else can be another instance or dictionary. Other defaults are `True`.
- `.update_strength([src])`,`.update_params([src])`,`.update_knob([src])`: as before
- `.update_model(strenghts=True|False, knobs=True|False, params=True|False)`: update model
- `.diff([src])`: print difference with source, default is model. Other defaults are `True`.
- `.copy()`: make a copy

The classes have various plotting methods all starting with `.plot` and `.plot_<something>`:
- `.plot([1|2]), `: Plots Beam 1 and Beam 2 or Beam 1 and Beam 2 separately.



Matching
------------------------------------

Matching is driven by some parameters in the LHCOptics and sections instances. Most parameters are related to observables and thereore calculated from twiss. Since matching is approxiamted even observables that are matched for are not strictly the same as the parameters. We have
-`get_params_from twiss`: that computes parameters from Twiss output
-`get_params_from_variables`: that extract parameters from variables in the model.

In matching the parameters used as input are copied in the model, while the parameters resulting in the output are copied from twiss. `update_params` by default uses parameters from variables.

`LHCOptics`, `LHCIR1`,...,`LHCA12`, ... and specialize `Knob` instance have matching functions that uses the parameters, limits from circuits and other option to match strength variables in the model and update parameter variables. The variables  stays in the model, until `.update` is explictly called. 
- `.match(solve=True|False, fail=True|False)`: return Xsuite Optimized instance, if `solve` also match, if `fail` raise an exception if case the targets are not matched.








