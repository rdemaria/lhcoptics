# lhcoptics

Python package to build LHC optics.

Usage:
```
git clone githttps://github.com/rdemaria/lhcoptics.git
cd lhcoptics
pip install -e .
python examples/data/make.py
```

```
from lhcoptics import LHCOptics
inj=LHCOptics.from_json("examples/data/opt_inj.json")
inj.set_xsuite_model("examples/data/lhc.json") # set an xsuite model
inj.ir5.plot()
```

NB: Currently needs `github.com/rdemaria/xdeps` branch `upstream/feature/check_limits`.

## LHCOptics
An `LHCOptics` contains data for specifying a full LHC Optics. It contains global params and knobs and a list of LHC sections (`ir1`, `ir2`, ... , `ir8`, `a12`, `a23`, ..., `a78`) that contains strengths, local params and local knobs. The optics can contain a `model` that allows to compute twiss, plot and rematch the optics.

Methods:
- `LHCOptics.set_repository(branch)`: Create a link or a directory `acc-models-lhc` in the local directory from `gitlab.cern.ch/acc-models/acc-models-lhc` if it does not exists. Branch are `2024`, `hl16` etc. If `$HOME/local/acc-models-lhc/branch` exists, a symlink to this directory is created, else the a full git clone is done.

- `LHCOptics.from_json(filename)`: Create optics from a json file
- `LHCOptics.from_dict(dct)`: Create optics from a dict containing strengths, paramns and knobs, irs and arcs dict
- `LHCOptics.from_madx(filename,model)`: Create optics from a madx istance by extracting strengths, params and knobs from variables. If model is 'madx' or 'xsuite' it will attach the model to the optics
- `LHCOptics.from_madxfile(madxfile)`: As before, but creating a madx instance from a madx script


## LHC section methods
An `LHCSection`  is indentified by name, start, end, strengths, params and knobs. A LHCSection is specialzed in LHC Arcs and LHCIRs.

Methods:
- `cls.from_json(filename)`: Create section  from a json file
- `cls.from_dict(dct)`: Create section from a dict containing strengths, paramns and knobs
- `cls.from_madxfile(filename)`: Create section from a madx file
- `cls.from_madx(madx)`: Create section from a madx instance
- `to_json(filename)`: Save section to a json file
- `to_dict()`: Save section to a dict
- `to_madx(filename,dst)`: Save section to dst which could be `str`, filename, a madx instance, an open file.

- `update_model()`: transfer strengths and knobs data to `parent.model`
- `update_strength(src)`: update existing strengths from `src.strengths`, or `src` or `parent.model`  if src is None
- `update_knobs(src)`: update existing knobs from `src.knobs`, or `src` or `parent.model`  if src is None
- `update_params(src)`: update existing params from `src.params`, or `src` or `parent.model`  if src is None
- `update(src)`: combine update_strengths, knobs and params.

- `get_params()`:  measure hardcoded parameters from model
- `get_params_from_twiss()`: measure hardcoded parameters from a twiss table
- `set_params()`: add or replace  hardcoded parameters from model

- `twiss(beam,method)`: get the twiss table table: beam  in (1,2), method see specific sections
- `plot(beam,method)`: plot the twiss table: beam in (1,2), method see specific sections


## LHCIRs methods
- `twiss_from_init(beam)`: get twiss table from boundary conditions
- `twiss_full(beam)`: get the twiss table from the full optics
- `twiss_from_params(beam)`: get the twiss table with init from params
- `set_init()`: set the initial conditions


## LHCArcs methods
- `twiss_init(beam)`: get twiss init at boundaries of the arc from periodic solution
- `twiss_init_cell(beam)`: get twiss init at boundaries of cell
- `twiss_cell(beam)`: get periodic solution of cell



## Code style

Class definitions, camel case alphabetically sorted:
- class variables lower snake case alphabetically sorted
- static methods
- class methods
- init
- other special double underscore methods
- methods starting with verb lower snake case alphabetically sorteddsf