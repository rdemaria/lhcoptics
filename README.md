# lhcoptics

Python package to query and build LHC optics.

Installation:
```
git clone https://github.com/rdemaria/lhcoptics.git
cd lhcoptics
pip install -e .
```

Usage:
```
from lhcoptics import LHC
inj=LHC().y2025.pp.ramp[0].plot()
```

## Documentation

The package consists of the following main components:
- `LHC`: Manages the acc-models-lhc LHC optics repository, allowing access to different LHC optics configurations.
- `LHCRepo`: Manages a branch or tag of the LHC optics repository.
- `LHCCycle`: Manages a LHC cycle, including the LHC optics and the LHC beam.
- `LHCProcess`: Manages a LHC beam process: a list of LHC optics and a settings parametrized by time.
- `LHCOptics`: An optics configuration consisting of strengths, knobs and parameters.

Example: plot the IR1 optics in the first step the ramp process of the 2055 proton-proton cycle

```
from lhcoptics import LHC
LHC().y2025.pp.ramp[0].ir1.plot()
```



## LHC
The `LHC` class is responsible for managing the LHC optics repository. It provides functionality to extract branches or tags from the repository and store local copies of the optics. The class also includes mechanisms to regularly check branches for updates.

The class stores data locally, by default in Python user site, or using the environment variable `LHCOPTICS_BASEDIR`. The Git repository URL can also be customized by `LHCOPTICS_GIT_URL`.

Usage:
```
from lhcoptics import LHC
# initialize LHC instance repository
# create a directory if it doesn't exist
# check gitlab for newer versions
lhc = LHC()
# for a check gitlab for updates
lhc.check_local_branches()
# list of branches
lhc.branches
# list of tags
lhc.tags
# extract 2025 branch
lhc.y2025
```


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
- methods starting with verb lower snake case alphabetically sorted




## General convetions

Repository objects have the following methods
- `__init__`: create object from path, read data from file, populate attributes
- `refresh`: re-read data from the file or from dict()
- `read_data`: read the from file
- `to_dict`: return a dictionary of data to be saved from attributures
- `save_data`: save data from attributes to file


## TODO

- collection
   - refactor OpticsDef
   - add madx files and settings
- LHCOptics for hl optics
    - custom knobs
- compiled optics files in EOS
- twiss table in EOS
- website
- documentation