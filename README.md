# lhcoptics

Build LHC optics

Usage:
```
python examples/data/make.py
```

```
from lhcoptics import LHCOptics

inj=LHCOptics.from_json("data/opt_inj.json")
inj.set_xsuite_model("data/lhc.json") # set an xsuite model
inj.get_params()

```

Currently needs `github.com/rdemaria/xtrack` branch `attr_in_twiss`
