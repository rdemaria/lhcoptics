import xtrack as xt

lhc = xt.load(
    "https://acc-models.web.cern.ch/acc-models/lhc/2025/xsuite/lhc.json"
)
lhc.vars.load("https://acc-models.web.cern.ch/acc-models/lhc/2025/scenarios/cycle/pp/ramp/0/optics.madx")
lhc.b1.twiss().plot()
