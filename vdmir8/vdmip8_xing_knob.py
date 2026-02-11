#
# -- IP8 VdM Optics - 2025
#
# Generate new Xing knob on_x8h
#   - use acbch6.r8b1 to avoid acbyhs5.r8b1 hitting the limit at -300urad 

#%%
from lhcoptics import LHCOptics, LHC
from lhcoptics import LHCCircuits

import pprint

#%%
# --- get optics at desqueeze

# - the last three files are (from the end) at beta* = 50, 44, 34 m

opt = LHC().y2025.vdm.desqueeze[-1]
# opt = LHC().y2025.vdm.desqueeze[-2]
opt.ir8.plot()
_tt = opt.twiss(1)
print(f' -- beta values at IP8: {_tt["betx", "ip8"]=:.6g} - {_tt["bety", "ip8"]=:.6g} ')


# %%

pprint.pprint(opt.ir8.knobs)
pprint.pprint(opt.ir8.knobs['on_x8h'].weights)

on_x8h = opt.ir8.knobs['on_x8h']
opt.ir8.plot(yl='x y')

# %%

def check_strengths_at_knob_value(aknob, circuits, value):
    ''' Check strengths at value for the knob'''
    for k, w in aknob.weights.items():
        _klimits = circuits.madname[k].get_klimits()
        if min(_klimits) <= w*value <= max(_klimits) :
            print(f'\t {k:>15s} = {w*value/1.0e-6:12.3f} [urad] - {_klimits}')
        else:
            print(f'\t {k:>15s} = {w*value/1.0e-6:12.3f} [urad] - {_klimits} - out of bounds')

    return
# %%
# --- check strengths
opt.set_circuits("data/lhccircuits.json")

check_strengths_at_knob_value(on_x8h, opt.circuits, 200)

# %%

# --- update the knob

print(on_x8h.const)
print(on_x8h.value)
on_x8h.weights

#%%
# --- maintain acbyhs5.r8b1 to the 200 urad settings and scale it to 300 urad

on_x8h.const.append("acbyhs5.r8b1") # remember to do it again or check
on_x8h.weights["acbyhs5.r8b1"]*=200/300
on_x8h.weights

