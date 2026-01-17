#!/usr/bin/env python
# coding: utf-8

# # Desqueeze VDM optics for IR8

# In[1]:


# lhcoptics is installed within pip

from lhcoptics import LHCOptics, LHC
from lhcoptics import LHCCircuits

import pprint


# ## Load VdM optics at 50m beta*

# In[10]:


print(f' ----- : VdM Optics 2025:')
print(f' === Processess ===')
pprint.pprint(LHC().y2025.vdm.beam_processes)
# pprint.pprint(LHC().y2025.vdm.desqueeze)
for k,v  in LHC().y2025.vdm.beam_processes.items():
    print(f' \t process: {k}')
    pprint.pprint(dict(LHC().y2025.vdm.beam_processes[k].optics), width=40, indent=4)


# In[11]:


# --- test: get collision optics at desqueeze
regenerate = False
if regenerate:
    opt = LHC().y2025.vdm.desqueeze[-3]
    opt.to_json("./data/vdmopt_2025_34m.json")

opt=LHCOptics.from_json("./data/vdmopt_2025_34m.json",xsuite_model=LHC().y2025.get_xsuite_json())

opt.ir8.plot()
_tt = opt.twiss(1)
print(f' -- beta values at IP8: {_tt["betx", "ip8"]=:.6g} - {_tt["bety", "ip8"]=:.6g} ')


# In[12]:


# opt50m.model.info('acbxh3.r8') # show definition
opt.ir8.knobs


# In[13]:


pprint.pprint(opt.ir8.knobs['on_x8h'].weights)
pprint.pprint(opt.ir8.knobs['on_sep8h'].weights)


# In[14]:


opt.ir8.plot(yl='x y')


# ## Check strengths

# In[15]:


opt.set_circuits("data/lhccircuits.json")


# In[16]:


if 0 :
    opt50m.model['on_x8h'] = -300
    opt50m.model['on_x8v'] = 0.0
    opt50m.model['on_sep8v'] = -1
    opt50m.model['on_sep8h'] = 0

    # --- print current value of the kicker with this knob settings:
    print(opt50m.model['acbyhs5.r8b1'])


# In[17]:


def check_strengths_at_knob_value(aknob, circ, value):
    ''' Check strengths at value for the knob'''
    for k, w in aknob.weights.items():
        _klimits = circ.madname[k].get_klimits()
        _setting = w*value
        if min(_klimits) <= w*value <= max(_klimits) :
            print(f'\t {k:>15s} = {w*value/1.0e-6:12.3f} [urad] - {_klimits}')
        else:
            print(f'\t {k:>15s} = {w*value/1.0e-6:12.3f} [urad] - {_klimits} - out of bounds')

    return


# In[18]:


on_x8h = opt.ir8.knobs['on_x8h']
on_x8h.weights


# In[19]:


check_strengths_at_knob_value(on_x8h, opt.circuits, 200)


# In[20]:


# --- update optics from model
# opt50m.ir8.update()
# opt50m.ir8.strengths


# ## Add new knob in LHCOptics
# 
# - example constructed from Riccardo.
# 
# - Update the existing knob `on_x8h` to allow reaching -300 urad maintaining `acbyhs5.r8b1`
# 
# - the new knob uses `acbch6.r8b1`  

# In[21]:


type(on_x8h)


# In[22]:


print(on_x8h.const)
print(on_x8h.value)
on_x8h.weights


# In[23]:


print(f' --- acbyhs5.r8b1: @200 urad = {on_x8h.weights["acbyhs5.r8b1"]*200}')
print(f' --- acbyhs5.r8b1: @300 urad = {on_x8h.weights["acbyhs5.r8b1"]*300}')


# In[24]:


# --- maintain acbyhs5.r8b1 to the 200 urad settings and scale it to 300 urad

on_x8h.const.append("acbyhs5.r8b1") # remember to do it again or check
on_x8h.weights["acbyhs5.r8b1"]*=200/300
on_x8h.weights


# In[25]:


print(f' --- acbyhs5.r8b1: @200 urad = {on_x8h.weights["acbyhs5.r8b1"]*200}')
print(f' --- acbyhs5.r8b1: @300 urad = {on_x8h.weights["acbyhs5.r8b1"]*300}')


# In[26]:


on_x8h.weights["acbch6.r8b1"]=2.49675503913e-07
on_x8h.weights


# In[27]:


opt.update_model()
on_x8h.match()
opt.ir8.update_knobs()


# In[28]:


on_x8h.plot()
print(on_x8h.weights["acbyhs5.r8b1"]*300)


# In[29]:


check_strengths_at_knob_value(opt.ir8.knobs['on_x8h'], opt.circuits, 300)


# In[30]:


opt.to_madx(output="./data/vdmoptics_2025.9")


# ## Add new knob in MAD-X
# 
# To make this work, I had to do a symlink : acc-models-lhc --> ~/.local/acc-models-lhc/2025

# In[35]:


get_ipython().system(' ls acc-models-lhc/scenarios/cycle/vdm/desqueeze/')


# In[32]:


get_ipython().system(' cat acc-models-lhc/2025/operation/optics/R2025hRP_A19_2mC19_2mA19_2mL50m.madx')


# In[ ]:


from cpymad.madx import Madx

mymad  = Madx()
mymad.options['echo'] = False
mymad.options['warn'] = False
mymad.options['info'] = False
# 111 = 50m, 84 = 44m, 42 = 34m 0 = 24m
mymad.call("acc-models-lhc/scenarios/cycle/vdm/desqueeze/42/model.madx")
mymad.use('lhcb1')
mymad.twiss()


# In[ ]:


mymad.input(f''' 
    Xangh:=-200.*on_x8h-100*on_x8hb; Xangv:= 200.*on_x8v;
    Pseph:= 1.00*on_sep8h; Psepv:=-1.00*on_sep8v;
    Xangs:=-10.*on_a8; Psepx:=0.50*on_o8; 
    offseth:=1.0*on_oh8; offsetv:= 1.0*on_ov8;
    ''')
mymad.input('value, scale0, on_x8h, on_sep8h, on_sep8v, on_xip8b1;')
mymad.input('value, acbxh1.l8, acbxh1.r8;')


# In[ ]:


mymad.input(f'''
    show, acbxh1.l8, acbxh2.l8, acbxh3.l8;
    show, acbxh1.r8, acbxh2.r8, acbxh3.r8;
    ''')


# In[ ]:


# ---  create new knob on_x8hb : adds on top of on_x8h to boost the crossing to 300 urad 
#      acts only on B1

# on_x8h runs from 0 to -200; 
# on_x8hb runs from 0 to -100; so combined run to -300 urad

# for Q4 the contribution of the on_x8h beyond -200 is removed by the on_x8hb ramping to -100 

mymad.input(f'''
    add2expr, var=acbyhs4.l8b1, expr=acbyhs4.l8b1xb*on_x8hb;
    add2expr, var=acbyhs4.r8b1, expr=acbyhs4.r8b1xb*on_x8hb;
    add2expr, var=acbyhs4.l8b2, expr=acbyhs4.l8b2xb*on_x8hb;
    add2expr, var=acbyhs4.r8b2, expr=acbyhs4.r8b2xb*on_x8hb;
    
    add2expr, var=acbchs5.l8b1, expr=acbchs5.l8b1xb*on_x8hb;
    add2expr, var=acbyhs5.r8b1, expr=acbyhs5.r8b1xb*on_x8hb;
    add2expr, var=acbchs5.l8b2, expr=acbchs5.l8b2xb*on_x8hb;
    add2expr, var=acbyhs5.r8b2, expr=acbyhs5.r8b2xb*on_x8hb;
    add2expr, var=acbch5.l8b1, expr=acbch5.l8b1xb*on_x8hb;
    add2expr, var=acbwh5.l8b1, expr=acbwh5.l8b1xb*on_x8hb;
    
    add2expr, var=acbch6.r8b1,  expr=acbch6.r8b1xb*on_x8hb;

    ''')


# In[ ]:


mymad.input(f'''
            
    scale0 :=1;
    limitMCBXH:=  63.5988*1.e-6*scale0;
    limitMCBXV:=  67.0164*1.e-6*scale0;
    limitMCBY :=  96.3*1.e-6*scale0;
    limitMCB  :=  80.8*1.e-6*scale0;
    limitMCBC :=  89.87*1.e-6*scale0;
    limitMCBW :=  80.14*1.e-6*scale0;
    limitMCBC_1.9K := 119.95*1.e-6*scale0;

    
    use, period=lhcb1,range=s.ds.l8.b1/e.ds.r8.b1;
    
    acbyhs4.r8b1xb := 0; acbch6.r8b1xb := 0;
 
       
    on_x8hb = 1;
    on_x8h=1; ! this is required in order to have the MCBX preset
    on_sep8v=1;on_x8v=0;on_sep8h=0;on_a8=0;on_o8=0;on_oh8=0;on_ov8=0;
        
    match, sequence=lhcb1,orbit, betx=1, bety=1, x=0, y=0, px=0, py=0;
    weight, x=1, px=10, y=1, py=10;
    
    constraint, sequence=lhcb1,range=ip8, x  =   0.001*Pseph + 0.001*Psepx + 0.001*offseth,
                                            y  =   0.001*Psepv  + 0.001*offsetv,
                                            px =   0.000001*Xangh,
				                            py =   0.000001*Xangv + 0.000001*Xangs;
    constraint, sequence=lhcb1,range=E.DS.R8.B1,x = 0.0, px = 0.0;
    constraint, sequence=lhcb1,range=E.DS.R8.B1,y = 0.0, py = 0.0;

    vary, name=acbyhs4.l8b1xb, step=1.e-5;
    vary, name=acbyhs4.r8b1xb, step=1.e-5;
    vary, name=acbyvs4.l8b1s, step=1.0e-5;
    vary, name=acbyvs4.r8b1s, step=1.0e-5;
    
    vary, name=acbchs5.l8b1x, step=1.e-5;
    !vary, name=acbyhs5.r8b1x, step=1.e-5;
    vary, name=acbcvs5.l8b1s, step=1.e-5;
    vary, name=acvyvs5.r8b1s, step=1.e-5;
    
    vary, name=acbch6.r8b1xb,  step=1.e-5;
    lmdif, calls = 2000, tolerance=1.e-41;
    endmatch;
    ''')


# In[ ]:


import sys
sys.path.append('/Users/iliasefthymiopoulos/Work/lhc-special-optics/soptlib')

import soptlib 
soptlib.__version__
pm = soptlib.pm
print(f' --- loaded pmadx version {pm.__version__}')


# In[ ]:


nir=8
mymad.globals['on_x8hb'] = 100
mymad.globals['on_x8h'] = 200
mymad.globals['on_sep8v'] = 0
mymad.globals['on_x8h'] = 0
mymad.globals['on_xip8b1'] = 0

mymad.input('value, on_x8hb;')
_tt, _ss = pm.twissLHC(mymad)
print(_ss[['q1','q2']])

ipbetas = soptlib.get_beta_star_at_ip(_tt)
pprint.pprint(ipbetas)

betxip8 = ipbetas['lhcb1']['ip8']['betx'] 
ax = _tt.loc['s.ds.l8.b1':'e.ds.r8.b1'].plot(x='s',y='x')
_tt.loc['s.ds.l8.b1':'e.ds.r8.b1'].plot(x='s',y='y', color='red', ax=ax)
ax.axhline(mymad.globals['on_sep8h']*1.0e-3, color='magenta', linestyle='--')
ax.axvline(_tt.loc['ip8'].s[0], color='green')
print(f' -- ip location {_tt.loc["ip8"].s[0]}')
ax.set_title(f'VdM - IP{nir} b*={betxip8} on_sep8h={mymad.globals["on_sep8h"]}, on_x8h={mymad.globals["on_x8h"]} on_x8hb={mymad.globals["on_x8hb"]}')
ax.grid()


# In[ ]:


_ttb1 = _tt[_tt.beam == 'lhcb1']
_ipos_ip8 = [i for i,k in enumerate(_ttb1.index.values) if 'ip8' in k][0]
print(_ipos_ip8)
_dfy = _ttb1.iloc[_ipos_ip8-10:_ipos_ip8+10][['s','name','x','y']].copy()
_dfy['sip'] = _dfy['s'].apply(lambda r : r - _dfy.loc['ip8'].s)
_dfy['xip'] = _dfy['x'].apply(lambda r : r - _dfy.loc['ip8'].x)
_dfy['x-ang'] = np.arctan2(_dfy.xip, _dfy.sip)

_dfy


# In[ ]:


mymad.input('show, acbyhs4.r8b1;')


# In[ ]:


mymad.globals['acbyhs5.r8b1']


# In[ ]:





# In[ ]:


if 0:
    _cir = opt50m.circuits.madname['acbyhs5.r8b1']
    _cir.get_klimits(pc=6.8e12)
    for k,v in opt50m.ir8.strengths.items():
        _cir = opt50m.circuits.madname[k]
        print(f' --- {k=} : {_cir=} : {v=} {_cir.get_klimits(pc=6.8e12)}')


# In[ ]:


if 0 : 
    for k,v in opt50m.ir8.strengths.items():
        _cir = opt50m.circuits.madname[k]
        if not k.startswith('a'):
            continue
        if v != 0:
            _ic = _cir.get_current(abs(v))
        else:
            _ic = 0
        print(k,v, _ic)
        assert _cir.imin < _cir.get_current(v) < _cir.imax, f'Out of range {k=}, {v=}, {_ic=} {_cir.imin=}, {_cir.imax=}'


# In[ ]:


print(f'--- : on_xh8h:')
pprint.pprint(opt50m.ir8.knobs['on_x8h'].weights)
print(f'--- : on_xh8v:')
pprint.pprint(opt50m.ir8.knobs['on_x8v'].weights)


# ## Check new optics files with MAD-x

# In[37]:


import sys
sys.path.append('/Users/iliasefthymiopoulos/Work/lhc-special-optics/soptlib')

import soptlib 
soptlib.__version__
pm = soptlib.pm
print(f' --- loaded pmadx version {pm.__version__}')


# In[ ]:


from cpymad.madx import Madx

mymad  = Madx()
mymad.options['echo'] = False
mymad.options['warn'] = False
mymad.options['info'] = False
# mymad.call("acc-models-lhc/scenarios/cycle/vdm/desqueeze/111/model.madx")  # 50m 
# mymad.call("acc-models-lhc/scenarios/cycle/vdm/desqueeze/84/model.madx") # 44m
mymad.call("acc-models-lhc/scenarios/cycle/vdm/desqueeze/42/model.madx") # 34m
mymad.use('lhcb1')

mymad.twiss()


# In[43]:


nir=8
mymad.globals['on_x8h'] = -300
mymad.globals['on_sep8v'] = -1

mymad.input('value, on_x8h;')
_tt, _ss = pm.twissLHC(mymad)
print(_ss[['q1','q2']])

ipbetas = soptlib.get_beta_star_at_ip(_tt)
pprint.pprint(ipbetas)

betxip8 = ipbetas['lhcb1']['ip8']['betx'] 
ax = _tt.loc['s.ds.l8.b1':'e.ds.r8.b1'].plot(x='s',y='x')
_tt.loc['s.ds.l8.b1':'e.ds.r8.b1'].plot(x='s',y='y', color='red', ax=ax)
ax.axhline(mymad.globals['on_sep8h']*1.0e-3, color='magenta', linestyle='--')
ax.axvline(_tt.loc['ip8'].s[0], color='green')
print(f' -- ip location {_tt.loc["ip8"].s[0]}')
ax.set_title(f'VdM - IP{nir} b*={betxip8} on_sep8h={mymad.globals["on_sep8h"]}, on_x8h={mymad.globals["on_x8h"]}')
ax.grid()


# In[41]:


# mymad.call(file="./data/vdmoptics_2025.11")
# mymad.call(file="./data/vdmoptics_2025.10")
mymad.call(file="./data/vdmoptics_2025.9")


# In[42]:


mymad.input('show, acbch6.r8b1;')
mymad.input('show, acbyhs5.r8b1;')


# ## Add new knob in LHCOPTICS-obs
# 
# - add the new knob and then match the normal x-ing knobs
# 
# 
# 

# In[ ]:


opt50m.ir8.knobs['on_x8h']


# In[ ]:


import xtrack as xt
from lhcoptics import Knob


opt50m.knobs["on_x8h_bmp"]=Knob("on_xh8_bmp",0,{"kqx.l1": 32})



# In[ ]:


all_knobs_ip8 = [
        'acbxh1.l8', 'acbxh2.l8', 'acbxh3.l8',
        'acbxv1.l8', 'acbxv2.l8', 'acbxv3.l8', 
        
        'acbxh1.r8', 'acbxh2.r8', 'acbxh3.r8', 
        'acbxv1.r8', 'acbxv2.r8', 'acbxv3.r8',

        'acbyhs4.l8b1', 'acbyhs4.l8b2',
        'acbyvs4.l8b1', 'acbyvs4.l8b2', 
    
        'acbyhs4.r8b1'  'acbyhs4.r8b2',
        'acbyvs4.r8b1', 'acbyvs4.r8b2',
                
        'acbchs5.l8b1', 'acbchs5.l8b2',
        'acbcvs5.l8b1', 'acbcvs5.l8b2',
                    
        'acbyhs5.r8b1', 'acbyhs5.r8b2', 
        'acbyvs5.r8b1', 'acbyvs5.r8b2', 
            
        'xip8b1', 'yip8b1', 'pxip8b1', 'pyip8b1',
        'xip8b2', 'yip8b2', 'pxip8b2', 'pyip8b2',
    ]


# In[ ]:


collider = opt50m.model.env
get_ipython().run_line_magic('pinfo', 'collider.match_knob')


# In[ ]:


def match_xing_ip8(collider):
    
    all_knobs_ip8 = [
            'acbxh1.l8', 'acbxh2.l8', 'acbxh3.l8',
            'acbxv1.l8', 'acbxv2.l8', 'acbxv3.l8', 
            
            'acbxh1.r8', 'acbxh2.r8', 'acbxh3.r8', 
            'acbxv1.r8', 'acbxv2.r8', 'acbxv3.r8',

            'acbyhs4.l8b1', 'acbyhs4.l8b2',
            'acbyvs4.l8b1', 'acbyvs4.l8b2', 
       
            'acbyhs4.r8b1'  'acbyhs4.r8b2',
            'acbyvs4.r8b1', 'acbyvs4.r8b2',
                  
            'acbchs5.l8b1', 'acbchs5.l8b2',
            'acbcvs5.l8b1', 'acbcvs5.l8b2',
                       
            'acbyhs5.r8b1', 'acbyhs5.r8b2', 
            'acbyvs5.r8b1', 'acbyvs5.r8b2', 
             
            'xip8b1', 'yip8b1', 'pxip8b1', 'pyip8b1',
            'xip8b2', 'yip8b2', 'pxip8b2', 'pyip8b2',
        ]

    # kill all existing knobs
    for kk in all_knobs_ip8:
        collider.vars[kk] = 0

    twinit_zero_orbit = [xt.TwissInit(), xt.TwissInit()]

    targets_close_bump = [
        xt.TargetSet(line='lhcb1', at=xt.END, x=0, px=0, y=0, py=0),
        xt.TargetSet(line='lhcb2', at=xt.END, x=0, px=0, y=0, py=0),
    ]

    bump_range_ip8 = {
        'start': ['s.ds.l8.b1', 's.ds.l8.b2'],
        'end': ['e.ds.r8.b1', 'e.ds.r8.b2'],
    }

    correctors_ir8_single_beam_h = [
        'acbyhs4.l8b1', 'acbyhs4.r8b2', 'acbyhs4.l8b2', 'acbyhs4.r8b1',
        'acbchs5.l8b2', 'acbchs5.l8b1', 'acbyhs5.r8b1', 'acbyhs5.r8b2']

    correctors_ir8_single_beam_v = [
        'acbyvs4.l8b1', 'acbyvs4.r8b2', 'acbyvs4.l8b2', 'acbyvs4.r8b1',
        'acbcvs5.l8b2', 'acbcvs5.l8b1', 'acbyvs5.r8b1', 'acbyvs5.r8b2']

    correctors_ir8_common_h = [
        'acbxh1.l8', 'acbxh2.l8', 'acbxh3.l8', 'acbxh1.r8', 'acbxh2.r8', 'acbxh3.r8']

    correctors_ir8_common_v = [
        'acbxv1.l8', 'acbxv2.l8', 'acbxv3.l8', 'acbxv1.r8', 'acbxv2.r8', 'acbxv3.r8']


    ##############################
    # Match crossing angle knobs #
    ##############################

    angle_match_ip8 = 170e-6

    # ---------- on_x8h ----------

    opt_x8h = collider.match_knob(
        knob_name='on_x8h', knob_value_end=(angle_match_ip8 * 1e6),
        targets=(targets_close_bump + [
            xt.TargetSet(line='lhcb1', at='ip8',  x=0, px=angle_match_ip8),
            xt.TargetSet(line='lhcb2', at='ip8',  x=0, px=-angle_match_ip8),
        ]),
        vary=[
            xt.VaryList(correctors_ir8_single_beam_h),
            xt.VaryList(correctors_ir8_common_h, tag='mcbx')],
        run=False, init=twinit_zero_orbit, **bump_range_ip8,
    )

    # Set mcbx by hand (reduce value by 10, to test matching algorithm)
    testkqx8=abs(collider.varval['kqx.l8'])*7000./0.3
    acbx_xing_ir8 = 1.0e-6 if testkqx8 > 210. else 11.0e-6 # Value for 170 urad crossing

    # Set mcbx by hand
    for icorr in [1, 2, 3]:
        collider.vars[f'acbxh{icorr}.l8_from_on_x8h'] = acbx_xing_ir8 * angle_match_ip8 / 170e-6
        collider.vars[f'acbxh{icorr}.r8_from_on_x8h'] = -acbx_xing_ir8 * angle_match_ip8 / 170e-6

    #   (reduce value by 10, to test matching algorithm)
    #   collider.vars[f'acbxh{icorr}.l8_from_on_x8h'] = acbx_xing_ir8 * angle_match_ip8 / 170e-6 * 0.1
    #   collider.vars[f'acbxh{icorr}.r8_from_on_x8h'] = -acbx_xing_ir8 * angle_match_ip8 / 170e-6 * 0.1

    # First round of optimization without changing mcbx
    opt_x8h.disable_vary(tag='mcbx')
    opt_x8h.step(3) # perform 3 steps without checking for convergence

    # Link all mcbx strengths to the first one
    collider.vars['acbxh2.l8_from_on_x8h'] =  collider.vars['acbxh1.l8_from_on_x8h']
    collider.vars['acbxh3.l8_from_on_x8h'] =  collider.vars['acbxh1.l8_from_on_x8h']
    collider.vars['acbxh2.r8_from_on_x8h'] = -collider.vars['acbxh1.l8_from_on_x8h']
    collider.vars['acbxh3.r8_from_on_x8h'] = -collider.vars['acbxh1.l8_from_on_x8h']
    collider.vars['acbxh1.r8_from_on_x8h'] = -collider.vars['acbxh1.l8_from_on_x8h']

    # Enable first mcbx knob (which controls the others)
    assert opt_x8h.vary[8].name == 'acbxh1.l8_from_on_x8h'
    opt_x8h.vary[8].active = True

    # Solve and generate knob
    opt_x8h.solve()
    opt_x8h.generate_knob()

    # ---------- on_x8v ----------

    opt_x8v = collider.match_knob(
        knob_name='on_x8v', knob_value_end=(angle_match_ip8 * 1e6),
        targets=(targets_close_bump + [
            xt.TargetSet(line='lhcb1', at='ip8', y=0, py=angle_match_ip8),
            xt.TargetSet(line='lhcb2', at='ip8', y=0, py=-angle_match_ip8),
        ]),
        vary=[
            xt.VaryList(correctors_ir8_single_beam_v),
            xt.VaryList(correctors_ir8_common_v, tag='mcbx')],
        run=False, init=twinit_zero_orbit, **bump_range_ip8,
    )

    # Set mcbx by hand
    testkqx8=abs(collider.varval['kqx.l8'])*7000./0.3
    acbx_xing_ir8 = 1.0e-6 if testkqx8 > 210. else 11.0e-6 # Value for 170 urad crossing
    # Set MCBX by hand
    for icorr in [1, 2, 3]:
        collider.vars[f'acbxv{icorr}.l8_from_on_x8v'] = acbx_xing_ir8 * angle_match_ip8 / 170e-6
        collider.vars[f'acbxv{icorr}.r8_from_on_x8v'] = -acbx_xing_ir8 * angle_match_ip8 / 170e-6

    # First round of optimization without changing mcbx
    opt_x8v.disable_vary(tag='mcbx')
    opt_x8v.step(3) # perform 3 steps without checking for convergence

    # Solve with all vary active and generate knob
    opt_x8v.enable_vary(tag='mcbx')
    opt_x8v.solve()
    opt_x8v.generate_knob()

    v = collider.vars
    f = collider.functions
    v['phi_ir8'] = 0.
    for irn in [8]:
        v[f'cphi_ir{irn}'] = f.cos(v[f'phi_ir{irn}'] * np.pi / 180.)
        v[f'sphi_ir{irn}'] = f.sin(v[f'phi_ir{irn}'] * np.pi / 180.)
        v[f'on_x{irn}h']   =  v[f'on_x{irn}'] * v[f'cphi_ir{irn}']
        v[f'on_x{irn}v']   =  v[f'on_x{irn}'] * v[f'sphi_ir{irn}']
        v[f'on_sep{irn}h'] = -v[f'on_sep{irn}'] * v[f'sphi_ir{irn}']
        v[f'on_sep{irn}v'] =  v[f'on_sep{irn}'] * v[f'cphi_ir{irn}']
        v[f'on_o{irn}h']   =  v[f'on_o{irn}'] * v[f'cphi_ir{irn}']
        v[f'on_o{irn}v']   =  v[f'on_o{irn}'] * v[f'sphi_ir{irn}']
        v[f'on_a{irn}h']   = -v[f'on_a{irn}'] * v[f'sphi_ir{irn}']
        v[f'on_a{irn}v']   =  v[f'on_a{irn}'] * v[f'cphi_ir{irn}']

    opt = {
        'on_x8h': opt_x8h, 'on_x8v': opt_x8v
    }
    return opt


# In[ ]:


# mtcknbip8 = opt50m.ir8.match_knobs(knob_name='on_x8h')


# In[ ]:


# collider['b1'].vars[]

# collider.elements[0]


# In[ ]:


_aux = match_xing_ip8(collider)


# ## Create collision optics  

# In[ ]:


vdmopt_50m = LHCOptics.from_json("data/opt_2025_50m.json")
# vdmopt_50m.set_xsuite_model("data/lhc.json")
vdmopt_50m.set_xsuite_model("acc-models-lhc/xsuite/lhc.json")

twb1, twb2 = vdmopt_50m.twiss()
vdmopt_50m


# In[ ]:


vdmopt_50m.ir8.knobs


# In[ ]:


vdmopt_50m.ir8.plot()


# In[ ]:


vdmopt_50m.check()


# In[ ]:


vdmopt_50m.ir8.knobs


# In[ ]:


vdmopt_50m.ir8.plot(yl='x y')


# In[ ]:


vdmopt_50m.model['on_x8h'] = -300
vdmopt_50m.ir8.plot(yl='x y')


# In[ ]:


vdmopt_50m.update()
vdmopt_50m.ir8.knobs


# In[ ]:


vdmopt_50m.set_circuits("data/lhccircuits.json")
vdmopt_50m.circuits


# In[ ]:


vdmopt_50m.ir8.strengths


# In[ ]:


for k,v in vdmopt_50m.ir8.strengths.items():
    _cir = vdmopt_50m.circuits.madname[k]
    if v != 0:
        _ic = _cir.get_current(abs(v))
    else:
        _ic = 0
    print(k,v, _ic)
    assert _cir.imin < _cir.get_current(v) < _cir.imax, f'Out of range {k=}, {v=}, {_ic=} {_cir.imin=}, {_cir.imax=}'


# In[ ]:


cir=vdmopt_50m.circuits.madname["acbyhs5.r8b1"]
print(cir.get_current(vdmopt_50m.ir8.strengths['acbyhs5.r8b1']))


# In[ ]:


cir=vdmopt_50m.circuits.madname["acbxh1.l8"]
print(cir.get_current(vdmopt_50m.ir8.strengths['acbxh1.l8']))


# In[ ]:


vdmopt_50m.model['on_x8h'] = -300
vdmopt_50m.update()
vdmopt_50m.ir8.update_strengths()


# In[ ]:


vdmopt_50m.ir8.knobs


# In[ ]:


vdmopt_50m.ir8.strengths


# In[ ]:


# 'RCBYHS5.R8B'


# In[ ]:


cir.imax


# ## - Create optics from MAD-X file

# In[ ]:


if 0: 
    from cpymad.madx import Madx

    mymad = Madx()

    mymad.input(f'''
        call, file="acc-models-lhc/lhc.seq";

        beam, sequence=lhcb1, bv= 1, particle=proton, charge=1, mass=0.938272046,
        pc= 450.0,   npart=1.2e11,kbunch=2556, ex=5.2126224777777785e-09,ey=5.2126224777777785e-09;
        beam, sequence=lhcb2, bv=-1, particle=proton, charge=1, mass=0.938272046,
        pc= 450.0,   npart=1.2e11,kbunch=2556, ex=5.2126224777777785e-09,ey=5.2126224777777785e-09;

        call,file="/afs/cern.ch/work/e/efthymio/public/vdm_optics/vdm2025/opticsfile_2025.11";

        on_x8h := -300;
        call,file="acc-models-lhc/aperture/aperture.b1.madx";
        call,file="acc-models-lhc/aperture/aperture.b2.madx";
        call,file="acc-models-lhc/aperture/aper_tol.b1.madx";
        call,file="acc-models-lhc/aperture/aper_tol.b2.madx";
        use,sequence=lhcb1;
        use,sequence=lhcb2;
        ''')

    mymad.twiss()
    vdmopt_50m = LHCOptics.from_madx(mymad)


# In[ ]:


vdmopt_50m.ir8.plot()


# In[ ]:


get_ipython().run_line_magic('pinfo2', 'vdmopt_50m.ir8.match_knobs')


# In[ ]:





# In[ ]:


def match_xing_ip8(collider):
    
    all_knobs_ip8 = [
            'acbxh1.l8', 'acbxh2.l8', 'acbxh3.l8',
            'acbxv1.l8', 'acbxv2.l8', 'acbxv3.l8', 
            
            'acbxh1.r8', 'acbxh2.r8', 'acbxh3.r8', 
            'acbxv1.r8', 'acbxv2.r8', 'acbxv3.r8',

            'acbyhs4.l8b1', 'acbyhs4.l8b2',
            'acbyvs4.l8b1', 'acbyvs4.l8b2', 
       
            'acbyhs4.r8b1'  'acbyhs4.r8b2',
            'acbyvs4.r8b1', 'acbyvs4.r8b2',
                  
            'acbchs5.l8b1', 'acbchs5.l8b2',
            'acbcvs5.l8b1', 'acbcvs5.l8b2',
                       
            'acbyhs5.r8b1', 'acbyhs5.r8b2', 
            'acbyvs5.r8b1', 'acbyvs5.r8b2', 
             
            'xip8b1', 'yip8b1', 'pxip8b1', 'pyip8b1',
            'xip8b2', 'yip8b2', 'pxip8b2', 'pyip8b2',
        ]


    # kill all existing knobs
    for kk in all_knobs_ip8:
        collider.vars[kk] = 0

    twinit_zero_orbit = [xt.TwissInit(), xt.TwissInit()]

    targets_close_bump = [
        xt.TargetSet(line='lhcb1', at=xt.END, x=0, px=0, y=0, py=0),
        xt.TargetSet(line='lhcb2', at=xt.END, x=0, px=0, y=0, py=0),
    ]

    bump_range_ip8 = {
        'start': ['s.ds.l8.b1', 's.ds.l8.b2'],
        'end': ['e.ds.r8.b1', 'e.ds.r8.b2'],
    }

    correctors_ir8_single_beam_h = [
        'acbyhs4.l8b1', 'acbyhs4.r8b2', 'acbyhs4.l8b2', 'acbyhs4.r8b1',
        'acbchs5.l8b2', 'acbchs5.l8b1', 'acbyhs5.r8b1', 'acbyhs5.r8b2']

    correctors_ir8_single_beam_v = [
        'acbyvs4.l8b1', 'acbyvs4.r8b2', 'acbyvs4.l8b2', 'acbyvs4.r8b1',
        'acbcvs5.l8b2', 'acbcvs5.l8b1', 'acbyvs5.r8b1', 'acbyvs5.r8b2']

    correctors_ir8_common_h = [
        'acbxh1.l8', 'acbxh2.l8', 'acbxh3.l8', 'acbxh1.r8', 'acbxh2.r8', 'acbxh3.r8']

    correctors_ir8_common_v = [
        'acbxv1.l8', 'acbxv2.l8', 'acbxv3.l8', 'acbxv1.r8', 'acbxv2.r8', 'acbxv3.r8']

    #########################
    # Match IP offset knobs #
    #########################

    offset_match = 0.5e-3

    # ---------- on_o8v ----------

    opt_o8v = collider.match_knob(
        knob_name='on_o8v', knob_value_end=(offset_match * 1e3),
        targets=(targets_close_bump + [
            xt.TargetSet(line='lhcb1', at='ip8', y=offset_match, py=0),
            xt.TargetSet(line='lhcb2', at='ip8', y=offset_match, py=0),
        ]),
        vary=xt.VaryList(correctors_ir8_single_beam_v),
        run=False, init=twinit_zero_orbit, **bump_range_ip8,
    )
    opt_o8v.solve()
    opt_o8v.generate_knob()

    # ---------- on_o8h ----------

    opt_o8h = collider.match_knob(
        knob_name='on_o8h', knob_value_end=(offset_match * 1e3),
        targets=(targets_close_bump + [
            xt.TargetSet(line='lhcb1', at='ip8', x=offset_match, px=0),
            xt.TargetSet(line='lhcb2', at='ip8', x=offset_match, px=0),
        ]),
        vary=xt.VaryList(correctors_ir8_single_beam_h),
        run=False, init=twinit_zero_orbit, **bump_range_ip8,
    )
    opt_o8h.solve()
    opt_o8h.generate_knob()

    ##############################
    # Match angular offset knobs #
    ##############################

    ang_offset_match = 30e-6

    # ---------- on_a8h ----------

    opt_a8h = collider.match_knob(
        knob_name='on_a8h', knob_value_end=(ang_offset_match * 1e6),
        targets=(targets_close_bump + [
            xt.TargetSet(line='lhcb1', at='ip8', x=0, px=ang_offset_match),
            xt.TargetSet(line='lhcb2', at='ip8', x=0, px=ang_offset_match),
        ]),
        vary=xt.VaryList(correctors_ir8_single_beam_h),
        run=False, init=twinit_zero_orbit, **bump_range_ip8,
    )

    opt_a8h.solve()
    opt_a8h.generate_knob()

    # ---------- on_a8v ----------

    opt_a8v = collider.match_knob(
        knob_name='on_a8v', knob_value_end=(ang_offset_match * 1e6),
        targets=(targets_close_bump + [
            xt.TargetSet(line='lhcb1', at='ip8', y=0, py=ang_offset_match),
            xt.TargetSet(line='lhcb2', at='ip8', y=0, py=ang_offset_match),
        ]),
        vary=xt.VaryList(correctors_ir8_single_beam_v),
        run=False, init=twinit_zero_orbit, **bump_range_ip8,
    )

    opt_a8v.solve()
    opt_a8v.generate_knob()

    ##############################
    # Match crossing angle knobs #
    ##############################

    angle_match_ip8 = 170e-6

    # ---------- on_x8h ----------

    opt_x8h = collider.match_knob(
        knob_name='on_x8h', knob_value_end=(angle_match_ip8 * 1e6),
        targets=(targets_close_bump + [
            xt.TargetSet(line='lhcb1', at='ip8',  x=0, px=angle_match_ip8),
            xt.TargetSet(line='lhcb2', at='ip8',  x=0, px=-angle_match_ip8),
        ]),
        vary=[
            xt.VaryList(correctors_ir8_single_beam_h),
            xt.VaryList(correctors_ir8_common_h, tag='mcbx')],
        run=False, init=twinit_zero_orbit, **bump_range_ip8,
    )

    # Set mcbx by hand (reduce value by 10, to test matching algorithm)
    testkqx8=abs(collider.varval['kqx.l8'])*7000./0.3
    acbx_xing_ir8 = 1.0e-6 if testkqx8 > 210. else 11.0e-6 # Value for 170 urad crossing

    # Set mcbx by hand
    for icorr in [1, 2, 3]:
        collider.vars[f'acbxh{icorr}.l8_from_on_x8h'] = acbx_xing_ir8 * angle_match_ip8 / 170e-6
        collider.vars[f'acbxh{icorr}.r8_from_on_x8h'] = -acbx_xing_ir8 * angle_match_ip8 / 170e-6

    #   (reduce value by 10, to test matching algorithm)
    #   collider.vars[f'acbxh{icorr}.l8_from_on_x8h'] = acbx_xing_ir8 * angle_match_ip8 / 170e-6 * 0.1
    #   collider.vars[f'acbxh{icorr}.r8_from_on_x8h'] = -acbx_xing_ir8 * angle_match_ip8 / 170e-6 * 0.1

    # First round of optimization without changing mcbx
    opt_x8h.disable_vary(tag='mcbx')
    opt_x8h.step(3) # perform 3 steps without checking for convergence

    # Link all mcbx strengths to the first one
    collider.vars['acbxh2.l8_from_on_x8h'] =  collider.vars['acbxh1.l8_from_on_x8h']
    collider.vars['acbxh3.l8_from_on_x8h'] =  collider.vars['acbxh1.l8_from_on_x8h']
    collider.vars['acbxh2.r8_from_on_x8h'] = -collider.vars['acbxh1.l8_from_on_x8h']
    collider.vars['acbxh3.r8_from_on_x8h'] = -collider.vars['acbxh1.l8_from_on_x8h']
    collider.vars['acbxh1.r8_from_on_x8h'] = -collider.vars['acbxh1.l8_from_on_x8h']

    # Enable first mcbx knob (which controls the others)
    assert opt_x8h.vary[8].name == 'acbxh1.l8_from_on_x8h'
    opt_x8h.vary[8].active = True

    # Solve and generate knob
    opt_x8h.solve()
    opt_x8h.generate_knob()

    # ---------- on_x8v ----------

    opt_x8v = collider.match_knob(
        knob_name='on_x8v', knob_value_end=(angle_match_ip8 * 1e6),
        targets=(targets_close_bump + [
            xt.TargetSet(line='lhcb1', at='ip8', y=0, py=angle_match_ip8),
            xt.TargetSet(line='lhcb2', at='ip8', y=0, py=-angle_match_ip8),
        ]),
        vary=[
            xt.VaryList(correctors_ir8_single_beam_v),
            xt.VaryList(correctors_ir8_common_v, tag='mcbx')],
        run=False, init=twinit_zero_orbit, **bump_range_ip8,
    )

    # Set mcbx by hand
    testkqx8=abs(collider.varval['kqx.l8'])*7000./0.3
    acbx_xing_ir8 = 1.0e-6 if testkqx8 > 210. else 11.0e-6 # Value for 170 urad crossing
    # Set MCBX by hand
    for icorr in [1, 2, 3]:
        collider.vars[f'acbxv{icorr}.l8_from_on_x8v'] = acbx_xing_ir8 * angle_match_ip8 / 170e-6
        collider.vars[f'acbxv{icorr}.r8_from_on_x8v'] = -acbx_xing_ir8 * angle_match_ip8 / 170e-6

    # First round of optimization without changing mcbx
    opt_x8v.disable_vary(tag='mcbx')
    opt_x8v.step(3) # perform 3 steps without checking for convergence

    # Solve with all vary active and generate knob
    opt_x8v.enable_vary(tag='mcbx')
    opt_x8v.solve()
    opt_x8v.generate_knob()

    ##########################
    # Match separation knobs #
    ##########################

    sep_match = 2e-3

    # ---------- on_sep8h ----------

    opt_sep8h = collider.match_knob(
        knob_name='on_sep8h', knob_value_end=(sep_match * 1e3),
        targets=(targets_close_bump + [
            xt.TargetSet(line='lhcb1', at='ip8',  x=sep_match, px=0),
            xt.TargetSet(line='lhcb2', at='ip8',  x=-sep_match, px=0),
        ]),
        vary=[
            xt.VaryList(correctors_ir8_single_beam_h),
            xt.VaryList(correctors_ir8_common_h, tag='mcbx')],
        run=False, init=twinit_zero_orbit, **bump_range_ip8,
    )

    # Set mcbx by hand
    testkqx8=abs(collider.varval['kqx.l8'])*7000./0.3
    acbx_sep_ir8 = 18e-6 if testkqx8 > 210. else 16e-6

    for icorr in [1, 2, 3]:
        collider.vars[f'acbxh{icorr}.l8_from_on_sep8h'] = acbx_sep_ir8 * sep_match / 2e-3
        collider.vars[f'acbxh{icorr}.r8_from_on_sep8h'] = acbx_sep_ir8 * sep_match / 2e-3

    # Match other correctors with fixed mcbx and generate knob
    opt_sep8h.disable_vary(tag='mcbx')
    opt_sep8h.solve()
    opt_sep8h.generate_knob()

    # ---------- on_sep8v ----------

    opt_sep8v = collider.match_knob(
        knob_name='on_sep8v', knob_value_end=(sep_match * 1e3),
        targets=(targets_close_bump + [
            xt.TargetSet(line='lhcb1', at='ip8',  y=sep_match, py=0),
            xt.TargetSet(line='lhcb2', at='ip8',  y=-sep_match, py=0),
        ]),
        vary=[
            xt.VaryList(correctors_ir8_single_beam_v),
            xt.VaryList(correctors_ir8_common_v, tag='mcbx')],
        run=False, init=twinit_zero_orbit, **bump_range_ip8,
    )

    # Set mcbx by hand
    testkqx8=abs(collider.varval['kqx.l8'])*7000./0.3
    acbx_sep_ir8 = 18e-6 if testkqx8 > 210. else 16e-6

    for icorr in [1, 2, 3]:
        collider.vars[f'acbxv{icorr}.l8_from_on_sep8v'] = acbx_sep_ir8 * sep_match / 2e-3
        collider.vars[f'acbxv{icorr}.r8_from_on_sep8v'] = acbx_sep_ir8 * sep_match / 2e-3

    # Match other correctors with fixed mcbx and generate knob
    opt_sep8v.disable_vary(tag='mcbx')
    opt_sep8v.solve()
    opt_sep8v.generate_knob()

    v = collider.vars
    f = collider.functions
    v['phi_ir8'] = 0.
    for irn in [8]:
        v[f'cphi_ir{irn}'] = f.cos(v[f'phi_ir{irn}'] * np.pi / 180.)
        v[f'sphi_ir{irn}'] = f.sin(v[f'phi_ir{irn}'] * np.pi / 180.)
        v[f'on_x{irn}h']   =  v[f'on_x{irn}'] * v[f'cphi_ir{irn}']
        v[f'on_x{irn}v']   =  v[f'on_x{irn}'] * v[f'sphi_ir{irn}']
        v[f'on_sep{irn}h'] = -v[f'on_sep{irn}'] * v[f'sphi_ir{irn}']
        v[f'on_sep{irn}v'] =  v[f'on_sep{irn}'] * v[f'cphi_ir{irn}']
        v[f'on_o{irn}h']   =  v[f'on_o{irn}'] * v[f'cphi_ir{irn}']
        v[f'on_o{irn}v']   =  v[f'on_o{irn}'] * v[f'sphi_ir{irn}']
        v[f'on_a{irn}h']   = -v[f'on_a{irn}'] * v[f'sphi_ir{irn}']
        v[f'on_a{irn}v']   =  v[f'on_a{irn}'] * v[f'cphi_ir{irn}']

    opt = {
        'on_o8h': opt_o8h, 'on_o8v': opt_o8v,
        'on_a8h': opt_a8h, 'on_a8v': opt_a8v,
        'on_x8h': opt_x8h, 'on_x8v': opt_x8v,
        'on_sep8h': opt_sep8h, 'on_sep8v': opt_sep8v,
    }
    return opt

