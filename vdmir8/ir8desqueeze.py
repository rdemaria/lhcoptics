#!/usr/bin/env python
# coding: utf-8

# # Desqueeze VDM optics for IR8

# In[1]:


# lhcoptics is installed within pip

from lhcoptics import LHCOptics
from lhcoptics import LHCCircuits



# In[2]:


vdmopt_inj = LHCOptics.from_json("data/opt_inj2025.json")
vdmopt_inj.set_xsuite_model("data/lhc.json")
twb1, twb2 = vdmopt_inj.twiss()
# vdmopt_inj


# In[3]:


vdmopt_inj.ir1.strengths


# In[4]:


vdmopt_inj.ir1.knob_names


# In[5]:


vdmopt_inj.plot()


# In[6]:


vdmopt_inj.ir8.plot()


# In[8]:


vdmopt_inj.ir8.knobs


# ## Create collision optics  

# In[12]:


vdmopt_50m = LHCOptics.from_json("data/opt_2025_50m.json")
# vdmopt_50m.set_xsuite_model("data/lhc.json")
vdmopt_50m.set_xsuite_model("acc-models-lhc/xsuite/lhc.json")

twb1, twb2 = vdmopt_50m.twiss()
vdmopt_50m


# In[13]:


vdmopt_50m.ir8.knobs


# In[14]:


vdmopt_50m.ir8.plot()


# In[15]:


vdmopt_50m.check()


# In[16]:


vdmopt_50m.ir8.knobs


# In[17]:


vdmopt_50m.ir8.plot(yl='x y')


# In[20]:


vdmopt_50m.model['on_x8h'] = -300
vdmopt_50m.ir8.plot(yl='x y')


# In[21]:


vdmopt_50m.update()
vdmopt_50m.ir8.knobs


# In[22]:


vdmopt_50m.set_circuits("data/lhccircuits.json")
vdmopt_50m.circuits


# In[23]:


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


# In[34]:


cir=vdmopt_50m.circuits.madname["acbyhs5.r8b1"]
print(cir.get_current(vdmopt_50m.ir8.strengths['acbyhs5.r8b1']))


# In[33]:


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


# In[26]:


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




