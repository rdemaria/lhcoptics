from lhcoptics import LHC, get_lsa
lhc=LHC()

lsa=get_lsa()


# 2025 - Proton Physics
lhc.y2025.add_cycle(
    name="pp",
    label="Proton Physics 2025",
    particles=["proton","proton"],
    charges=[1,1]
)
lhc.y2025.pp.refresh()
lhc.y2025.pp.add_process("ramp","RAMP-SQUEEZE-6.8TeV-ATS-2m-2025_V1")
lhc.y2025.pp.add_process("squeeze","SQUEEZE-6.8TeV-2m-1.2m-2025_V1")
lhc.y2025.pp.add_process("rotation","IP1-5-8-ROTATION-2025_V1")
lhc.y2025.pp.add_process("qchange","QCHANGE-6.8TeV-2025_V1")
lhc.y2025.pp.add_process("physics","PHYSICS-6.8TeV-1.2m-2025_V1")
lhc.y2025.pp.add_process("levelling","SQUEEZE-6.8TeV-1.2m-18cm-2025_V1")
lhc.y2025.pp.set_data_from_lsa()
lhc.y2025.pp.gen_repo_data()
#lhc.y2025.pp.gen_eos_data() done by gitlab actions


# 2025 - VDM
lsa.findBeamProcesses(".*vdm.*2025.*")
lhc.y2025.vdm.refresh()
lhc.y2025.vdm.add_process("ramp","RAMP-DESQUEEZE-6.8TeV-19_2m-VdM-2025_V1")
lhc.y2025.vdm.add_process("qchange","QCHANGE-6.8TeV-VdM-2025_V1")
lhc.y2025.vdm.add_process("desqueeze","DESQUEEZE-6.8TeV-VdM-IP8_50m-2025_V1")
lhc.y2025.vdm.add_process("physics","PHYSICS-6.8TeV-VdM-2025_V1")
lhc.y2025.vdm.set_data_from_lsa()
lhc.y2025.vdm.gen_repo_data()
#lhc.y2025.vdm.gen_eos_data()   done by gitlab actions

# 2025 - IONS
lhc.y2025.add_cycle(
    name="ions",
    label="PbPb Physics 2025",
    particles=["ions","ions"],
    charges=[82,82],
    masses=[193.68715,193.68715]
)

lhc.y2025.ions.add_process("ramp","RAMP-SQUEEZE-6.8TeV-IONS-1m-2025_V1")
lhc.y2025.ions.add_process("qchange","QCHANGE-6.8TeV-IONS-2025_V1")
lhc.y2025.ions.add_process("squeeze","SQUEEZE-6.8TeV-IONS-1m-50cm-1m-2025_V1")
lhc.y2025.ions.add_process("physics","PHYSICS-6.8TeV-IONS-2025_V1")
lhc.y2025.ions.set_data_from_lsa()
lhc.y2025.ions.gen_repo_data()
#lhc.y2025.ions.gen_eos_data() done by gitlab actions

# Website

lhc.y2025.gen_html_pages()

lhc.y2025.vdm.ramp.label=lhc.y2025.pp.ramp.label
lhc.y2025.vdm.desqueeze.label="Desqueeze to large beta*"
lhc.y2025.vdm.qchange.label=lhc.y2025.pp.qchange.label
lhc.y2025.vdm.physics.label=lhc.y2025.pp.physics.label
lhc.y2025.vdm.save_data(level=1)
lhc.y2025.vdm.refresh()


lhc.y2025.ions.ramp.label=lhc.y2025.pp.ramp.label
lhc.y2025.ions.qchange.label=lhc.y2025.pp.qchange.label
lhc.y2025.ions.physics.label=lhc.y2025.pp.physics.label
lhc.y2025.ions.levelling.label=lhc.y2025.pp.levelling.label
lhc.y2025.ions.save_data(level=1)
lhc.y2025.ions.refresh()



###    2024    ###

lhc.y2024.description="Optics models used in 2024"
lhc.y2024.save_data()


# 2024 - VDM
lhc.y2024.add_cycle(
    name="vdm",
    label="VDM 2024",
    particles=["proton","proton"],
    charges=[1,1]
)
lsa.findBeamProcesses(".*vdm.*2024.*")
lhc.y2024.vdm.refresh()
lhc.y2024.vdm.add_process("ramp","RAMP-DESQUEEZE-6.8TeV-19_2m-VdM-2024RP_V1")
lhc.y2024.vdm.add_process("qchange","QCHANGE-6.8TeV-2024RP-VdM_V1")
lhc.y2024.vdm.add_process("physics","PHYSICS-6.8TeV-2024RP-VdM_V1")
lhc.y2024.vdm.set_data_from_lsa()
lhc.y2024.vdm.gen_repo_data()
#lhc.y2024.vdm.gen_eos_data()  done by gitlab actions

# 2024 - Proton Physics
lhc.y2024.add_cycle(
    name="pp",
    label="Proton Physics 2024",
    particles=["proton","proton"],
    charges=[1,1]
)
lhc.y2024.pp.refresh()
lhc.y2024.pp.add_process("ramp","RAMP-SQUEEZE-6.8TeV-ATS-2m-2024_V1")
lhc.y2024.pp.add_process("squeeze","SQUEEZE-6.8TeV-2m-1.2m-LHCb-2024_V1")
lhc.y2024.pp.add_process("qchange","QCHANGE-6.8TeV-2024_V1")
lhc.y2024.pp.add_process("physics","PHYSICS-6.8TeV-1.2m-2024_V1")
lhc.y2024.pp.add_process("levelling","SQUEEZE-6.8TeV-1.2m-30cm-2024_V1")
lhc.y2024.pp.set_data_from_lsa()
lhc.y2024.pp.gen_repo_data()
#lhc.y2024.pp.gen_eos_data() done by gitlab actions

lhc.y2024.add_cycle(
    name="ions",
    label="PbPb Physics 2024",
    particles=["ions","ions"],
    charges=[82,82],
    masses=[193.68715,193.68715]
)

lhc.y2024.ions.add_process("ramp","RAMP-SQUEEZE-6.8TeV-IONS-1m-2024_V1")
lhc.y2024.ions.add_process("qchange","QCHANGE-6.8TeV-IONS-2024_V1")
lhc.y2024.ions.add_process("squeeze","SQUEEZE-6.8TeV-IONS-1m-50cm-2024_V1")
lhc.y2024.ions.add_process("physics","PHYSICS-6.8TeV-IONS-2024_V1")
lhc.y2024.ions.set_data_from_lsa()
lhc.y2024.ions.gen_repo_data()
#lhc.y2024.ions.gen_eos_data() done by gitlab actions

## cp ../../2025/scripts/generate_web_data.py .

# website
lhc.y2024.pp.ramp.label="Ramp to 6.8 TeV and squeeze to 2 m"
lhc.y2024.pp.squeeze.label="Squeeze to 1.2 m and LHCb to 1.0 m"
lhc.y2024.pp.qchange.label="Change to physics tune"
lhc.y2024.pp.physics.label="Proton Physics at 1.2 m"
lhc.y2024.pp.levelling.label="Squeeze to 30 cm"

lhc.y2024.vdm.ramp.label="Ramp to 6.8 TeV and desqueeze to 19 m"
lhc.y2024.vdm.qchange.label="Change to physics tune"
lhc.y2024.vdm.physics.label="VDM at 19 m"

lhc.y2024.ions.ramp.label="Ramp to 6.8 TeV and squeeze to 1 m"
lhc.y2024.ions.qchange.label="Change to physics tune"
lhc.y2024.ions.squeeze.label="Squeeze to 50 cm"
lhc.y2024.ions.physics.label="PbPb Physics at 1 m"



###  2023  ###

"""
cp ../2025/.gitlab-ci.yml .
cp ../2025/scripts/generate_web_data.py scripts/
cp ../2025/scripts/generate_eos_data.py scripts/
"""

lhc.y2023.label="Run 3 - 2023"
lhc.y2023.description="Optics models used in 2023"
lhc.y2023.start="2023-01-01 00:00:00"
lhc.y2023.end="2023-12-31 23:59:59"
lhc.y2023.save_data()

run=lhc.y2023.get_run_data()
# proton physics
run.fills[8645].main_processes()
lhc.y2023.add_cycle(
    name="pp",
    label="Proton Physics 2023",
    particles=["proton","proton"],
    charges=[1,1]
)
lhc.y2023.pp.add_process("ramp","RAMP-SQUEEZE-6.8TeV-ATS-2m-2023_V1","Ramp to 6.8 TeV and squeeze to 2 m")
lhc.y2023.pp.add_process("squeeze","SQUEEZE-6.8TeV-2m-1.2m-2023_V1","Squeeze to 1.2 m")
lhc.y2023.pp.add_process("rotation","LHCb-ROTATION-2023_V1","Rotate IP8 crossing plane")
lhc.y2023.pp.add_process("qchange","QCHANGE-6.8TeV-2023_V1","Change to physics tune")
lhc.y2023.pp.add_process("physics","PHYSICS-6.8TeV-1.2m-2023_V1","Proton Physics at 1.2 m")
lhc.y2023.pp.add_process("levelling","SQUEEZE-6.8TeV-1.2m-30cm-2023_V1","Squeeze to 30 cm")
lhc.y2023.pp.save_data(level=1)
lhc.y2023.pp.set_data_from_lsa()
lhc.y2023.pp.gen_repo_data()

# VDM
run.fills[9128].main_processes()
lhc.y2023.add_cycle(
    name="vdm",
    label="VDM Physics 2023",
    particles=["proton","proton"],
    charges=[1,1]
)
lhc.y2023.vdm.add_process("ramp","RAMP-DESQUEEZE-6.8TeV-19_2m-VdM-2022_V1","Ramp to 6.8 TeV and desqueeze to 19 m")
lhc.y2023.vdm.add_process("qchange","QCHANGE-6.8TeV-2022-VdM_V1","Change to physics tune")
lhc.y2023.vdm.add_process("physics","PHYSICS-6.8TeV-2022-VdM_V1","VDM at 19 m")
lhc.y2023.vdm.save_data(level=1)
lhc.y2023.vdm.set_data_from_lsa(start="2022-01-01 00:00:00")
lhc.y2023.vdm.gen_repo_data()

# Ions
run.fills[9180].main_processes()

lhc.y2023.add_cycle(
    name="ions",
    label="PbPb Physics 2023",
    particles=["ions","ions"],
    charges=[82,82],
    masses=[193.68715,193.68715]
)

lhc.y2023.ions.add_process("ramp","RAMP-SQUEEZE-6.8TeV-IONS-50cm-2023_V2_V1","Ramp to 6.8 TeV and squeeze to 50cm m")
lhc.y2023.ions.save_data(level=1)
lhc.y2023.ions.set_data_from_lsa()
lhc.y2023.ions.gen_repo_data()

from lhcoptics import LHCOptics
madx=lhc.y2023.pp.ramp.get_madx_model(idx=0)
opt=LHCOptics.from_madx(madx,make_model='xsuite')
opt.model.env.to_json("/home/rdemaria/local/acc-models-lhc/2023/xsuite/lhc.json")

## High Beta 2023 processes

run=lhc.y2023.get_run_data()
run.fills[9162].main_processes()

lhc.y2023.add_cycle(
    name="highbeta",
    label="High Beta Proton Physics 2023",
    particles=["proton","proton"],
    charges=[1,1],
    masses=[0.9382720813,0.9382720813],
)

lhc.y2023.highbeta.add_process(
    "ramp",
    "RAMP-DESQUEEZE-6.8TeV-75m-HB-2023_V1",
    "Ramp to 6.8 TeV and desqueeze to 75 m"
)

lhc.y2023.highbeta.add_process(
    "desqueeze1",
    "SQUEEZE-6.8TeV-75m-120m-HB-2023_V1",
    "Desqueeze to 120 m"
)

lhc.y2023.highbeta.add_process(
    "desqueeze2",
    "SQUEEZE-6.8TeV-120m-3km-HB-2023_V1",
    "Desqueeze to 3 km"
)
lhc.y2023.highbeta.add_process(
    "qchange",
    "QCHANGE-6.8TeV-HB-3km-2023_V1",
    "Change to physics tune"
)
lhc.y2023.highbeta.add_process(
    "physics",
    "PHYSICS-6.8TeV-HB-3km-2023_V1",
    "High Beta Physics at 3 km"
)

lhc.y2023.highbeta.save_data(level=1)
lhc.y2023.highbeta.set_data_from_lsa()
lhc.y2023.highbeta.gen_repo_data()

##### 2022 #####

lhc.y2022.label="Run 3 - 2022"
lhc.y2022.description="Optics models used in 2022"
lhc.y2022.start="2022-01-01 00:00:00"
lhc.y2022.end="2022-12-31 23:59:59"
lhc.y2022.save_data()

run=lhc.y2022.get_run_data()


## proton physics
run.fills[9101].main_processes()

"""
RAMP-SQUEEZE-6.8TeV-ATS-1.3m_V1'
SQUEEZE-6.8TeV-1.3m-60cm_V1'
QCHANGE-6.8TeV-2022_V1'
PHYSICS-6.8TeV-2022_V1']
"""

lhc.y2022.add_cycle(
    name="pp",
    label="Proton Physics 2022",
    particles=["proton","proton"],
    charges=[1,1],
    masses=[0.9382720813,0.9382720813],
)
lhc.y2022.pp.add_process(
    "ramp",
    "RAMP-SQUEEZE-6.8TeV-ATS-1.3m_V1",
    "Ramp to 6.8 TeV and squeeze to 1.3 m"
)
lhc.y2022.pp.add_process(
    "squeeze",
    "SQUEEZE-6.8TeV-1.3m-60cm_V1",
    "Squeeze to 60 cm"
)
lhc.y2022.pp.add_process(
    "qchange",
    "QCHANGE-6.8TeV-2022_V1",
    "Change to physics tune"
)
lhc.y2022.pp.add_process(
    "physics",
    "PHYSICS-6.8TeV-2022_V1",
    "Proton Physics at 60 cm"
)
lhc.y2022.pp.save_data(level=1)
lhc.y2022.pp.set_data_from_lsa()
lhc.y2022.pp.gen_repo_data()

# Vdm 2022

run.fills[9128].main_processes()

"""
'RAMP-DESQUEEZE-6.8TeV-19_2m-VdM-2022_V1',
 'QCHANGE-6.8TeV-2022-VdM_V1',
 'PHYSICS-6.8TeV-2022-VdM_V1',
"""

lhc.y2022.add_cycle(
    name="vdm",
    label="VDM Physics 2022",
    particles=["proton","proton"],
    charges=[1,1],
    masses=[0.9382720813,0.9382720813],
)
lhc.y2022.vdm.add_process(
    "ramp",
    "RAMP-DESQUEEZE-6.8TeV-19_2m-VdM-2022_V1",
    "Ramp to 6.8 TeV and desqueeze to 19 m"
)
lhc.y2022.vdm.add_process(
    "qchange",
    "QCHANGE-6.8TeV-2022-VdM_V1",
    "Change to physics tune"
)
lhc.y2022.vdm.add_process(
    "physics",
    "PHYSICS-6.8TeV-2022-VdM_V1",
    "VDM Physics at 19 m"
)
lhc.y2022.vdm.save_data(level=1)
lhc.y2022.vdm.set_data_from_lsa()
lhc.y2022.vdm.gen_repo_data()

# xsuite  model

from lhcoptics import LHCOptics
madx=lhc.y2022.pp.ramp.get_madx_model(idx=0)
opt=LHCOptics.from_madx(madx,make_model='xsuite')
opt.model.env.to_json("/home/rdemaria/local/acc-models-lhc/2022/xsuite/lhc.json")

### 2021 ###

"""
cp ../2025/.gitlab-ci.yml .
cp ../2025/scripts/generate_web_data.py scripts/
cp ../2025/scripts/generate_eos_data.py scripts/
"""

lhc.y2021.label="Run 3 - 2021"
lhc.y2021.description="Optics models used in 2021"
lhc.y2021.start="2021-01-01 00:00:00"
lhc.y2021.end="2021-12-31 23:59:59"
lhc.y2021.save_data()
run=lhc.y2021.get_run_data()



# 2018 - Proton Physics
"""
cp ../2025/.gitlab-ci.yml .
cp ../2025/scripts/generate_web_data.py scripts/
cp ../2025/scripts/generate_eos_data.py scripts/
"""


lhc.y2018.label="Run 2 - 2018"
lhc.y2018.description="Optics models used in 2018"
lhc.y2018.start="2018-01-01 00:00:00"
lhc.y2018.end="2018-12-31 23:59:59"
lhc.y2018.save_data()
run=lhc.y2018.get_run_data()

# proton physics


['RAMP_PELP-SQUEEZE-6.5TeV-ATS-1m-2018_V3_V1',
 'QCHANGE-6.5TeV-2018_V1',
 'DISCRETE_LHCRING_ADT_FLATTOP',
 'SQUEEZE-6.5TeV-ATS-1m-30cm-2018_V1',
 'PHYSICS-6.5TeV-30cm-120s-2018_V1']

lhc.y2018.add_cycle(
    name="pp",
    label="Proton Physics 2018",
    particles=["proton","proton"],
    charges=[1,1],
    masses=[0.9382720813,0.9382720813],
)
lhc.y2018.pp.add_process(
    "ramp",
    "RAMP_PELP-SQUEEZE-6.5TeV-ATS-1m-2018_V3_V1",
    "Ramp to 6.5 TeV and squeeze to 1 m"
)
lhc.y2018.pp.add_process(
    "qchange",
    "QCHANGE-6.5TeV-2018_V1",
    "Change to physics tune"
)
lhc.y2018.pp.add_process(
    "squeeze",
    "SQUEEZE-6.5TeV-ATS-1m-30cm-2018_V1",
    "Squeeze to 30 cm"
)
lhc.y2018.pp.add_process(
    "physics",
    "PHYSICS-6.5TeV-30cm-120s-2018_V1",
    "Proton Physics at 30 cm"
)
lhc.y2018.pp.save_data(level=1)
lhc.y2018.pp.set_data_from_lsa(part="value")
lhc.y2018.pp.gen_repo_data()

# VDM 2018
run.fills[6864].main_processes()


[
 'RAMP-DESQUEEZE-19m-2016_V1',
 'PHYSICS-6.5TeV-19-24m-2016_V1',
]

lhc.y2018.add_cycle(
    name="vdm",
    label="VDM Physics 2018",
    particles=["proton","proton"],
    charges=[1,1],
    masses=[0.9382720813,0.9382720813],
)
lhc.y2018.vdm.add_process(
    "ramp",
    "RAMP-DESQUEEZE-19m-2016_V1",
    "Ramp to 6.5 TeV and desqueeze to 19 m"
)
lhc.y2018.vdm.add_process(
    "physics",
    "PHYSICS-6.5TeV-19-24m-2016_V1",
    "VDM Physics at 19 m"
)
lhc.y2018.vdm.save_data(level=1)
lhc.y2018.vdm.set_data_from_lsa(
    start="2016-01-03 17:12:51",
    part="value")
lhc.y2018.vdm.gen_repo_data()

# High Beta 2018

run.fills[6877].main_processes()
[
 'RAMP-DESQUEEZE-6.5TeV-67m-HighB-2018_V1',
 'QCHANGE-6.5TeV-2018-HighBeta-90m_V2',
 'SQUEEZE-6.5TeV-HighB-67m-75m-2018_V1',
 'SQUEEZE-6.5TeV-highB-75m-90m-2018_V1',
 'PHYSICS-6.5TeV-90m-HighB-120s-2018_V1',
]

lhc.y2018.add_cycle(
    name="pp_highbeta",
    label="High Beta Proton Physics 2018",
    particles=["proton","proton"],
    charges=[1,1]
)

lhc.y2018.pp_highbeta.masses=[0.9382720813,0.9382720813]
lhc.y2018.pp_highbeta.save_data(level=1)

lhc.y2018.pp_highbeta.add_process(
    "ramp",
    "RAMP-DESQUEEZE-6.5TeV-67m-HighB-2018_V1",
    "Ramp to 6.5 TeV and desqueeze to 67 m"
)
lhc.y2018.pp_highbeta.add_process(
    "qchange",
    "QCHANGE-6.5TeV-2018-HighBeta-90m_V2",
    "Change to physics tune"
)
lhc.y2018.pp_highbeta.add_process(
    "desqueeze1",
    "SQUEEZE-6.5TeV-HighB-67m-75m-2018_V1",
    "Desqueeze to 75 m"
)
lhc.y2018.pp_highbeta.add_process(
    "desqueeze2",
    "SQUEEZE-6.5TeV-highB-75m-90m-2018_V1",
    "Desqueeze to 90 m"
)
lhc.y2018.pp_highbeta.add_process(
    "physics",
    "PHYSICS-6.5TeV-90m-HighB-120s-2018_V1",
    "High Beta Physics at 90 m"
)
lhc.y2018.pp_highbeta.save_data(level=1)
lhc.y2018.pp_highbeta.set_data_from_lsa(part="value")
lhc.y2018.pp_highbeta.gen_repo_data()


# ions 2018
run.fills[7444].main_processes()

[
 'RAMP-SQUEEZE-6.37TeV-ATS-Ion-2018_V2',
 'QCHANGE-6.37TeV-Ion-2018_V1',
 'SQUEEZE-6.37TeV-ATS-1m-50cm-2018_ION_V1',
 'PHYSICS-6.37TeV-50cm-240s-Ion-2018_V1']

lhc.y2018.add_cycle(
    name="ions",
    label="PbPb Physics 2018",
    particles=["ions","ions"],
    charges=[82,82],
    masses=[193.68715,193.68715]
)
lhc.y2018.ions.add_process(
    "ramp",
    "RAMP-SQUEEZE-6.37TeV-ATS-Ion-2018_V2",
    "Ramp to 6.37 TeV and squeeze to 1 m"
)
lhc.y2018.ions.add_process(
    "qchange",
    "QCHANGE-6.37TeV-Ion-2018_V1",
    "Change to physics tune"
)
lhc.y2018.ions.add_process(
    "squeeze",
    "SQUEEZE-6.37TeV-ATS-1m-50cm-2018_ION_V1",
    "Squeeze to 50 cm"
)
lhc.y2018.ions.add_process(
    "physics",
    "PHYSICS-6.37TeV-50cm-240s-Ion-2018_V1",
    "PbPb Physics at 50 cm"
)
lhc.y2018.ions.save_data(level=1)
lhc.y2018.ions.set_data_from_lsa(part="value")
lhc.y2018.ions.gen_repo_data()


# xsuite  model
from lhcoptics import LHCOptics
madx=lhc.y2018.pp.ramp.get_madx_model(idx=0)
opt=LHCOptics.from_madx(madx,make_model='xsuite')
opt.model.env.to_json("/home/rdemaria/local/acc-models-lhc/2018/xsuite/lhc.json")



#HL-LHC 1.6 Optics sets

lhc.hl16.add_set(
    name="round",
    label="End of levelling round optics",
)

lhc.hl16.round.add_optics(
    name="hv150_l1500",
    label="Round optics 15cm in IP1 and IP5 and 1.5m in IP8",
    optics="strengths/round/opt_round_150_1500_optphases.madx",
    settings={
        "on_x1": 250, "on_x5":250,
        "on_x2": 200, "on_x8": 200,
        "phi_ir1": 0, "phi_ir5": 90,
        "phi_ir2": 90, "phi_ir8": 90,
    }
)
