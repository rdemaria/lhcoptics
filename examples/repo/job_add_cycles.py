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
lhc.y2025.pp.gen_data_from_lsa()
lhc.y2025.pp.gen_eos_data()


# 2025 - VDM
lsa.findBeamProcesses(".*vdm.*2025.*")
lhc.y2025.vdm.refresh()
lhc.y2025.vdm.add_process("ramp","RAMP-DESQUEEZE-6.8TeV-19_2m-VdM-2025_V1")
lhc.y2025.vdm.add_process("qchange","QCHANGE-6.8TeV-VdM-2025_V1")
lhc.y2025.vdm.add_process("desqueeze","DESQUEEZE-6.8TeV-VdM-IP8_50m-2025_V1")
lhc.y2025.vdm.add_process("physics","PHYSICS-6.8TeV-VdM-2025_V1")
lhc.y2025.vdm.gen_data_from_lsa()
lhc.y2025.vdm.gen_eos_data()

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
lhc.y2025.ions.gen_data_from_lsa()
lhc.y2025.ions.gen_eos_data()

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
lhc.y2024.vdm.gen_data_from_lsa()
lhc.y2024.vdm.gen_eos_data()

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
lhc.y2024.pp.add_process("physics","PHYSICS-6.8TeV-1.2m-2024_V1")
lhc.y2024.pp.add_process("levelling","SQUEEZE-6.8TeV-1.2m-30cm-2024_V1")
lhc.y2024.pp.gen_data_from_lsa()
lhc.y2024.pp.gen_eos_data()

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
lhc.y2024.ions.gen_data_from_lsa()
lhc.y2024.ions.gen_eos_data()

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