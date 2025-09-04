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


# 2018 - Proton Physics
lhc.y2018.add_cycle(
    name="pp_highbeta",
    label="High Beta Proton Physics 2018",
    particles=["proton","proton"],
    charges=[1,1]
)
lhc.y2018.pp_highbeta.refresh()
lhc.y2018.pp_highbeta.add_process("physics","PHYSICS-6.5TeV-90m-HighB-120s-2018_V1")
lhc.y2018.pp_highbeta.gen_data_from_lsa()
sett=lhc.y2018.pp_highbeta.physics.get_settings_from_lsa(part="value")
lhc.y2018.pp_highbeta.physics.set_settings_from_lsa(sett)
lhc.y2018.pp_highbeta.physics.gen_optics_dir()
lhc.y2018.pp_highbeta.gen_eos_data()



# Website

repo=lhc.y2025
repo.gen_html_pages()

repo.vdm.ramp.label=repo.pp.ramp.label
repo.vdm.desqueeze.label="Desqueeze to large beta*"
repo.vdm.qchange.label=repo.pp.qchange.label
repo.vdm.physics.label=repo.pp.physics.label
repo.vdm.save_data(level=1)
repo.vdm.refresh()


repo.ions.ramp.label=repo.pp.ramp.label
repo.ions.qchange.label=repo.pp.qchange.label
repo.ions.physics.label=repo.pp.physics.label
repo.ions.levelling.label=repo.pp.levelling.label
repo.ions.save_data(level=1)
repo.ions.refresh()
