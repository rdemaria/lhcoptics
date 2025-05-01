from lhcoptics import LHC, get_lsa
lsa=get_lsa()
lsa.findBeamProcesses(".*vdm.*2025*")


lhc=LHC()
lhc.y2025.vdm.refresh()
lhc.y2025.vdm.add_process("ramp","RAMP-DESQUEEZE-6.8TeV-19_2m-VdM-2025_V1")
lhc.y2025.vdm.add_process("qchange","QCHANGE-6.8TeV-VdM-2025_V1")
lhc.y2025.vdm.add_process("desqueeze","DESQUEEZE-6.8TeV-VdM-IP8_50m-2025_V1")
lhc.y2025.vdm.add_process("collapse","PHYSICS-6.8TeV-VdM-2025_V1")
lhc.y2025.vdm.gen_data_from_lsa()

lhc.y2024.add_cycle(
    name="vdm",
    label="VDM 2024",
    particles=["proton","proton"],
    charges=[1,1]
)
lsa.findBeamProcesses(".*vdm.*2024*")
lhc.y2024.vdm.refresh()
lhc.y2024.vdm.add_process("ramp","RAMP-DESQUEEZE-6.8TeV-19_2m-VdM-2022_V1")
lhc.y2024.vdm.add_process("qchange","QCHANGE-6.8TeV-2024RP-VdM_V1")
lhc.y2024.vdm.add_process("collapse","PHYSICS-6.8TeV-2024RP-VdM_V1")
lhc.y2024.vdm.gen_data_from_lsa()

lhc.y2024.add_cycle(
    name="pp",
    label="Proton Physics 2024",
    particles=["proton","proton"],
    charges=[1,1]
)
lhc.y2024.pp.refresh()
lhc.y2024.pp.add_process("ramp","RAMP-SQUEEZE-6.8TeV-ATS-2m-2024_V1")
lhc.y2024.pp.add_process("squeeze","SQUEEZE-6.8TeV-2m-1.2m-LHCb-2024_V1")
lhc.y2024.pp.add_process("collapse","PHYSICS-6.8TeV-1.2m-2024_V1")
lhc.y2024.pp.add_process("levelling","SQUEEZE-6.8TeV-1.2m-30cm-2024_V1")
lhc.y2024.pp.gen_data_from_lsa()