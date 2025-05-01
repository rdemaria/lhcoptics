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
