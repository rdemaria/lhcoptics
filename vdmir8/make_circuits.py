from lhcoptics import LHCCircuits

circuits = LHCCircuits.from_lsa().to_json("data/lhccircuits.json")
