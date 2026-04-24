# Issues

Fixes:
- add energy dependent knob as Knob instances
- add i_mo, on_mo knobs
- add experimental knobs
- investigate strange bug, dqx.b12 knobs are matched but
  needs update_model to have a proper model for beam2
- aperture for collision, add phase dep logic in limits
- make transition examples
- check 50cm strengths margins
- match injection transition
- remove twiss defaults, enforce correct options
- add dp_trim knobs
- add q'' knobs
- check lumiscan knobs for 2 and 8 I am using different strengths
- replace LHC with LHCDev and update gitlab actions
- check web site generation
- Make sure to use model.create_knob (that deletes and recreate the knob) before matching in all knobs

Features:
- add matching for MBs (in particular for flat)
- add per-arc dispersion knobs
- introduce multi i/k power converter for triplets
- match IR1/IR5 ats phase
- implement measured transfer functions for HL-LHC triplets
- implement fringe fields in the triplets for beta-beating
- redefine the voltages of H crab beam2 in the sequence
- add rotation crabbing plane
- update logic coupling knobs in case one of ir1 or ir5 is not ats
- verify disp knobs closeness with coupling
- improve max_chrom target in global_w chrom for flat optics
- test matching chrom and w at the same time
