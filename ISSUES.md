# Issues

Fixes:
- symplify opt creation from simple madx files (e.g.no knob structure)
- restructure strengths manangement (use metadata list) and insert dipoles as well (in view of knobs using D1/D2)
- lhc.vars.load() does not create the knob structure with name expected by opt.update()
  a workaround is needed. LHCOptics.from_xsuite does the correct thing, but it is not atomic
  very dangerous...

- opt.get_params(full=True) still give atsfactor=1 and betxip1 not correct
- investigate strange bug, dqx.b12 knobs are matched but
  needs update_model to have a proper model for beam2
- aperture for collision, add phase dep logic in limit
- replace optics in v0 and v3 for MAD-X compatibility
- make transition examples
- check 50cm strengths margins
- add dp_trim knobs
- match injection transition
- remove twiss defaults, enforce correct options
- add detuning knobs
- add q'' knobs
- check lumiscan knobs for 2 and 8 I am using different strengths
- check web site generation

Features:
- add matching for MBs (in particular for flat)
- add per-arc dispersion knobs
- Make sure to use model.create_knob (that deletes and recreate the knob) before matching in all knobs
- replace LHC with LHCDev and update gitlab actions
- introduce multi i/k power converter for triplets
- match IR1/IR5 ats phase
- used named particles in json for correct energy
- implement measured transfer functions for HL-LHC triplets
- implement fringe fields in the triplets for beta-beating
- redefine the voltages of H crab beam2 in the sequence
- add rotation crabbing plane
- remove arc skew quads from irs
- update logic coupling knobs in case one of ir1 or ir5 is not ats
- add verbose option in xdeps to silence matching
- verify disp knobs closeness with coupling
- improve max_chrom target in global_w chrom for flat optics
- test matching chrom and w at the same time
- move kqs from ir strengths
- add energy dependent knobs (e.g. i_mo)
