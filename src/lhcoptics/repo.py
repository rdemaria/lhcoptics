from pathlib import Path
from .optics import LHCOptics

from yaml import load, dump

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from .lsa_util import get_lsa


def read_yaml(filename):
    with open(filename) as f:
        return load(f, Loader=Loader)





class LHC:
    """LHC optics repository"""

    def __init__(self, name, basedir=None, cycles=None):
        name = str(name)
        self.basedir = self.get_basedir(name) if basedir is None else basedir
        self.name = name
        self.cycles = self.get_cycles() if cycles is None else cycles
        self.knob_table = {}

    def __repr__(self):
        return f"<LHC {self.name!r}>"

    def __getattr__(self, name):
        try:
            return self.cycles[name]
        except KeyError:
            raise KeyError(f"{name!r} not found in {self}.")

    def get_basedir(self, name):
        import subprocess

        accmodels = Path("acc-models-lhc")
        if accmodels.exists():
            if not (accmodels / "lhc.seq").exists():
                raise FileNotFoundError("acc-models-lhc/lhc.seq not found")
            # else:
            #    if (accmodels / ".git").exists():
            #        subprocess.run(["git", "switch", version], cwd=str(accmodels))
        elif (
            lcl := (Path.home() / "local" / "acc-models-lhc" / name)
        ).exists():
            accmodels.symlink_to(lcl)
        else:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "https://gitlab.cern.ch/acc-models/lhc.git",
                    "acc-models-lhc",
                ]
            )

        return accmodels

    def get_cycles(self):
        cycle_list= read_yaml(self.basedir / "scenarios/cycle/list.yaml")
        return {
            name: LHCCycle(name, basedir=self.basedir)
            for name in cycle_list
        }


class LHCCycle:
    def __init__(self, name, processes=None, basedir=None):
        self.name = name
        self.basedir = basedir
        self.cycledir = basedir / name
        self.processes = (
            self.get_processes() if processes is None else processes
        )

    def __repr__(self):
        return f"<Cycle {self.name!r}>"

    def __getattr__(self, name):
        return self.processes[name]

    def get_processes(self):
        dct = {}
        for bp in read_yaml(self.cycledir / f"beam_processes.yaml"):
            ((name, beamprocess),) = bp.items()
            dct[name] = LHCProcess(name, beamprocess,basedir=self.basedir)
        return dct

    def get_fills(self, lhcrun):
        bp_to_match = [bp.name for bp in self.beam_processes]

        def match(fill):
            fillbp = set([bp.split("@")[0] for ts, bp in fill.beam_processes])
            return all([bp in fillbp for bp in bp_to_match])

        return sorted([ff.filln for ff in lhcrun.fills.values() if match(ff)])

class LHCProcess:
    def __init__(self, name, beamprocess, basedir=None, optics=None):
        self.name = name
        self.basedir = basedir
        self.beamprocess = beamprocess
        self.beamprocessdir = self.basedir / "beam_processes" / self.beamprocess
        self.optics_table = (
            self.get_optics_table() if optics is None else optics
        )

    def __repr__(self):
        return f"<Process {self.name!r}:{self.beamprocess}>"

    def set_optics_from_lsa(self):
        lsa=get_lsa()
        tbl=lsa.getOpticTable(self.beamprocess)
        self.optics_table={tt.time: tt.name for tt in tbl}
        return self

    def get_optics_table(self):
        return read_yaml(self.beamprocessdir / "optics.yaml")

    def save_to_repo(self):
        self.beamprocessdir.mkdir(parents=True, exist_ok=True)
        with open(self.beamprocessdir / "optics.yaml", "w") as f:
            dump(self.optics_table, f, Dumper=Dumper)

    def get_settings(self, params, lhcrun=None):
        import jpype

        if lhcrun is None:
            t1 = None
            t2 = None
        else:
            t1 = lhcrun.t1
            t2 = lhcrun.t2

        out = []
        for param in params:
            try:
                print(f"getting {param}")
                trims = lsa.getTrims(param, beamprocess=self.name, start=t1, end=t2)[
                    param
                ]
                for ts, trim in zip(*trims):
                    out.append([ts, param, trim])
            except jpype.JException as ex:
                print("Error extracting parameter '%s': %s" % (param, ex))
            except KeyError as ex:
                print("Empty response for '%s': %s" % (param, ex))
        out.sort()
        return out




class LHCKnobs:
    @classmethod
    def from_file(cls, fn):
        lst = []
        for ln in open(fn):
            try:
                lst.append(LHCKnob(*ln.strip().split(", ")))
            except:
                pass
        return cls(lst)

    def __init__(self, knob_list):
        self.mad = {}
        self.lsa = {}
        for knob in knob_list:
            self.mad[knob.mad] = knob
            self.lsa[knob.lsa] = knob

    def mad_value(self, lsa_name, lsa_value):
        return self.lsa[lsa_name].mad_value(lsa_value)

    def lsa_value(self, mad_name, mad_value):
        return self.mad[mad_name].lsa_value(mad_value)

    def to_mad(self, lsa_name, lsa_value):
        return self.lsa[lsa_name].to_mad(lsa_value)

    def add(self, knob):
        self.mad[knob.mad] = knob
        self.lsa[knob.lsa] = knob

    def remove(self, lsa=None, mad=None):
        if lsa is not None:
            knob = self.lsa[lsa]
        elif mad is not None:
            knob = self.lsa[lsa]
        del self.mad[knob.mad]
        del self.lsa[knob.lsa]

    def add_knob(self, mad, lsa, scaling, test):
        self.add(LHCKnob(mad, lsa, scaling, test))

    def get_settings(self):
        return list(self.lsa.keys())



class LHCKnobDef:
    def __init__(self, mad, lsa, scaling, test):
        self.mad = mad
        self.lsa = lsa
        self.scaling = float(scaling)
        self.test = float(test)

    def mad_value(self, lsa_value):
        return lsa_value * self.scaling

    def lsa_value(self, mad_value):
        return mad_value / self.scaling

    def to_mad(self, lsa_value):
        return f"{self.mad}={self.mad_value(lsa_value)};"

class LHCRun:
    def __init__(self, year):
        self.year = year
        self.t1 = f"{year}-01-01 00:00:00"
        self.t2 = f"{year}-12-31 23:59:59"
        # self.set_fills()
        self.cycle = {}

    def read_cycles(self, cycle_path=Path(".")):
        for cycle_name in open(cycle_path / "cycles.txt"):
            cycle_name = cycle_name.strip()
            self.cycle[cycle_name] = LHCCycle.from_dir(
                cycle_name, cycle_path / cycle_name
            )

    def save_models(self, knobs, cycle_path=Path(".")):
        for cycle_name, cycle in self.cycle.items():
            print(f"Saving {cycle_name}")
            cycle.save_models(knobs, cycle_path / cycle_name)

    def set_fills(self):
        self.fills = {}
        fills = lsa.findBeamProcessHistory(self.t1, self.t2, accelerator="lhc")
        for filln, bp_list in fills.items():
            # beam_processes=[(ts,bp.split('@')[0]) for ts,bp in bp_list]
            beam_processes = [(ts, bp) for ts, bp in bp_list]
            self.fills[filln] = LHCFill(filln, beam_processes)

    def find_beam_processes(self, regexp="", full=True):
        reg = re.compile(regexp)
        out = {}
        for filln, fill in self.fills.items():
            for ts, bp in fill.beam_processes:
                res = reg.match(bp)
                if res:
                    if full and "@" not in bp:
                        out.setdefault(bp, []).append(filln)
        return out

    def hist_beam_processes(self, regexp="", full=True):
        lst = self.find_beam_processes(regexp, full=full)
        return list(sorted((len(v), k) for k, v in lst.items()))

    def get_used_beamprocess(self):
        out = set()
        for fill in self.fills.values():
            out.update(fill.get_used_beamprocess())
        return out

    def __repr__(self):
        return f"LHCRun({self.year})"



