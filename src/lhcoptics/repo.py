from pathlib import Path
import re
import os
import time


from .lsa_util import get_lsa
from .utils import (
    gitlab_get_branches_and_tags,
    file_one_day_old,
    write_yaml,
    read_yaml,
    git_get_current_commit,
    git_pull,
    git_clone_repo,
)


default_basedir = os.getenv(
    "LHCOPTICS_BASEDIR", default=Path.home() / "local" / "acc-models-lhc"
)

projectt_id = 76507
git_url = os.getenv(
    "LHCOPTICS_GIT_URL", "https://gitlab.cern.ch/acc-models/acc-models-lhc.git"
)


def check_repobasedir(basedir):
    if basedir is None:
        basedir = default_basedir
    if not Path(basedir):
        resp = input(
            f"""LHC optics base directory does not exists in '{basedir}'.
               Shall I Create it? [y/n]"""
        )
        if resp.lower() == "y":
            basedir.mkdir(parents=True)
        else:
            raise ValueError(f"Repository {basedir} does not exists.")
        raise ValueError("create ")
    return basedir


class LHC:
    """LHC optics repository

    The structure is as follows:
    branch_or_tag.cycle_or_collection.process_or_stage -> optics model


    """

    def __init__(self, basedir=None):
        self.basedir = check_repobasedir(basedir)
        self.branches, self.tags = self.get_branches_and_tags()
        self.check_local_branches()

    def __repr__(self):
        return f"<LHC Repo at '{self.basedir}'>"

    def __dir__(self):
        othernames = []
        for name in self.branches:
            if name[0] in "0123456789":
                othernames.append("y" + name)
            else:
                othernames.append(name)
        return super().__dir__() + othernames

    def __getattr__(self, name):
        if name.startswith("y"):
            name = name[1:]
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"No branch or tag named {name}")

    def _ipython_key_completions_(self):
        return list(self.branches.keys()) + list(self.tags.keys())

    def __getitem__(self, name):
        if name in self.branches:
            return LHCRepo(name, self.basedir/name)
        elif name in self.tags:
            return LHCRepo(name, self.basedir/name)
        else:
            raise KeyError(f"No branch or tag named {name}")

    def check_local_branches(self, dry_run=False):
        """
        Check if the local branches are up to date with the remote branches
        """
        for branch, commit in self.branches.items():
            branch_dir = self.basedir / branch
            if branch_dir.exists():
                dir_commit = git_get_current_commit(branch_dir)
                if dir_commit != commit:
                    print(f"Branch {branch} is not up to date")
                    print(f"Local commit: {dir_commit}")
                    print(f"Remote commit: {commit}")
                    if not dry_run:
                        print(git_pull(branch_dir))

    def get_branches_and_tags(self, method="auto"):
        """
        Get the list of branches in the repository

        method: str
            "auto": use gitlab API to get the list of branches if the local list is one day old or use
            "local": use the local list of branches
            "update": force update the local list of branches
        """
        cache_file = self.basedir / "cache.yaml"
        cache_file_good = cache_file.exists() and file_one_day_old(cache_file)
        if (method == "auto" and cache_file_good) or (method == "local"):
            cache = read_yaml(cache_file)
        elif (method == "update") or (
            method == "auto" and not cache_file_good
        ):
            print("Getting branches and tags from gitlab")
            cache = gitlab_get_branches_and_tags(
                project_id=76507,
                gitlab_url="https://gitlab.cern.ch",
                timeout=0.5,
            )
            if cache is None:
                raise ValueError("Error getting branches and tags from gitlab")
            write_yaml(cache, cache_file)
        else:
            raise ValueError("Error getting branches and tags")

        branches = cache["branches"]
        tags = cache["tags"]

        for name in list(branches):
            if len(name) != 4 or name[0] not in "2h":
                del branches[name]

        self.branches = branches
        self.tags = tags

        return branches, tags


class LHCRepo:
    """LHC optics for a specific branch or tag"""

    def __init__(
        self,
        name,
        basedir=None,
        collections=None,
    ):
        name = str(name)
        self.name = name
        self.basedir = Path(basedir)
        if not self.basedir.exists():
            print(f"Cloning {self.basedir}")
            git_clone_repo( git_url, self.basedir, name)

        # self.set_beam_processes()
        # self.set_cycles()
        # self.collections = collections if collections is not None else {}
        # self.knobs = LHCKnobDefs.from_file(
        #    self.basedir / "operation" / "knobs.txt"
        # )

    def __repr__(self):
        return f"<LHC Repo '{self.basedir}'>"

    def __getattr__(self, name):
        if name in self.cycle:
            return self.cycle[name]
        elif name in self.beam_processes:
            return self.beam_processes[name]
        elif name in self.collections:
            return self.collections[name]
        try:
            return self.cycle[name]
        except KeyError:
            raise AttributeError(f"{name!r} not found in {self}.")

    def set_cycles(self):
        listfile = self.basedir / "scenarios/cycle/list.yaml"
        if listfile.exists():
            cycle_list = read_yaml(listfile)
            cycle = {
                name: LHCCycle(name, basedir=self.basedir)
                for name in cycle_list
            }
        else:
            cycle = {}
            cycle_list = []
        self.cycle = cycle
        self.cycle_list = cycle_list

    def new_cycle(self, name):
        cycle = LHCCycle(name, self)
        self.insert_cycle(cycle)
        return cycle

    def insert_cycle(self, cycle, idx=None):
        self.cycle[cycle.name] = cycle
        cycle.parent = self
        if idx is not None:
            self.cycle_list.insert(idx, cycle.name)
        else:
            self.cycle_list.append(cycle.name)

    def set_beam_processes(self):
        bpfile = self.basedir / "scenarios/bp/list.yaml"
        if bpfile.exists():
            beam_processes = {
                name: LHCProcess(name, bp, basedir=self.basedir)
                for name, bp in read_yaml(bpfile)
            }
        else:
            beam_processes = {}
        self.beam_processes = beam_processes


class LHCCycle:
    def __init__(
        self, name, parent, processes=None, label=None, description=None
    ):
        self.name = name
        self.parent = parent
        self.label = label
        self.description = description
        self.cycledir = self.basedir / "scenarios" / "cycle" / self.name
        self.descfile = self.cycledir / "desc.yaml"
        self.read_data(self)

    def read_data(self):
        if self.descfile.exists():
            data = read_yaml(self.descfile)
            self.label = data.get("label", self.label)
            self.description = data.get("description", self.description)
            self.processes = {}
            for bp in data.get("beam_processes", []):
                ((name, beamprocess),) = bp.items()
                self.processes[name] = self.parent.bp[beamprocess]
            self._set_processes(data.get("processes", None))

    def save_to_repo(self):
        self.cycledir.mkdir(parents=True, exist_ok=True)
        with open(self.cycledir / "beam_processes.yaml", "w") as f:
            dump(self.processes, f, Dumper=Dumper)

    def __repr__(self):
        return f"<Cycle {self.name!r}>"

    def __getattr__(self, name):
        return self.processes[name]

    def _set_processes(self, processes):
        self.process_list = []
        self.process = {}
        if processes is None:
            self.processes = {}
        bpfile = (
            self.basedir
            / "scenarios/cycle"
            / self.name
            / "beam_processes.yaml"
        )
        for bp in read_yaml(self.cycledir / "beam_processes.yaml"):
            ((name, beamprocess),) = bp.items()
            dct[name] = LHCProcess(
                name=name, beamprocess=beamprocess, parent=self
            )
        self.processes = dct

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
        self.beamprocessdir = self.basedir / "bp" / self.beamprocess
        self.optics_table = (
            self.get_optics_table() if optics is None else optics
        )

    def __repr__(self):
        return f"<Process {self.name!r}:{self.beamprocess}>"

    def set_optics_from_lsa(self):
        get_lsa().lsa.getOpticTable(self.beamprocess)
        self.optics_table = {tt.time: tt.name for tt in tbl}
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
                trims = get_lsa().getTrims(
                    param, beamprocess=self.name, start=t1, end=t2
                )[param]
                for ts, trim in zip(*trims):
                    out.append([ts, param, trim])
            except jpype.JException as ex:
                print("Error extracting parameter '%s': %s" % (param, ex))
            except KeyError as ex:
                print("Empty response for '%s': %s" % (param, ex))
        out.sort()
        return out


class LHCKnobDefs:
    @classmethod
    def from_file(cls, fn):
        lst = []
        for ln in open(fn):
            try:
                lst.append(LHCKnobDef(*ln.strip().split(", ")))
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

    def __repr__(self):
        return f"<LHCKnobDefs {len(self.lsa)} knobs>"


class LHCKnobDef:
    def __init__(self, mad, lsa, scaling, test):
        self.mad = str(mad)
        self.lsa = str(lsa)
        self.scaling = float(scaling)
        self.test = float(test)

    def mad_value(self, lsa_value):
        return lsa_value * self.scaling

    def lsa_value(self, mad_value):
        return mad_value / self.scaling

    def to_mad(self, lsa_value):
        return f"{self.mad}={self.mad_value(lsa_value)};"

    def __repr__(self):
        return f"<Knob {self.lsa}:{self.mad}>"


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
