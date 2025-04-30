"""
Repository structure

The structure

repository/branch_or_tag/cycle_or_collection/process_or_stage -> optics model

optics model:
- params
- strengths
- knobs
- sections
    - params
    - strengths
    - knobs
"""

from pathlib import Path
import re
import os
import site
import numpy as np


from .lsa_util import get_lsa
from .utils import (
    gitlab_get_branches_and_tags,
    file_one_day_old,
    write_yaml,
    read_yaml,
    git_get_current_commit,
    git_pull,
    git_clone_repo,
    unixtime_to_string,
    string_to_unixtime,
)


default_basedir = os.getenv(
    "LHCOPTICS_BASEDIR", default=Path(site.getuserbase()) / "acc-models-lhc"
)
default_basedir = Path(default_basedir)


# gitlab projectt_id for API
projectt_id = 76507
git_url = os.getenv(
    "LHCOPTICS_GIT_URL", "https://gitlab.cern.ch/acc-models/acc-models-lhc.git"
)


def check_repobasedir(basedir):
    if basedir is None:
        basedir = default_basedir
    if not Path(basedir).exists():
        resp = input(
            f"""LHC optics base directory does not exists in '{basedir}'.
               Shall I Create it? [y/n]"""
        )
        if resp.lower() == "y":
            basedir.mkdir(parents=True)
        else:
            raise ValueError(f"Repository {basedir} does not exists.")
        raise ValueError(f"Error creating {basedir}")
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
        self._repo_cache={}

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
        if name in self._repo_cache:
            return self._repo_cache[name]
        if name in self.branches or name in self.tags:
           repo_dir = self.basedir / name
           if not repo_dir.exists():
              print(f"Repository {repo_dir} does not exist")
              self.clone_repo(name)
           self._repo_cache[name] = LHCRepo(name, repo_dir)
           return self._repo_cache[name]
        else:
            raise KeyError(f"No branch or tag named {name}")

    def force_update(self):
        self._repo_cache.clear()
        cache_file = self.basedir / "cache.yaml"
        cache_file.unlink(missing_ok=True)
        self.branches, self.tags = self.get_branches_and_tags(method="update")
        self.check_local_branches(dry_run=True)

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
                    self.branches[branch] = commit
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

    def clone_repo(self, branch_or_tag):
        """
        Clone the repository
        """
        repo_dir = self.basedir / branch_or_tag
        if repo_dir.exists():
            print(f"Repository {repo_dir} already exists")
            return
        print(f"Cloning repository {repo_dir}")
        git_clone_repo(git_url, repo_dir, branch_or_tag)




class LHCRepo:
    """LHC optics for a specific branch or tag"""

    def __init__(
        self,
        name,
        basedir,
    ):
        self.name = str(name)
        self.basedir = Path(basedir)
        self.yaml = self.basedir / "scenarios" / "readme.yaml"
        if not self.yaml.exists():
            print(f"Metadata {self.yaml} does not exist")
        else:
           self.data = read_yaml(self.basedir / "scenarios" / "readme.yaml")
           self.knobs = self.get_knobs()
           self.cycles = self.get_cycles()
           self.start = self.data.get("start", None)
           self.end = self.data.get("end", None)

    def __repr__(self):
        return f"<LHC {self.name} '{self.basedir}'>"

    def get_knobs(self):
        fn = self.basedir / "operation" / "knobs.txt"
        if fn.exists():
            knobs = LHCKnobDefs.from_file(fn)
        else:
            knobs = LHCKnobDefs([])
        knobs.add_knob("nrj", "LHCBEAM/MOMENTUM", 1.0, 1)
        knobs.add_knob("vrf400", "RFBEAM1/TOTAL_VOLTAGE", 1.0, 1)
        return knobs

    def get_cycles(self):
        cycles = {}
        for cycle_name in self.data.get("cycles", []):
            cycles[cycle_name] = LHCCycle.from_dir(
                name=cycle_name,
                basedir=self.basedir / "scenarios" / "cycle" / cycle_name,
                parent=self,
            )
        return cycles

    def __getattr__(self, name):
        if name in self.cycles:
            return self.cycles[name]
        else:
            raise AttributeError(f"No cycle named {name}")

    def __dir__(self):
        return super().__dir__() + list(self.cycles.keys())


class LHCCycle:
    @classmethod
    def from_dir(cls, name, basedir, parent=None):
        basedir = Path(basedir)
        if not basedir.exists():
            print(f"Cycle directory {basedir} does not exist")
        return cls(name, basedir=basedir, parent=parent)

    def __init__(self, name, basedir, parent):
        self.name = name
        self.basedir = Path(basedir)
        self.parent = parent
        self.yaml = self.basedir / "readme.yaml"
        self.data = self.read_data(self.yaml)
        self.beam_processes = {}
        for process in self.data.get("beam_processes", []):
            self.beam_processes[process] = LHCProcess(
                name=process,
                basedir=self.basedir / process,
                parent=self,
            )
        self.start = self.data.get("start")
        self.end = self.data.get("end")

    def __getattr__(self, name):
        if name in self.beam_processes:
            return self.beam_processes[name]
        else:
            raise AttributeError(f"No process named {name}")

    def __dir__(self):
        return super().__dir__() + list(self.beam_processes.keys())

    def read_data(self, yaml):
        if not yaml.exists():
            print(f"Cycle data file {yaml} does not exist")
            return {}
        return read_yaml(yaml)

    def __repr__(self):
        return f"<Cycle {self.name!r}>"

    def get_fills(self, lhcrun):
        bp_to_match = [bp.name for bp in self.beam_processes]

        def match(fill):
            fillbp = set([bp.split("@")[0] for ts, bp in fill.beam_processes])
            return all([bp in fillbp for bp in bp_to_match])

        return sorted([ff.filln for ff in lhcrun.fills.values() if match(ff)])


class LHCProcess:
    knob_blacklist = set(
        [
            "LHCBEAM1/CMINUS_IM.IP7",
            "LHCBEAM1/CMINUS_RE.IP7",
            "LHCBEAM1/DP_TRIM_PERMIL",
            "LHCBEAM1/QH_TRIM",
            "LHCBEAM1/QH_TRIM_FIDEL",
            "LHCBEAM1/QH_TRIM_INT",
            "LHCBEAM1/QV_TRIM",
            "LHCBEAM1/QV_TRIM_FIDEL",
            "LHCBEAM1/QV_TRIM_INT",
            "LHCBEAM2/CMINUS_IM.IP7",
            "LHCBEAM2/CMINUS_RE.IP7",
            "LHCBEAM2/DP_TRIM_PERMIL",
            "LHCBEAM2/QH_TRIM",
            "LHCBEAM2/QH_TRIM_FIDEL",
            "LHCBEAM2/QH_TRIM_INT",
            "LHCBEAM2/QV_TRIM",
            "LHCBEAM2/QV_TRIM_FIDEL",
            "LHCBEAM2/QV_TRIM_INT",
            "LHCBEAM1/IP1_SEPSCAN_X_MM",
            "LHCBEAM1/IP1_SEPSCAN_Y_MM",
            "LHCBEAM1/IP2_SEPSCAN_X_MM",
            "LHCBEAM1/IP2_SEPSCAN_Y_MM",
            "LHCBEAM1/IP5_SEPSCAN_X_MM",
            "LHCBEAM1/IP5_SEPSCAN_Y_MM",
            "LHCBEAM1/IP8_SEPSCAN_X_MM",
            "LHCBEAM1/IP8_SEPSCAN_Y_MM",
            "LHCBEAM2/IP1_SEPSCAN_X_MM",
            "LHCBEAM2/IP1_SEPSCAN_Y_MM",
            "LHCBEAM2/IP2_SEPSCAN_X_MM",
            "LHCBEAM2/IP2_SEPSCAN_Y_MM",
            "LHCBEAM2/IP5_SEPSCAN_X_MM",
            "LHCBEAM2/IP5_SEPSCAN_Y_MM",
            "LHCBEAM2/IP8_SEPSCAN_X_MM",
            "LHCBEAM2/IP8_SEPSCAN_Y_MM",
        ]
    )

    def __init__(self, name, basedir, parent=None):
        self.name = name
        self.basedir = basedir
        self.parent = parent
        self.yaml = self.basedir / "readme.yaml"
        self.data = self.read_data(self.yaml)
        self.label = self.data.get("label", None)
        self.beamprocess = self.data.get("beamprocess", None)
        self.optics = {}
        self.energy = self.data.get("energy", None)
        self.charge = self.data.get("charge", None)

    def read_data(self, yaml=None):
        if yaml is None:
            yaml = self.yaml
        if not yaml.exists():
            print(f"Process data file {yaml} does not exist")
            return {}
        else:
            return read_yaml(yaml)

    def to_dict(self):
        return {
            "name": self.name,
            "label": self.label,
            "beamprocess": self.beamprocess,
        }

    def save_data(self):
        self.data = self.to_dict()
        write_yaml(self.data, self.yaml)

    def __repr__(self):
        return f"<Process {self.name}:{self.beamprocess!r} {len(self.optics)} optics>"

    def set_optics_from_lsa(self):
        tbl = get_lsa().getOpticTable(self.beamprocess)
        self.optics = {int(tt.time): str(tt.name) for tt in tbl}
        return self

    def get_last_setting(self, params=None, part="target"):
        """Get the last setting for the given parameters"""
        if params is None:
            params = set(self.parent.parent.knobs.lsa) - self.knob_blacklist

        out = []
        for param in params:
            try:
                print(f"getting {param}")
                trims = get_lsa().getLastTrim(
                    param, beamprocess=self.beamprocess, part=part
                )
                out.append(trims)
            except Exception as ex:
                print("Empty response for '%s': %s" % (param, ex))
        return out

    def get_settings(self, params=None, start=None, end=None, part="target"):
        """Get settings for the given parameters"""
        if start is None:
            t1 = (
                self.parent.start
                if self.parent.start is not None
                else self.parent.parent.start
            )

        if end is None:
            t2 = (
                self.parent.end
                if self.parent.end is not None
                else self.parent.parent.end
            )

        if params is None:
            params = list(self.parent.parent.knobs.lsa)

        out = {}
        for param in params:
            try:
                print(f"getting {param}")
                trims = get_lsa().getTrims(
                    param,
                    beamprocess=self.beamprocess,
                    start=t1,
                    end=t2,
                    part=part,
                )
                out.update(trims)
            except Exception as ex:
                print("Empty response for '%s': %s" % (param, ex))
        return out

    def clear_dir(self):
        """Clear the optitcs directory"""
        for ll in self.basedir.iterdir():
            if ll.is_dir():
                print(f"Removing {ll}")
                for ff in ll.iterdir():
                    if ff.is_file():
                        print(f"Removing {ff}")
                        os.remove(ff)
                os.rmdir(ll)

    def gen_optics_dir(self, lsa_settings=None):
        """Generate optics directory structure"""
        self.clear_dir()
        if lsa_settings is None:
            lsa_settings = self.get_settings()
        knobs = self.parent.parent.knobs
        for ts, name in self.optics.items():
            optics_dir = self.basedir / str(ts)
            optics_dir.mkdir(parents=True, exist_ok=True)
            yaml = optics_dir / "readme.yaml"
            data = {
                "name": name,
                "settings": {},
                "ts": ts,
                "particle": self.parent.particle,
                "charge": self.parent.charge,
            }
            for knobname, setting in sorted(lsa_settings.items()):
                if knobname not in self.knob_blacklist:
                    tdata, vdata = setting.data[-1]
                    value = float(np.interp(ts, tdata, vdata))
                    knob = knobs.lsa[knobname]
                    # trimtime = unixtime_to_string(setting.time[-1])
                    # print(f"Knob {knobname} value {value} at {trimtime}")

                    data["settings"][knob.mad] = knob.mad_value(value)

            print(f"Writing {yaml}")
            write_yaml(data, yaml)
            data["energy"] = data["settings"]["nrj"]
            data["optics_path"] = (
                f"acc-models-lhc/operation/optics/{name}.madx"
            )
            data["settings_path"] = optics_dir / "settings.madx"
            self.gen_madx_model(data, output=optics_dir / "model.madx")
            self.gen_madx_settings(data, output=optics_dir / "settings.madx")

    def gen_madx_model(self, data, output=None):
        """Generate MADX files for the optics"""
        madx = """
        call, file="acc-models-lhc/lhc.seq";
        beam,  sequence=lhcb1, particle={particle}, energy={energy}, charge={charge}, bv=1;
        beam,  sequence=lhcb2, particle={particle}, energy={energy}, charge={charge}, bv=-1;
        call, file="acc-models-lhc/{optics_path}";
        call, file="acc-models-lhc/{settings_path}";
        """.format(
            **data
        )
        if output is None:
            return madx
        else:
            with open(output, "w") as f:
                f.write(madx)
            print(f"Writing {output}")
            return madx

    def gen_madx_settings(self, data, output=None):
        """Generate MADX settings file"""
        madx = []
        for knobname, setting in sorted(data["settings"].items()):
            madx.append(f"{knobname}={setting};")
        if output is None:
            return madx
        else:
            with open(output, "w") as f:
                f.write(madx)
            print(f"Writing {output}")
            return madx


class LHCKnobDefs:
    @classmethod
    def from_file(cls, fn):
        lst = []
        for ln in open(fn):
            if ln.startswith("#"):
                continue
            if ln.strip() == "":
                continue
            lst.append(LHCKnobDef(*ln.strip().split(", ")))
        return cls(lst)

    def __init__(self, knob_list):
        self.mad = {}
        self.lsa = {}
        for knob in knob_list:
            self.mad[knob.mad] = knob
            self.lsa[knob.lsa] = knob

    def mad_value(self, lsa_name, lsa_value):
        """Return the MAD value for a given LSA name and LSA value"""
        return self.lsa[lsa_name].mad_value(lsa_value)

    def lsa_value(self, mad_name, mad_value):
        """Return the LSA value for a given MAD name and value"""
        return self.mad[mad_name].lsa_value(mad_value)

    def to_mad(self, lsa_name, lsa_value):
        """Return the MAD string for a given LSA name and LSA value"""
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
        self.add(LHCKnobDef(mad, lsa, scaling, test))

    def get_settings(self):
        return list(self.lsa.keys())

    def __repr__(self):
        return f"<LHCKnobDefs {len(self.lsa)} knobs>"


class LHCKnobDef:
    def __init__(self, mad, lsa, scaling, test):
        self.mad = str(mad).lower()
        self.lsa = str(lsa)
        self.scaling = float(scaling)
        self.test = float(test)

    def interpolate(self, ts, tdata, vdata):
        return np.interp(ts, tdata, vdata)

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
        fills = get_lsa().findBeamProcessHistory(
            self.t1, self.t2, accelerator="lhc"
        )
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


class LHCFill:
    def __init__(self, filln, beam_processes):
        self.filln = filln
        self.beam_processes = beam_processes
        self.start = min([ts for ts, bp in beam_processes])
        self.end = max([ts for ts, bp in beam_processes])
