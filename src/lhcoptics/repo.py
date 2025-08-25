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
    git_status,
    git_add,
    git_commit,
    git_push,
)
from .optics import LHCOptics


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
    return basedir


class LHC:
    """LHC optics repository

    The structure is as follows:
    branch_or_tag.cycle_or_collection.process_or_stage -> optics model

    """

    def __init__(self, basedir=None):
        self.basedir = check_repobasedir(basedir)
        self.cache_file = self.basedir / "cache.yaml"
        self.branches, self.tags = self.get_branches_and_tags()
        self.check_local_branches()
        self._repo_cache = {}

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

    def __getitem__(self, name):
        """Get the repository for a given branch or tag"""
        if name in self._repo_cache:
            return self._repo_cache[name]
        if name in self.branches or name in self.tags:
            repo_dir = self.basedir / name
            if not repo_dir.exists():
                print(f"Repository {repo_dir} does not exist")
                self.clone_repo(name)
            self._repo_cache[name] = LHCRepo(name, repo_dir, parent=self)
            return self._repo_cache[name]
        else:
            raise KeyError(f"No branch or tag named {name}")

    def _ipython_key_completions_(self):
        return list(self.branches.keys()) + list(self.tags.keys())

    def __repr__(self):
        return f"<LHC Repo at '{self.basedir}'>"

    def check_local_branches(self, dry_run=False, force_update=False):
        """
        Check if the local branches are up to date with the remote branches
        """
        if force_update:
            self.get_branches_and_tags(force_update=True)
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

    def get_branches_and_tags(self, force_update=False):
        """
        Get the list of branches in the repository
        """
        cache_file_good = self.cache_file.exists() and file_one_day_old(self.cache_file)
        if force_update or not cache_file_good:
            print("Getting branches and tags from gitlab")
            cache = gitlab_get_branches_and_tags(
                project_id=76507,
                gitlab_url="https://gitlab.cern.ch",
                timeout=0.5,
            )
            if cache is None:
                print("Error getting branches and tags from gitlab")
            else:
                write_yaml(cache, self.cache_file)

        if self.cache_file.exists():
            cache = read_yaml(self.cache_file)
            branches = cache["branches"]
            tags = cache["tags"]

            for name in list(branches):
                if len(name) != 4 or name[0] not in "2h":
                    del branches[name]

            self.branches = branches
            self.tags = tags

            return branches, tags
        else:
            raise ValueError("Error creating cache file")

    def clone_repo(self, branch_or_tag):
        """
        Manually the repository given a branch or tag
        """
        repo_dir = self.basedir / branch_or_tag
        if not repo_dir.exists():
            print(f"Cloning repository {repo_dir}")
            git_clone_repo(git_url, repo_dir, branch_or_tag)

    def refresh(self):
        """Refresh the repository"""
        self._repo_cache.clear()
        self.check_local_branches(force_update=True)


class LHCRepo:
    """LHC optics for a specific branch or tag"""

    def __init__(self, name, basedir, parent=None):
        self.name = str(name)
        self.basedir = Path(basedir)
        self.yaml = self.basedir / "scenarios" / "readme.yaml"
        self.parent = parent
        self.refresh()

    def __repr__(self):
        return f"<LHC {self.name} '{self.basedir}', {len(self.cycles)} cycles, {len(self.sets)} sets>"

    def __getattr__(self, name):
        if name in self.cycles:
            return self.cycles[name]
        elif name in self.sets:
            return self.sets[name]
        else:
            raise AttributeError(f"No cycle or set named {name}")

    def __dir__(self):
        return super().__dir__() + list(self.cycles.keys()) + list(self.sets.keys())

    def read_data(self):
        if self.yaml.exists():
            return read_yaml(self.yaml)
        else:
            print(f"Repo data file {self.yaml} does not exist")
            return {}

    def save_data(self):
        write_yaml(self.to_dict(), self.yaml)

    def refresh(self, data=None):
        if data is None:
            data = self.read_data()
        self.data = data
        self.knobs = self.get_knobs()
        self.cycles = self.get_cycles()
        self.sets = self.get_sets()
        self.start = self.data.get("start", None)
        self.end = self.data.get("end", None)
        self.label = self.data.get("label", None)

    def to_dict(self):
        return {
            "name": self.name,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "cycles": list(self.cycles),
            "sets": list(self.sets),
        }

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
            cycles[cycle_name] = LHCCycle(
                name=cycle_name,
                basedir=self.basedir / "scenarios" / "cycle" / cycle_name,
                parent=self,
            )
        return cycles

    def get_sets(self):
        sets = {}
        for set_name in self.data.get("sets", []):
            sets[set_name] = LHCOpticsSet(
                name=set_name,
                basedir=self.basedir / "scenarios" / "set" / set_name,
                parent=self,
            )
        return sets

    def add_cycle(self, name, label=None, particles=None, charges=None, masses=None):
        """Add a new cycle to the repository"""
        if name in self.cycles:
            raise ValueError(f"Cycle {name} already exists")
        (self.basedir / "scenarios" / "cycle" / name).mkdir(parents=True, exist_ok=True)
        cycle = LHCCycle(
            name=name,
            basedir=self.basedir / "scenarios" / "cycle" / name,
            parent=self,
        )
        cycle.label = label
        cycle.particles = particles
        cycle.charges = charges
        cycle.masses = masses
        cycle.save_data()
        self.cycles[name] = cycle
        self.save_data()
        return cycle

    def add_set(self, name, label=None):
        """Add a new optics set to the repository"""
        if name in self.cycles:
            raise ValueError(f"Optics set {name} already exists")
        (self.basedir / "scenarios" / "set" / name).mkdir(parents=True, exist_ok=True)
        optics_set = LHCOpticsSet(
            name=name,
            basedir=self.basedir / "scenarios" / "set" / name,
            parent=self,
        )
        optics_set.label = label
        optics_set.save_data()
        self.sets[name] = optics_set
        self.save_data()
        return optics_set

    def git_status(self):
        """Get the git status of the repository"""
        print(git_status(self.basedir))

    def git_add(self, *args):
        """Add the changes to the repository"""
        print(git_add(self.basedir, *args))

    def git_commit(self, message, *args):
        """Commit the changes to the repository"""
        print(git_commit(self.basedir, message, *args))

    def git_push(self, *args):
        print(git_push(self.basedir, *args))

    def get_xsuite_json(self):
        return self.basedir / "xsuite" / "lhc.json"

    def get_xsuite_env(self):
        import xtrack as xt

        return xt.Environment.from_json(self.get_xsuite_json())


class LHCCycle:
    """
    Represents an optics cycle in the LHC (Large Hadron Collider).

    This class manages the configuration and metadata for a specific optics cycle,
    including beam processes, particles, charges, masses, and associated optics.
    Attributes:
        name (str): Unique identifier for the cycle
        basedir (Path): Base directory for the cycle's files
        parent (object, optional): Parent object in the hierarchy
        beam_processes (dict): Dictionary of beam processes associated with the cycle
        start (str, optional): Start time or identifier of the cycle
        end (str, optional): End time or identifier of the cycle
        particles (tuple): Types of particles in the cycle
        charges (tuple): Charges of the particles
        masses (tuple): Masses of the particles
        label (str, optional): Human-readable label for the cycle
        optics (dict): Specific optics configuration details
    """

    def __init__(self, name, basedir, parent=None):
        self.name = name
        self.basedir = Path(basedir)
        if not basedir.exists():
            print(f"Cycle directory {basedir} does not exist")
        self.parent = parent
        self.yaml = self.basedir / "readme.yaml"
        self.refresh()

    def __getattr__(self, name):
        if name in self.beam_processes:
            return self.beam_processes[name]
        else:
            raise AttributeError(f"No process named {name}")

    def __dir__(self):
        return super().__dir__() + list(self.beam_processes.keys())

    def __repr__(self):
        return f"<Cycle {self.name!r} {len(self.beam_processes)} beam_processes>"

    def get_fills(self, lhcrun):
        bp_to_match = [bp.name for bp in self.beam_processes]

        def match(fill):
            fillbp = set([bp.split("@")[0] for ts, bp in fill.beam_processes])
            return all([bp in fillbp for bp in bp_to_match])

        return sorted([ff.filln for ff in lhcrun.fills.values() if match(ff)])

    def read_data(self):
        if self.yaml.exists():
            return read_yaml(self.yaml)
        else:
            print(f"Cycle data file {self.yaml} does not exist")
            return {}

    def refresh(self, data=None):
        if data is None:
            data = self.read_data()
        self.data = data
        self.beam_processes = {}
        for process in self.data.get("beam_processes", []):
            self.beam_processes[process] = LHCProcess(
                name=process,
                basedir=self.basedir / process,
                parent=self,
            )
        self.start = self.data.get("start")
        self.end = self.data.get("end")
        self.particles = self.data.get("particles", (None, None))
        self.charges = self.data.get("charges", (1, 1))
        self.masses = self.data.get("masses", (0.938272046, 0.938272046))
        self.label = self.data.get("label", None)
        self.optics = self.data.get("optics", {})

    def to_dict(self):
        return {
            "name": self.name,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "particles": self.particles,
            "charges": self.charges,
            "masses": self.masses,
            "beam_processes": list(self.beam_processes),
            "optics": self.optics,
        }

    def save_data(self):
        data = self.to_dict()
        write_yaml(data, self.yaml)

    def add_process(self, name, beam_process, label=None):
        """Add a new process to the cycle"""
        if name in self.beam_processes:
            raise ValueError(f"Process {name} already exists")
        (self.basedir / name).mkdir(parents=True, exist_ok=True)
        bp = LHCProcess(
            name=name,
            basedir=self.basedir / name,
            parent=self,
        )
        bp.label = label
        bp.beam_process = beam_process
        bp.save_data()
        self.beam_processes[name] = bp
        self.save_data()
        return bp

    def gen_data_from_lsa(self):
        """Generate the data from LSA"""
        for process in self.beam_processes.values():
            process.gen_data_from_lsa()
        self.save_data()

    def gen_eos_data(self, eos_repo_path=None):
        for process in self.beam_processes.values():
            process.gen_eos_data(eos_repo_path=eos_repo_path)

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
        self.refresh()

    def __getitem__(self, idx):
        return self.get_lhcoptics(idx=idx)

    def __repr__(self):
        return f"<Process {self.name}:{self.beam_process!r} {len(self.optics)} optics>"

    @property
    def ts(self):
        return list(self.optics.keys())

    def _lsa_settings_to_dict(self, lsa_settings):
        """Convert LSA settings to a dictionary"""
        knobs = self.parent.parent.knobs
        settings = {}
        for knobname, setting in sorted(lsa_settings.items()):
            if knobname not in self.knob_blacklist:
                tdata, vdata = setting.data
                madname = knobs.lsa[knobname].mad
                settings[madname] = [tdata.tolist(), vdata.tolist()]
        return settings

    def check_madx(self, idx=None, ts=None, madx=None):
        """Check the MADX model for the given time step"""
        if idx is not None:
            ts = self.ts[idx]
        tw1, tw2 = self.get_madx_twiss(ts=ts, madx=madx)
        tw1.rows["ip.*"].cols["betx bety x*1e3 y*1e3 px*1e6 py*1e6"].show()
        tw2.rows["ip.*"].cols["betx bety x*1e3 y*1e3 px*1e6 py*1e6"].show()

    def check_settings_with_lsa(self, lsa_settings=None):
        if lsa_settings is None:
            lsa_settings = self.get_last_settings_from_lsa(verbose=False)
        lsa = self._lsa_settings_to_dict(lsa_settings)
        for kk in {*lsa.keys(), *self.settings.keys()}:
            if kk not in lsa:
                print(f"LSA setting {kk!r} not found\n")
            elif kk not in self.settings:
                print(f"Setting {kk!r} not found in optics settings\n")
            else:
                if lsa[kk] != self.settings[kk]:
                    print(f"Setting {kk!r} does not match LSA settings")
                    print(f"LSA: {lsa[kk]}")
                    print(f"Optics: {self.settings[kk]}\n")
        return lsa == self.settings

    def clear_dir(self):
        """Clear the optitcs directory"""
        for ll in self.basedir.iterdir():
            if ll.is_dir():
                print(f"Removing {ll}")
                for ff in ll.iterdir():
                    if ff.is_file() or ff.is_symlink():
                        print(f"Removing {ff}")
                        os.remove(ff)
                os.rmdir(ll)

    def gen_data_from_lsa(self):
        """Generate the data from LSA"""
        self.set_optics_from_lsa()
        self.set_settings_from_lsa()
        self.save_data()
        self.gen_optics_dir()

    def gen_eos_data(self, eos_repo_path=None):
        """
        Generate EOS data for the given time step

        - eos_repo_path: is the root of the repository in EOS
        """
        eos_path = self.get_eos_path(eos_repo_path=eos_repo_path)

        for idx, ts in enumerate(self.optics):
            eos_ts_path = eos_path / str(ts)
            eos_ts_path.mkdir(parents=True, exist_ok=True)
            eos_madx_optics_path = eos_ts_path / "optics.madx"
            eos_lhcoptics_path = eos_ts_path / "optics.json"
            print(f"Generating optics at {eos_ts_path}")
            opt = self.get_lhcoptics(idx=idx)
            opt.to_json(eos_lhcoptics_path)
            opt.to_madx(eos_madx_optics_path)

            madx = self.get_madx_model(idx=idx)
            madx.use("lhcb1")
            madx.twiss(file=str(eos_ts_path / "twiss_lhcb1.tfs"))
            madx.use("lhcb2")
            madx.twiss(file=str(eos_ts_path / "twiss_lhcb2.tfs"))

            # Copy the model and settings files to EOS
            # opt = self.get_lhcoptics(idx=idx)
            # opt.to_madx(

        return eos_path

    def gen_madx_model(self, data, output=None):
        """Generate MADX files for the optics"""
        madx = """\
call, file="acc-models-lhc/lhc.seq";
beam,  sequence=lhcb1, particle={particles[0]}, energy={energies[0]}, charge={charges[0]}, mass={masses[0]}, bv=1;
beam,  sequence=lhcb2, particle={particles[1]}, energy={energies[1]}, charge={charges[1]}, mass={masses[1]}, bv=-1;
call, file="acc-models-lhc/{optics_path}";
call, file="acc-models-lhc/{settings_path}";""".format(**data)
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
        madx = "\n".join(madx)
        if output is None:
            return madx
        else:
            with open(output, "w") as f:
                f.write(madx)
            print(f"Writing {output}")
            return madx

    def gen_optics_dir(self, lsa_settings=None):
        """Generate optics directory structure"""
        self.clear_dir()
        for ts, name in self.optics.items():
            optics_dir = self.basedir / str(ts)
            optics_dir.mkdir(parents=True, exist_ok=True)
            yaml = optics_dir / "readme.yaml"
            charges = self.parent.charges
            data = {
                "name": name,
                "settings": {},
                "ts": ts,
                "particles": self.parent.particles,
                "charges": charges,
                "masses": self.parent.masses,
            }
            for knobname, (tdata, vdata) in self.settings.items():
                value = float(np.interp(ts, tdata, vdata))
                data["settings"][knobname] = value

            print(f"Writing {yaml}")
            write_yaml(data, yaml)
            data["energies"] = (
                data["settings"]["nrj"] * charges[0],
                data["settings"]["nrj"] * charges[1],
            )
            data["optics_path"] = f"{name}"
            reldir = optics_dir.relative_to(self.parent.parent.basedir)
            data["settings_path"] = reldir / "settings.madx"
            self.gen_madx_model(data, output=optics_dir / "model.madx")
            self.gen_madx_settings(data, output=optics_dir / "settings.madx")
            (optics_dir / "acc-models-lhc").symlink_to(
                "../../../../..", target_is_directory=True
            )

    def get_eos_path(self, eos_repo_path=None):
        """Get the EOS path for the given time step
        If idx is provided, it will use the corresponding time step from self.ts.
        """
        repo = self.parent.parent
        time_step_relative_path = self.basedir.relative_to(repo.basedir)
        if eos_repo_path is None:
            eos_repo_path = Path("/eos/project-a/acc-models/public/lhc/") / repo.name
        else:
            eos_repo_path = Path(eos_repo_path)
        return eos_repo_path/time_step_relative_path

    def get_lhcoptics(self, idx=None, ts=None, xsuite=True):
        """Get the LHC optics for the given time step"""
        if idx is not None:
            ts = self.ts[idx]
        madx = self.get_madx_model(ts=ts)
        name = f"{self.parent.name}_{self.name}_{ts}"
        if xsuite is True:
            xsuite_model = self.parent.parent.basedir / "xsuite" / "lhc.json"
        else:
            xsuite_model = None
        opt = LHCOptics.from_madx(
            madx=madx,
            name=name,
            xsuite_model=xsuite_model,
        )
        opt.set_params()
        return opt

    def get_lhcoptics_from_eos(self, idx=None, ts=None, xsuite=True):
        """Get the LHC optics for the given time step"""
        if idx is not None:
            ts = self.ts[idx]
        eos_path = self.get_eos_path()
        if eos_path.exists():
            eos_optics = eos_path / str(ts) / "optics.json"
            if eos_optics.exists():
                opt = LHCOptics.from_json(eos_optics)
                if xsuite is True:
                    xsuite_model = self.parent.parent.basedir / "xsuite" / "lhc.json"
                    opt.set_xsuite_model(xsuite_model)
            else:
                raise FileNotFoundError(f"Optics file {eos_optics} does not exist")
        return opt

    def get_madx_model(self, idx=None, ts=None, madx=None, stdout=False):
        """Get the MADX model for the given time step
        If idx is provided, it will use the corresponding time step from self.ts.

        """
        if idx is not None:
            ts = self.ts[idx]
        if ts in self.optics:
            optics_dir = self.basedir / str(ts)
            # madxfile = optics_dir / "model.madx"
            madxfile = "model.madx"
            if madx is None:
                from cpymad.madx import Madx
                madx = Madx(stdout=stdout)
            madx.chdir(str(optics_dir))
            madx.call(str(madxfile))
            return madx
        else:
            raise ValueError(f"Optics {ts} not found in {self.name}")

    def get_madx_twiss(self, idx=None, ts=None, madx=None):
        """Get the MADX twiss for the given time step"""
        if idx is not None:
            ts = self.ts[idx]
        madx = self.get_madx_model(ts=ts, madx=madx)
        madx.use("lhcb1")
        tw1 = madx.twiss()
        madx.use("lhcb2")
        tw2 = madx.twiss()
        from xdeps import Table

        return Table(tw1), Table(tw2)

    def get_last_settings_from_lsa(self, params=None, part="target", verbose=False):
        """Get the last setting for the given parameters"""
        if params is None:
            params = set(self.parent.parent.knobs.lsa) - self.knob_blacklist

        out = {}
        for param in params:
            try:
                if verbose:
                    print(f"getting {param}")
                trims = get_lsa().getLastTrim(
                    param, beamprocess=self.beam_process, part=part
                )
                out[param] = trims
            except Exception as ex:
                if verbose:
                    print("Empty response for '%s': %s" % (param, ex))
        return out

    def get_settings_from_lsa(
        self, params=None, start=None, end=None, part="target", verbose=False
    ):
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
            params = set(self.parent.parent.knobs.lsa) - self.knob_blacklist

        out = {}
        for param in params:
            try:
                if verbose:
                    print(f"getting {param}")
                trims = get_lsa().getTrims(
                    param,
                    beamprocess=self.beam_process,
                    start=t1,
                    end=t2,
                    part=part,
                )
                out.update(trims)
            except Exception as ex:
                if verbose:
                    print("Empty response for '%s': %s" % (param, ex))
        return out

    def read_data(self):
        if self.yaml.exists():
            return read_yaml(self.yaml)
        else:
            print(f"Process data file {self.yaml} does not exist")
            return {}

    def refresh(self, data=None):
        if data is None:
            data = self.read_data()
        self.data = data
        self.label = self.data.get("label", None)
        self.beam_process = self.data.get("beam_process", None)
        self.optics = self.data.get("optics", {})
        self.settings = self.data.get("settings", {})

    def to_dict(self):
        return {
            "name": self.name,
            "label": self.label,
            "beam_process": self.beam_process,
            "optics": self.optics,
            "settings": self.settings,
        }

    def save_data(self):
        data = self.to_dict()
        from ruamel.yaml import CommentedMap, CommentedSeq

        data["optics"] = CommentedMap(data["optics"])
        data["settings"] = CommentedMap(data["settings"])
        data["settings"].fa.set_block_style()
        for knobname, setting in sorted(data["settings"].items()):
            data["settings"][knobname] = CommentedSeq(setting)
            data["settings"][knobname].fa.set_flow_style()
        write_yaml(data, self.yaml)

    def set_optics_from_lsa(self):
        tbl = get_lsa().getOpticTable(self.beam_process)
        self.optics = {int(tt.time): f"operation/optics/{tt.name}.madx" for tt in tbl}
        return self

    def set_settings_from_lsa(self):
        """Get the settings from LSA"""
        self.settings = {}
        lsa_settings = self.get_settings_from_lsa()
        knobs = self.parent.parent.knobs
        for knobname, setting in sorted(lsa_settings.items()):
            if knobname not in self.knob_blacklist:
                tdata, vdata = setting.data[-1]
                knob = knobs.lsa[knobname]
                vdata = knob.mad_value(vdata)
                self.settings[knob.mad] = tdata.tolist(), vdata.tolist()


class LHCOpticsSet:
    """LHC optics set"""

    def __init__(self, name, basedir, parent=None):
        self.name = name
        self.basedir = basedir
        if not basedir.exists():
            print(f"Optics set directory {basedir} does not exist")
        self.parent = parent
        self.yaml = self.basedir / "readme.yaml"
        self.refresh()

    def __getattr__(self, name):
        if name in self.optics:
            return self.optics[name]
        else:
            raise AttributeError(f"No optics named {name}")

    def __dir__(self):
        return super().__dir__() + list(self.optics.keys())

    def __repr__(self):
        return f"<OpticsSet {self.name!r} {len(self.optics)} optics>"

    def read_data(self):
        if self.yaml.exists():
            return read_yaml(self.yaml)
        else:
            print(f"Optics set data file {self.yaml} does not exist")
            return {}

    def refresh(self, data=None):
        if data is None:
            data = self.read_data()
        self.data = data
        self.label = self.data.get("label", None)
        self.optics = {}
        for opt in self.data.get("optics", []):
            self.optics[opt] = LHCOpticsDef(
                name=opt,
                basedir=self.basedir / opt,
                parent=self,
            )

    def to_dict(self):
        return {
            "name": self.name,
            "label": self.label,
            "optics": list(self.optics),
        }

    def save_data(self):
        data = self.to_dict()
        from ruamel.yaml import CommentedSeq

        data["optics"] = CommentedSeq(data["optics"])
        data["optics"].fa.set_block_style()
        write_yaml(data, self.yaml)

    def add_optics(
        self,
        name,
        label=None,
        optics=None,
        settings=None,
        particles=None,
        charges=None,
        masses=None,
    ):
        """Add a new optics to the set"""
        if name in self.optics:
            raise ValueError(f"Optics {name} already exists")
        (self.basedir / name).mkdir(parents=True, exist_ok=True)
        optics_def = LHCOpticsDef(
            name=name,
            basedir=self.basedir / name,
            parent=self,
        )
        optics_def.refresh(
            data=dict(
                name=name,
                label=label,
                optics=optics,
                settings=settings,
                particles=particles,
                charges=charges,
                masses=masses,
            )
        )
        optics_def.save_data()
        self.optics[name] = optics_def
        self.save_data()
        return optics


class LHCOpticsDef:
    """LHC optics definition

    It is defined by an optics file and a set of settings
    """

    def __init__(
        self,
        name,
        basedir,
        parent=None,
    ):
        self.name = name
        self.basedir = basedir
        self.parent = parent
        self.yaml = self.basedir / "readme.yaml"
        self.refresh()

    def __repr__(self):
        return f"<OpticsDef {self.name}: {self.optics!r}>"

    def read_data(self):
        if self.yaml.exists():
            return read_yaml(self.yaml)
        else:
            print(f"Optics definition data file {self.yaml} does not exist")
            return {}

    def refresh(self, data=None):
        if data is None:
            self.data = self.read_data()
        else:
            self.data.update(data)
        self.name = self.data.get("name", None)
        self.label = self.data.get("label", None)
        self.optics = self.data.get("optics", None)
        self.settings = self.data.get("settings", {})
        self.particles = self.data.get("particles", ("proton", "proton"))
        self.charges = self.data.get("charges", (1, 1))
        self.masses = self.data.get("masses", (0.938272046, 0.938272046))

    def to_dict(self):
        return {
            "name": self.name,
            "label": self.label,
            "optics": self.optics,
            "settings": self.settings,
            "particles": self.particles,
            "charges": self.charges,
            "masses": self.masses,
        }

    def save_data(self):
        data = self.to_dict()
        for k, v in data.items():
            if k in set("particles charges masses") and v is None:
                del data[k]
        write_yaml(data, self.yaml)


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
        self.lsa = lsa  # string or None
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
        fills = get_lsa().findBeamProcessHistory(self.t1, self.t2, accelerator="lhc")
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
