from pathlib import Path
from .optics import LHCOptics
import yaml


from yaml import load, dump

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def read_yaml(filename):
    with open(filename) as f:
        return load(f, Loader=Loader)


class LHC:
    """LHC optics repository"""

    @classmethod
    def from_repository(cls, version):
        import subprocess

        accmodels = Path("acc-models-lhc")
        if accmodels.exists():
            if not (accmodels / "lhc.seq").exists():
                raise FileNotFoundError("acc-models-lhc/lhc.seq not found")
            else:
                if (accmodels / ".git").exists():
                    subprocess.run(["git", "switch", version], cwd=accmodels)
        elif (
            lcl := (Path.home() / "local" / "acc-models-lhc" / version)
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
        return cls(accmodels, version)

    def __init__(self, basedir, version, cycles=None):
        self.basedir = basedir
        self.cycles_dir= basedir / "scenarios"
        self.version = version
        self.cycles = self.get_cycles() if cycles is None else cycles

    def get_cycles(self):
        return {
            name: LHCCycle(name) for name in read_yaml(self.cycle_dir/"cycles.yaml")
        }

    def __getattr__(self, name: str):
        if name in self.cycles:
            return self.cycles[name]
        else:
            raise AttributeError(f"{name} not found in LHC repository")

    class LHCCycle:
        def __init__(self, name, processes=None):
            self.name = name
            self.processes = self.get_processes() if processes is None else processes

        def get_processes(self):
            return {
                name: LHCProcess(name, beamprocess)
                for name, beamprocess in read_yaml(self.path / f"{self.name}.yaml")
            }

    def __getattr__(self, name: str):
        if name in self.processes:
            return self.processes[name]
        else:
            raise AttributeError(f"{name} not found in LHC repository")

    