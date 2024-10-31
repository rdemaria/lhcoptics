import json
import re
import time
from datetime import datetime

import matplotlib.pyplot as plt
#import matplotlib.dates as mdates
import numpy as np

from .lsa_util import get_lsa
#from .nxcals_util import get_nxcals

def str_to_unixtime(datestr):
    return time.mktime(time.strptime(datestr,"%Y-%m-%d %H:%M:%S.%f"))

def format_xticks_as_datetime():
    ax=plt.gca()
    unix_times = ax.get_xticks()
    ax.set_xticks(unix_times)
    datetime_labels = [datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') for ts in unix_times]
    ax.set_xticklabels(datetime_labels, rotation=45, ha='right')
    plt.tight_layout()

kqx_limits = {
    "kqx.l1": [0, 205.0],
    "kqx.r1": [-205.0, 0],
    "kqx.l2": [0, 205.0],
    "kqx.r2": [-205.0, 0],
    "kqx.l5": [-205.0, 0],
    "kqx.r5": [0, 205.0],
    "kqx.l8": [0, 205.0],
    "kqx.r8": [-205.0, 0],
}


def madname_from_pcname(pc):
    name = ".".join(pc.split(".")[2:]).lower()
    if name.startswith("rcb"):
        return "a" + name[1:]
    else:
        return "k" + name[1:]


class LHCCircuit:

    def __init__(
        self,
        pcname,
        logicalname=None,
        madname=None,
        imin=0,
        imax=0,
        inominal=0,
        iultimate=0,
        vmin=0,
        vmax=0,
        ipmin=0,
        ipmax=0,
        ippmin=0,
        ippmax=0,
        vpmin=0,
        vpmax=0,
        polarity=0,
        calibname=None,
        calibsign=None,
        r1=0,
        l1=0,
        c1=0,
    ):
        self.pcname = pcname
        self.logicalname = logicalname
        self.madname = madname
        self.calibname = calibname
        self.calibsign = calibsign
        self.r1 = r1
        self.l1 = l1
        self.c1 = c1
        self.imax = imax
        self.imin = imin
        self.inominal = inominal
        self.iultimate = iultimate
        self.vmax = vmax
        self.vmin = vmin
        self.ipmax = ipmax
        self.ipmin = ipmin
        self.ippmax = ippmax
        self.ippmin = ippmin
        self.vpmax = vpmax
        self.vpmin = vpmin
        self.polarity = polarity

    @staticmethod
    def get_logicalhardware_from_lsa(pcname, first=True):
        """
        Contains also Rtot,Ltot, tau
        """
        lst = (
            get_lsa()
            ._deviceService.findLogicalHardwaresByActualDeviceNames([pcname])
            .get(pcname)
            .toArray()
        )
        if first:
            return lst[0]
        else:
            return lst

    @staticmethod
    def get_pcinfo_from_lsa(pcname):
        return get_lsa()._deviceService.findPowerConverterInfo(pcname)

    @classmethod
    def from_lsa(cls, pcname):
        pcinfo = cls.get_pcinfo_from_lsa(pcname)
        lh = cls.get_logicalhardware_from_lsa(pcname)
        pc = cls(
            pcname=pcname,
            logicalname=lh.getName(),
            madname=lh.getMadStrengthName(),
            calibname=lh.getCalibrationName(),
            calibsign=lh.getCalibrationSign(),
            imin=pcinfo.getIMinOp(),
            imax=pcinfo.getIPNo(),
            inominal=pcinfo.getINom(),
            iultimate=pcinfo.getIUlt(),
            ipmin=pcinfo.getDidtMin(),
            ipmax=pcinfo.getDidtMax(),
            ippmax=pcinfo.getAccelerationLimit(),
            ippmin=pcinfo.getDecelerationLimit(),
        )
        # patch RTQX
        if "RTQX" in pcname:
            pc.madname = f"ktqx{pcname[-4:].lower()}"
        elif ".RQF." in pcname:
            pc.calibname = lh.getName() + "B1"
            pc.calibsign = 1
        elif ".RQD." in pcname:
            pc.calibname = lh.getName() + "B1"
            pc.calibsign = -1
        return pc

    @classmethod
    def from_dict(cls, dct):
        return cls(**dct)

    def to_dict(self):
        return {
            k: v for k, v in self.__dict__.items() if not k.startswith("_")
        }

    def get_field(self, current):
        return self._calib.get_field(current) * self.calibsign

    def get_current(self, k, p0c=7e12):
        return self._calib.get_current(k, p0c) * self.calibsign

    def get_klimits(self, pc=7e12):
        brho = pc / 299792458
        limits = self.get_field([self.imin, self.imax]) / brho
        limits.sort()
        return limits

    def get_lsa_pcinfo(self):
        return LHCCircuit.get_pcinfo_from_lsa(self.pcname)

    def get_lsa_logicalhardware(self):
        return LHCCircuit.get_logicalhardware_from_lsa(self.pcname)

    def __repr__(self) -> str:
        return f"<LHCCircuit {self.pcname!r}>"

    def plot_calib(self):
        self._calib.plot()

    def plot_calib_deviation(self):
        self._calib.plot_deviation()

    def get_measurements(self,db, start, end):
        if isinstance(start, str):
            start = str_to_unixtime(start)
        if isinstance(end, str):
            end = str_to_unixtime(end)
        return db.get(self.get_nxcals_variables(), start, end)

    def get_nxcals_variables(self):
        return [f"{self.pcname}:{ss}" for ss in ["I_MEAS", "V_MEAS","I_REF","V_REF"]]


class LHCCalibration:

    def __init__(self, name, current, field, fieldtype):
        self.name = name
        self.current = current
        self.field = field
        self.fieldtype = fieldtype

    @classmethod
    def from_lsa_calib(cls, calib):
        name = calib.getName()
        map = calib.getCalibrationFunctionMap()
        if len(map) == 0:
            return None
        for calibfuntype in map.keys():
            if calibfuntype.getFunctionTypeName() == "B_FIELD":
                break
        calibfun = map[calibfuntype]
        field = np.array(calibfun.toXArray())
        current = np.array(calibfun.toYArray())
        return cls(
            name=name,
            current=current,
            field=field,
            fieldtype=calibfuntype.getFunctionTypeName(),
        )

    @classmethod
    def from_dict(cls, dct):
        dct["current"] = np.array(dct["current"])
        dct["field"] = np.array(dct["field"])
        return cls(**dct)

    @property
    def imin(self):
        return self.current.min()

    @property
    def imax(self):
        return self.current.max()

    @property
    def fmin(self):
        return self.field.min()

    @property
    def fmax(self):
        return self.field.max()

    def get_field(self, current):
        return np.interp(current, self.current, self.field)

    def get_current(self, k, p0c=7e12):
        brho = p0c / 299792458
        return np.interp(k * brho, self.field, self.current)

    def to_dict(self):
        dct = self.__dict__.copy()
        dct["current"] = dct["current"].tolist()
        dct["field"] = dct["field"].tolist()
        return dct

    def plot(self):
        plt.figure(num=self.name)
        plt.plot(self.current, self.field, "k", label="Field")
        plt.xlabel("Current [A]")
        plt.ylabel("Field [T/m^n]")
        dfdi = (self.field[-1] - self.field[0]) / (
            self.current[-1] - self.current[0]
        )
        plt.plot(
            self.current, dfdi * self.current, "r", label="Field Deviation"
        )
        plt.title(self.name)
        plt.grid(True)
        plt.legend()

    def plot_deviation(self):
        plt.figure(num=self.name + " deviation")
        dfdi = (self.field[-1] - self.field[0]) / (
            self.current[-1] - self.current[0]
        )
        plt.plot(self.current, self.field - dfdi * self.current, "k")
        plt.xlabel("Current [A]")
        plt.ylabel("Field Deviation [T/m^n]")
        plt.title(self.name + " deviation")
        plt.grid(True)

    def __repr__(self) -> str:
        if len(self.current) > 0:
            ilim = f"{self.imin:.0f}:{self.imax:.0f}"
            flim = f"{self.fmin:.3g}:{self.fmax:.3g}"
            return f"<LHCCalibration {self.name}, I {ilim}, {self.fieldtype} {flim}>"
        else:
            return f"<LHCCalibration {self.name}, empty>"



class LHCCircuits:

    def __init__(self, circuits, calibrations=None, circuits2in1=None):
        self.pcname = circuits
        if calibrations is None:
            calibrations = {}
        self.calibrations = calibrations
        for cir in self.pcname.values():
            cir._calib = self.calibrations.get(cir.calibname)
            if cir._calib is not None and cir._calib.imin < 0:
                cir.imin = -cir.imax

        self.madname = {}
        self.logicalname = {}
        for cir in self.pcname.values():
            if hasattr(cir, "madname"):
                self.madname[cir.madname] = cir
                logicalname = ".".join(cir.pcname.split(".")[2:])
                self.logicalname[logicalname] = cir
        if circuits2in1 is None or len(circuits2in1) == 0:
            circuits2in1 = {}
            for pcname1, pc1 in list(self.pcname.items()):
                if pcname1.endswith("B1"):
                    name2in1 = pcname1[:-2]
                    pcname2 = name2in1 + "B2"
                    pc2 = self.pcname.get(pcname2)
                    if pc2 is not None:
                        circuits2in1[name2in1] = LHCCircuit2in1(
                            name2in1, pcname1, pcname2, pc1, pc2
                        )
        else:
            for cir in circuits2in1.values():
                cir.pc1 = self.pcname[cir.pcname1]
                cir.pc2 = self.pcname[cir.pcname2]

        self.pcname2in1 = circuits2in1
        self.madname2in1 = {}
        self.logicalname2in1 = {}
        for cir in self.pcname2in1.values():
            self.madname2in1[cir.pc1.madname] = cir
            self.madname2in1[cir.pc2.madname] = cir
            self.logicalname2in1[cir.pc1.logicalname] = cir
            self.logicalname2in1[cir.pc2.logicalname] = cir
            self.logicalname2in1[cir.pc1.logicalname[:-2]] = cir

    @staticmethod
    def get_pc_names_from_lsa():
        lsa = get_lsa()
        pcs = lsa.findParameterNames(groupName="ALL MAGNETS", regexp=".*/IREF")
        pcs = [pc.split("/IREF")[0] for pc in pcs]
        return pcs

    @staticmethod
    def get_calibs_from_lsa(calibnames):
        lsa = get_lsa()
        # req = lsa._cern.lsa.domain.devices.CalibrationsRequest.byCalibrationNames(
        #    calibnames
        # )
        # temp fix for the triplets
        req = lsa._cern.lsa.domain.devices.CalibrationsRequest.ALL
        calibs = lsa._deviceService.findCalibrations(req)
        calibrations = {}
        for lsacalib in calibs:
            calib = LHCCalibration.from_lsa_calib(lsacalib)
            if calib is not None:
                calibrations[calib.name] = calib
        return calibrations

    @classmethod
    def from_lsa(cls):
        pcnames = cls.get_pc_names_from_lsa()
        circuits = {}
        for pcname in pcnames:
            cir = LHCCircuit.from_lsa(pcname)
            circuits[cir.pcname] = cir

        calibnames = {
            cir.calibname
            for cir in circuits.values()
            if cir.calibname is not None
        }
        calibrations = cls.get_calibs_from_lsa(calibnames)
        return cls(circuits, calibrations=calibrations)

    @classmethod
    def from_dict(cls, dct):
        circuits = {k: LHCCircuit.from_dict(v) for k, v in dct["circuits"].items()}
        calibrations = {
            k: LHCCalibration.from_dict(v)
            for k, v in dct["calibrations"].items()
        }
        circuits2in1 = {
            k: LHCCircuit2in1.from_dict(v)
            for k, v in dct.get("circuits2in1", {}).items()
        }
        return cls(circuits, calibrations=calibrations, circuits2in1=circuits2in1)

    @classmethod
    def from_json(cls, filename):
        with open(filename) as f:
            data = json.load(f)
            return cls.from_dict(data)

    def to_dict(self):
        return {
            "circuits": {k: v.to_dict() for k, v in self.pcname.items()},
            "calibrations": {
                k: v.to_dict() for k, v in self.calibrations.items()
            },
            "circuits2in1": {
                k: v.to_dict() for k, v in self.pcname2in1.items()
            },
        }

    def to_json(self, filename):
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return self

    def get_2in1(self, madname=None, logicalname=None):
        if madname:
            pc = self.madname[madname]
        elif logicalname:
            pc = self.logicalname[logicalname]
        return self.pcname[pc.pcname[:-2]]

    def get_current(self, kname, kval, pc=7e12):
        return self.pcname[kname].get_current(kval, pc)

    def get_field(self, kname, current):
        return self.pcname[kname].get_field(current)

    def get_klimits(self, kname, pc=7e12):
        if kname.startswith("kqx"):
            brho = pc / 299792458
            minval, maxval = kqx_limits[kname]
            return [minval / brho, maxval / brho]
        elif kname.startswith("ktqx"):
            brho = pc / 299792458
            minval, maxval = [-20, 20]  ## arbitrary limits
            return [minval / brho, maxval / brho]
        else:
            return self.madname[kname].get_klimits(pc)

    def get_current_triplet_from_trims(self, side, kqx, ktqx1, ktqx2, pc=7e12):
        k3 = kqx
        k1 = ktqx1 + kqx
        k2 = -ktqx2 - kqx
        return self.get_current_triplet(side, k1, k2, k3, pc)

    def get_current_triplet(self, side, k1, k2, k3, pc=7e12):
        side = side.upper()
        n1 = f"MQXA1.{side}"
        n2 = f"MQXB2.{side}"
        n3 = f"MQXA3.{side}"
        i1 = self.calibrations[n1].get_current(k1, pc)
        i2 = self.calibrations[n2].get_current(k2, pc)
        i3 = self.calibrations[n3].get_current(k3, pc)
        return i1, i2, i3

    def get_field_triplet(self, side, i1, i2, i3):
        side = side.upper()
        n1 = f"MQXA1.{side}"
        n2 = f"MQXB2.{side}"
        n3 = f"MQXA3.{side}"
        f1 = self.calibrations[n1].get_field(i1)
        f2 = self.calibrations[n2].get_field(i2)
        f3 = self.calibrations[n3].get_field(i3)
        return f1, f2, f3

    def get_current_triplet_trims(self, side, k1, k2, k3, pc=7e12):
        i1, i2, i3 = self.get_current_triplet(side, k1, k2, k3, pc)
        return {
            f"RQX.{side}": i3,
            f"RTQX1.{side}": i3 - i1,
            f"RQTX2.{side}": i2 - i3,
        }

    def find_calib(self, name):
        return [
            calib
            for cname, calib in self.calibrations.items()
            if re.match(name, cname)
        ]

    def find_pc(self, madname=None, pcname=None, logicalname=None):
        out = []
        if madname:
            out += [
                pc
                for pc in self.pcname.values()
                if pc.madname and re.match(madname, pc.madname)
            ]
        if pcname:
            out += [
                pc
                for pc in self.pcname.values()
                if re.match(pcname, pc.pcname)
            ]
        if logicalname:
            out += [
                pc
                for pc in self.pcname.values()
                if re.match(logicalname, pc.logicalname)
            ]
        return out

    def __getitem__(self, key):
        if key in self.pcname:
            return self.pcname[key]
        elif key in self.madname:
            return self.madname[key]
        else:
            raise KeyError(f"Key {key} not found in {self}")

    def __repr__(self) -> str:
        return f"<LHCCircuits {len(self.pcname)} circuits, {len(self.pcname2in1)} 2-in-1>"


class LHCCircuit2in1:
    def __init__(self, pcname, pcname1, pcname2, pc1=None, pc2=None, rc=0):
        self.pcname = pcname
        self.pcname1 = pcname1
        self.pcname2 = pcname2
        self.pc1 = pc1
        self.pc2 = pc2
        self.rc = rc

    @classmethod
    def from_dict(cls, dct):
        return cls(**dct)

    def to_dict(self):
        attrs = ["pcname", "pcname1", "pcname2", "rc"]
        return {k: getattr(self, k) for k in attrs}

    def __repr__(self) -> str:
        return f"<LHCCircuit2in1 {self.pcname!r}>"
