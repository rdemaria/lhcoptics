import json
import numpy as np


def madname_from_pcname(pc):
    name = ".".join(pc.split(".")[2:]).lower()
    if name.startswith("rcb"):
        return "a" + name[1:]
    else:
        return "k" + name[1:]


class LHCCircuits:
    @staticmethod
    def get_pc_names_from_lsa(lsa):
        pcs = lsa.findParameterNames(groupName="ALL MAGNETS", regexp=".*/IREF")
        pcs = [pc.split("/IREF")[0] for pc in pcs]
        return pcs

    @staticmethod
    def get_calibs_from_lsa(calibnames,lsa):
        req=lsa._cern.lsa.domain.devices.CalibrationsRequest.byCalibrationNames(calibnames)
        calibs = lsa._deviceService.findCalibrations(req)
        calibrations={}
        for lsacalib in calibs:
            calib=LHCCalibration.from_lsa_calib(lsacalib)
            calibrations[calib.name]=calib
        return calibrations

    @classmethod
    def from_lsa(cls, lsa=None):
        if lsa is None:
            import pjlsa

            lsa = pjlsa.LSAClient()
        pcnames = cls.get_pc_names_from_lsa(lsa)
        circuits = {}
        for pcname in pcnames:
            cir = LHCCircuit.from_lsa(pcname, lsa)
            circuits[cir.madname] = cir
        calibnames={cir.calibname for cir in circuits.values() if cir.calibname is not None} 
        calibrations = cls.get_calibs_from_lsa(calibnames,lsa)
        return cls(circuits, calibrations=calibrations)

    @classmethod
    def from_dict(cls, dct):
        circuits = {k: LHCCircuit.from_dict(v) for k, v in dct["circuits"].items()}
        calibrations = {k: LHCCalibration.from_dict(v) for k, v in dct["calibrations"].items()}
        return cls(circuits, calibrations)

    @classmethod
    def from_json(cls, filename):
        with open(filename) as f:
            data = json.load(f)
            return cls.from_dict(data)

    def __init__(self, circuits, calibrations=None):
        self.circuits = circuits
        if calibrations is None:
            calibrations = {}
        self.calibrations = calibrations
        for cir in self.circuits.values():
            cir._calib=self.calibrations.get(cir.calibname)


    def to_dict(self):
        return {
            "circuits": {k: v.to_dict() for k, v in self.circuits.items()},
            "calibrations": {
                k: v.to_dict() for k, v in self.calibrations.items()
            },
        }

    def to_json(self, filename):
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f)


    def __getitem__(self, key):
        return self.circuits[key]

    def __repr__(self) -> str:
        return f"<LHCCircuits {len(self.circuits)} circuits>"


class LHCCircuit:
    @staticmethod
    def get_logicalhardware_from_lsa(pcname, lsa):
        """
        Contains also Rtot,Ltot, tau
        """
        return (
            lsa._deviceService.findLogicalHardwaresByActualDeviceNames(
                [pcname]
            )
            .get(pcname)
            .toArray()[0]
        )

    @staticmethod
    def get_pcinfo_from_lsa(pcname, lsa):
        return lsa._deviceService.findPowerConverterInfo(pcname)

    @classmethod
    def from_lsa(cls, pcname, lsa):
        pcinfo = cls.get_pcinfo_from_lsa(pcname, lsa)
        lh = cls.get_logicalhardware_from_lsa(pcname, lsa)
        return cls(
            pcname=pcname,
            logicalname=lh.getName(),
            madname=lh.getMadStrengthName(),
            calibname=lh.getCalibrationName(),
            imin=pcinfo.getIMinOp(),
            imax=pcinfo.getIPNo(),
            inominal=pcinfo.getINom(),
            iultimate=pcinfo.getIUlt(),
            ipmin=pcinfo.getDidtMin(),
            ipmax=pcinfo.getDidtMax(),
            ippmax=pcinfo.getAccelerationLimit(),
            ippmin=pcinfo.getDecelerationLimit(),
        )

    @classmethod
    def from_dict(cls, dct):
        return cls(**dct)

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
        calibsign=0,
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

    def get_calib_from_lsa(self, lsa):
        pcname = self.pcname
        req = lsa._cern.lsa.domain.devices.CalibrationsRequest.byLogicalHardwareName(
            pcname
        )
        calib = lsa._deviceService.findCalibration(req)
        if calib is None:
            raise ValueError(f"Calibration not found for {pcname}")
        calib = calib.getCalibrationFunctionMap().values().toArray()[0]
        field = np.array(calib.toXArray())
        current = np.array(calib.toYArray())
        return current, field

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def __repr__(self) -> str:
        return f"<LHCCircuit {self.pcname!r}>"


class LHCCircuit2in1:
    def __init__(self, circuit1, circuit2, rc):
        self.circuit1 = circuit1
        self.circuit2 = circuit2
        self.rc = rc


class LHCCalibration:
    @classmethod
    def from_lsa_calib(cls, calib):
        name = calib.getName()
        map=calib.getCalibrationFunctionMap()
        for calibfuntype in map.keys():
            if calibfuntype.getFunctionTypeName()=="B_FIELD":
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
        dct['current']=np.array(dct['current'])
        dct['field']=np.array(dct['field'])
        return cls(**dct)

    def __init__(self,name,current,field,fieldtype):
        self.name = name
        self.current = current
        self.field = field
        self.fieldtype = fieldtype

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

    def kmin(self,pc=7e12):
        brho=pc/299792458
        return self.fmin/brho
 
    def kmax(self,pc=7e12):
        brho=pc/299792458
        return self.fmax/brho

    def to_dict(self):
        dct=self.__dict__.copy()
        dct['current']=dct['current'].tolist()
        dct['field']=dct['field'].tolist()
        return dct

    def __repr__(self) -> str:
        ilim=f"{self.imin:.0f}:{self.imax:.0f}"
        flim=f"{self.fmin:.3g}:{self.fmax:.3g}"
        return f"<LHCCalibration {self.name}, I {ilim}, {self.fieldtype} {flim}>"