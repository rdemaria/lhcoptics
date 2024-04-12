import json


def madname_from_pcname(pc):
    name = ".".join(pc.split(".")[2:]).lower()
    if name.startswith("rcb"):
        return "a" + name[1:]
    else:
        return "k" + name[1:]

def get_calib_mad(lsa, madname):
    print(madname)
    pcname = lsa.findPCNameByMadStrength(madname)
    if pcname is not None:
        if pcname.split(".")[0] in ["RQD", "RQF", "RB"]:
            pcname = pcname + "B1"
        cal = lsa.getCalibration(pcname)
        if cal is not None:
            return [cal.field, cal.current]

class LHCCircuits:
    @classmethod
    def from_lsa(cls):
        import pjlsa
        lsa=pjlsa.LSAClient()

    @classmethod
    def from_json(cls, filename):
        pass

    def __init__(self, circuits):
        self.circuits = circuits

    def to_json(self, filename):
        pass


class LHCCircuit:
    def __init__(
        self,
        name,
        madname,
        calib,
        r1,
        l1,
        c1,
        imax,
        imin,
        vmax,
        vmin,
        ipmax,
        ipmin,
        vpmax,
        vpmin,
        polarity,
    ):
        self.name = name
        self.madname = madname
        self.calib = calib
        self.r1 = r1
        self.l1 = l1
        self.c1 = c1
        self.imax = imax
        self.imin = imin
        self.vmax = vmax
        self.vmin = vmin
        self.ipmax = ipmax
        self.ipmin = ipmin
        self.vpmax = vpmax
        self.vpmin = vpmin
        self.polarity = polarity


class LHCCircuit2in1:
    def __init__(self, circuit1, circuit2, rc):
        self.circuit1 = circuit1
        self.circuit2 = circuit2
        self.rc = rc
