import re

_lsa = None


def get_lsa():
    global _lsa
    if _lsa is None:
        import pjlsa

        _lsa = pjlsa.LSAClient()
    return _lsa



class LHCRun:
    def __init__(self, year):
        self.year = year
        self.t1 = f"{year}-01-01 00:00:00"
        self.t2 = f"{year}-12-31 23:59:59"
        # self.set_fills()

    def set_fills(self):
        lsa=get_lsa()
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

class LHCFill:
    def __init__(self, filln, beam_processes):
        self.filln = filln
        self.beam_processes = beam_processes

    def get_data(self):
        spark = get_spark()
        return spark.getLHCFillData(self.filln)

    def get_start(self):
        return self.beam_processes[0][0]

    def bp_in_fill(self, beam_process):
        for ts, bp in self.beam_processes:
            if beam_process == bp:
                return True
        else:
            return False

    def get_used_beamprocess(self, segments=False):
        out = set()
        for _, bp in self.beam_processes:
            if segments or "@" not in bp:
                out.add(bp)
        return out

    def __repr__(self):
        return f"LHCFill({self.filln})"

class LHCBeamProcess:
    def __init__(self, name):
        self.name = name

    def get_optic_table(self):
        lsa=get_lsa()
        return lsa.getOpticTable(self.name)

    def save_models(self, knobs, model_path):
        settings = knobs.get_settings()
        modelseq = self.get_modelseq(settings)
        modelseq.save_models(knobs, model_path)

    def get_modelseq(self, settings, presettings=None):
        lsa=get_lsa()
        optable = lsa.getOpticTable(self.name)

        models = []
        index = []
        for op in optable:
            models.append(op.name)
            index.append(op.time)


        trims = {}
        for pp in settings:
            try:
                print(f"Extracting last trim {pp} for {self.name}")
                trims[pp] = lsa.getLastTrim(pp, self.name, part="target")
            except Exception:
                print(f"Error extracting last trim {pp} for {self.name}")

        for pp, trim in trims.items():
            indexes, values = trim.data
            #modelseq.apply_trim(pp, indexes, values)

        #self.modelseq = modelseq

        #return modelseq

    def get_trims(self, params, lhcrun=None):
        lsa=get_lsa()
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

    def get_model_history(self, params, lhcrun, presettings):
        lsa=get_lsa()
        optable = lsa.getOpticTable(self.name)

        models = []
        index = []
        for op in optable:
            predict = {"opname": op.name}
            predict.update(presettings)
            models.append(predict)
            index.append(op.time)

        models = (index, models)

        trims = self.get_trims(params, lhcrun)

        for filln, fill in lhcrun.fills.items():
            if fill.bp_in_fill(self.name):
                trims.append([fill.get_start(), "FILLN", filln])

        trims.sort()

        out = {}

        cfill = 0
        for ts, setting, val in trims:
            print(cfill, ts, setting, val)
            if setting == "FILLN":
                out[cfill] = models.copy()
                cfill = val
            else:
                models.apply_trim(setting, val[0], val[1])
        out[cfill] = models.copy()
        return out

    def __repr__(self):
        return f"LHCBeamProcess({self.name!r})"
