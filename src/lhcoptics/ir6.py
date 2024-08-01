import numpy as np

from .irs import LHCIR

def beta_dump(bet,alf):
    al_dump=761
    return bet-2*alf*al_dump+al_dump**2*(1+alf**2)/bet

def tcdq_mingap(bet,dx):
    nsig = 10.1
    dpoverp = 2e-4
    emitx=2.5e-6/(7000/0.9382720814)
    maxorbitdrift = 0.6e-3
    return nsig*np.sqrt(emitx*bet)-3e-4-abs(dx*dpoverp)-maxorbitdrift

def tcdq_gap(bet):
    nsigma=10.1
    emitx=2.5e-6/(7000/0.9382720814)
    return nsigma*np.sqrt(emitx*bet)


class LHCIR6(LHCIR):
    name = "ir6"


    def get_params_from_twiss(self, tw1, tw2):
        params = LHCIR.get_params_from_twiss(self, tw1, tw2)

        muxmkdb1=tw1["mux","mkd.h5l6.b1"]
        params["dmuxkickb1_tcsg"]=tw1["mux","tcsp.a4r6.b1"]- muxmkdb1
        dmuxb1=np.array([tw1["mux",f"tcdqa.{abc}4r6.b1"] for abc in "abc"])-muxmkdb1
        for i,abc in enumerate("abc"):
            params[f"dmuxkickb1_tcdq{abc}"]=dmuxb1[i]
        params["dmuxkickb1"]=abs(dmuxb1-0.25).max()

        params["dxq5l6b1"]=tw1["dx","mqy.5l6.b1"]
        params["dxq4r6b1"]=tw1["dx","mqy.4r6.b1"]
        params["dxtcdqb1"]=tw1["dx","tcdqa.a4r6.b1"]
        params["betxtcdqb1"]=tw1["betx","tcdqa.a4r6.b1"]
        params["betytcdqb1"]=tw1["bety","tcdqa.a4r6.b1"]
        params["betxtcdsb1"]=tw1["betx","tcdsa.4l6.b1"]
        params["betytcdsb1"]=tw1["bety","tcdsa.4l6.b1"]
        params["betxtcsgb1"]=tw1["betx","tcsp.a4r6.b1"]
        params["betytcsgb1"]=tw1["bety","tcsp.a4r6.b1"]
        params["betxmkdb1"]=tw1["betx","mkd.h5l6.b1"]
        params["betymkdb1"]=tw1["bety","mkd.h5l6.b1"]
        params["bxdumpb1"]=beta_dump(params["betxip6b1"],params["alfxip6b1"])
        params["bydumpb1"]=beta_dump(params["betyip6b1"],params["alfyip6b1"])

        params["dmuxkickb1_bds"]=tw1["mux","mkd.o5l6.b1"]
        params["dmuxkickb1_bdsa"]=tw1["mux","mkd.a5l6.b1"]
        params["dmuxkickb1_eds"]=tw1["mux","e.ds.r6.b1"]- tw1["mux","mkd.o5l6.b1"]
        params["tcdqmingapb1"]=tcdq_mingap(params["betxtcdqb1"],params["dxtcdqb1"])
        params["tcdqgapb1"] =tcdq_gap(params["betxtcdqb1"])

        muxmkdb2=tw2["mux","mkd.h5r6.b2"]
        params["dmuxkickb2_tcsg"]=tw2["mux","tcsp.a4l6.b1"]- muxmkdb2
        dmuxb2=np.array([tw2["mux",f"tcdqa.{abc}4l6.b2"] for abc in "abc"])-muxmkdb2
        for i,abc in enumerate("abc"):
            params[f"dmuxkickb2_tcdq{abc}"]=dmuxb2[i]
        params["dmuxkickb2"]=abs(dmuxb2-0.25).max()

        params["dxq5r6b2"]=tw2["dx","mqy.5r6.b2"]
        params["dxq4l6b2"]=tw2["dx","mqy.4l6.b2"]
        params["dxtcdqb2"]=tw2["dx","tcdqa.a4l6.b1"]
        params["betxtcdqb2"]=tw2["betx","tcdqa.a4l6.b2"]
        params["betytcdqb2"]=tw2["bety","tcdqa.a4l6.b2"]
        params["betxtcdsb2"]=tw2["betx","tcdsa.4r6.b2"]
        params["betytcdsb2"]=tw2["bety","tcdsa.4r6.b2"]
        params["betxtcsgb2"]=tw2["betx","tcsp.a4l6.b2"]
        params["betytcsgb2"]=tw2["bety","tcsp.a4l6.b2"]
        params["betxmkdb2"]=tw2["betx","mkd.h5r6.b2"]
        params["betymkdb2"]=tw2["bety","mkd.h5r6.b2"]
        params["bxdumpb2"]=beta_dump(params["betxip6b2"],params["alfxip6b2"])
        params["bydumpb2"]=beta_dump(params["betyip6b2"],params["alfyip6b2"])

        params["dmuxkickb2_bds"]=tw1["mux","mkd.o5l6.b2"]
        params["dmuxkickb2_bdsa"]=tw1["mux","mkd.a5l6.b2"]
        params["dmuxkickb2_eds"]=tw1["mux","e.ds.r6.b2"]- tw1["mux","mkd.o5l6.b2"]
        params["tcdqmingapb2"]=tcdq_mingap(params["betxtcdqb2"],params["dxtcdqb2"])
        params["tcdqgapb2"] =tcdq_gap(params["betxtcdqb2"])
        return params
    
    
    def set_init(self):
        if self.parent.is_ats():
            self.init_left={
                1: self.parent.ir5.get_init_ats_right(1),
                2: self.parent.ir5.get_init_ats_right(2),
            }
        else:
            self.init_left = {
               1: self.arc_left.get_init_right(1),
               2: self.arc_left.get_init_right(2),
        }        
    
        self.init_right = {
                1: self.arc_right.get_init_left(1),
                2: self.arc_right.get_init_left(2),
            }