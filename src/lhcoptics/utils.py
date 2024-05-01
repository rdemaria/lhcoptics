def print_diff_dict_float(dct1,dct2):
    allk= set(dct1.keys()) | set(dct2.keys())
    for k in allk:
        if k not in dct1:
            print(f"{k:20} {dct2[k]:15.6g} only in other")
        elif k not in dct2:
            print(f"{k:20} {dct1[k]:15.6g} only in self")
        elif dct1[k] != dct2[k]:
            print(f"{k:20} {dct1[k]:15.6g} != {dct2[k]:15.6g}")

def print_diff_dict_objs(dct1,dct2):
    allk= set(dct1.keys()) | set(dct2.keys())
    for k in allk:
        if k not in dct1:
            print(f"{k:20} only in other")
        elif k not in dct2:
            print(f"{k:20} only in self")
        else:
            dct1[k].diff(dct2[k])

