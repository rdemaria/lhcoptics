from pathlib import Path
from collections import namedtuple

def deliver_list_str(out, output=None):
    if output is str:
        return "\n".join(out)
    elif output is list:
        return out
    elif hasattr(output, "input"):
        for ll in out:
            if len(ll) > 0 and ll[0] != "!":
                output.input(ll)
    elif hasattr(output, "writelines"):
        for ll in out:
            output.write(ll)
            output.write("\n")
    elif isinstance(output, str) or isinstance(output, Path):
        with open(output, "w") as f:
            for ll in out:
                f.write(ll)
                f.write("\n")
    elif output is None:
        print("\n".join(out))
    else:
        raise ValueError(f"Unknown output type {output}")


def git_get_current_branch(directory):
    import subprocess

    return (
        subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=directory,
            capture_output=True,
            text=True,
        )
        .stdout.strip()
        .strip()
    )


def git_set_branch(directory, branch):
    import subprocess

    subprocess.run(["git", "switch", branch], cwd=directory)



def iter_rows(table):
    Row = namedtuple("Row", table._col_names)
    for i in range(len(table)):
        yield Row(*[table._data[cn][i] for cn in table._col_names])

def print_diff_dict_float(dct1, dct2):
    allk = set(dct1.keys()) | set(dct2.keys())
    for k in sorted(allk):
        if k not in dct1:
            print(f"{k:20} {dct2[k]:15.6g} only in other")
        elif k not in dct2:
            print(f"{k:20} {dct1[k]:15.6g} only in self")
        elif dct1[k] != dct2[k]:
            print(f"{k:20} {dct1[k]:15.6g} != {dct2[k]:15.6g}")


def print_diff_dict_objs(dct1, dct2):
    allk = set(dct1.keys()) | set(dct2.keys())
    for k in sorted(allk):
        if k not in dct1:
            print(f"{k:20} only in other")
        elif k not in dct2:
            print(f"{k:20} only in self")
        else:
            dct1[k].diff(dct2[k])



