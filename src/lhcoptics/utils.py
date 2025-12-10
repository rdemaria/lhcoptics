from pathlib import Path
from collections import namedtuple
import subprocess

import requests
import json
import time
from datetime import datetime, timezone
import xml.etree.ElementTree as ET
import numpy as np

import ruamel.yaml

yaml=ruamel.yaml.YAML()
yaml.indent(mapping=4, sequence=2, offset=2)

def get_yaml():
    return yaml

def string_to_unixtime(timestr: str, utc: bool = False) -> int:
    """
    Convert a time string "%Y-%m-%d %H:%M:%S" to Unix time.
    If utc=True, interpret the string as UTC else local time;
    """
    dt = datetime.strptime(timestr, "%Y-%m-%d %H:%M:%S")
    if utc:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone()
    return int(dt.timestamp())

def unixtime_to_string(unixtime: int, utc: bool = False) -> str:
    """
    Convert Unix time to a time string "%Y-%m-%d %H:%M:%S".
    If utc=True, interpret the string as UTC else local time;
    """
    if utc:
        dt = datetime.fromtimestamp(unixtime, tz=timezone.utc)
    else:
        dt = datetime.fromtimestamp(unixtime)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

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


def print_diff_dict_float(dct1, dct2, keys='union'):
    if keys == 'union':
        allk = set(dct1.keys()) | set(dct2.keys())
    elif keys == 'intersection':
        allk = set(dct1.keys()) & set(dct2.keys())
    else:
        allk = set(keys)
    for k in sorted(allk):
        if k not in dct1:
            print(f"{k:20} {dct2[k]:15.6g} only in other")
        elif k not in dct2:
            print(f"{k:20} {dct1[k]:15.6g} only in self")
        elif dct1[k] != dct2[k]:
            print(f"{k:20} {dct1[k]:15.6g} != {dct2[k]:15.6g}")




def print_diff_dict_objs(dct1, dct2, keys='union'):
    if keys == 'union':
        allk = set(dct1.keys()) | set(dct2.keys())
    elif keys == 'intersection':
        allk = set(dct1.keys()) & set(dct2.keys())
    else:
        allk = set(keys)
    for k in sorted(allk):
        if k not in dct1:
            print(f"{k:20} only in other")
        elif k not in dct2:
            print(f"{k:20} only in self")
        else:
            dct1[k].diff(dct2[k])


def gitlab_get_branches_and_tags(
    project_id=76507,
    private_token=None,
    gitlab_url="https://gitlab.cern.ch",
    timeout=0.5,
):
    """
    Retrieves the branches and tags of a GitLab repository using the GitLab API.

    Args:
        project_id (int): The ID of the GitLab project.
        private_token (str, optional): Your GitLab personal access token.
            Defaults to None. If None, assumes the repository is public.
        gitlab_url (str, optional): The base URL of your GitLab instance.

    Returns:
        dict: A dictionary containing the lists of branches and tags.
              Returns None and prints an error message if the request fails.
              Example:
              {
                  "branches": ["main", "develop", "feature/new-feature"],
                  "tags": ["v1.0.0", "v1.1.0", "v2.0.0"]
              }
    """
    headers = {}
    if private_token:
        headers = {"PRIVATE-TOKEN": private_token}
    branches_url = (
        f"{gitlab_url}/api/v4/projects/{project_id}/repository/branches"
    )
    tags_url = f"{gitlab_url}/api/v4/projects/{project_id}/repository/tags"

    try:
        # Get branches
        branches_response = requests.get(
            branches_url, headers=headers, timeout=timeout
        )
        branches_response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        branches_data = branches_response.json()
        branches = dict(
            [(branch["name"], branch["commit"]["id"]) for branch in branches_data]
        )

        # Get tags
        tags_response = requests.get(
            tags_url, headers=headers, timeout=timeout
        )
        tags_response.raise_for_status()  # Raise HTTPError for bad responses
        tags_data = tags_response.json()
        tags = dict(
            [(tag["name"], tag["commit"]["id"]) for tag in tags_data]
        )

        return {"branches": branches, "tags": tags}

    except json.JSONDecodeError:
        print("Error: Failed to decode JSON response from GitLab.")
    except KeyError as e:
        print(
            f"Error: Unexpected data structure from GitLab API.  Missing key: {e}"
        )
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

    return None


def file_one_day_old(path):
    """
    Check if a file is older than one day.
    """
    return path.stat().st_mtime > time.time() - 24 * 3600

def file_expired(path, max_age):
    """
    Check if a file is older than max_age seconds.
    """
    return path.stat().st_mtime > time.time() - max_age


def read_yaml(filename):
    return yaml.load(open(filename, "r"))

def write_yaml(data, filename):
    yaml.dump(data, open(filename, "w"))


def read_json(filename):
    with open(filename, "r") as fid:
        return json.load(fid)

def write_json(data, filename):
    with open(filename, "w") as fid:
        json.dump(data, fid, indent=4)

def git_get_current_commit(directory):
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=directory,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to get current commit: {result.stderr.strip()}"
        )
    return result.stdout.strip()

def git_pull(directory):
    result = subprocess.run(
        ["git", "pull"],
        cwd=directory,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to pull branch: {result.stderr.strip()}"
        )
    return result.stdout.strip()

def git_get_branch_name(directory):
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=directory,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to get branch name: {result.stderr.strip()}"
        )
    return result.stdout.strip()

def git_clone_repo(repo_url, target_directory, branch=None):
    """
    Clone a Git repository.
    """
    cmds=[ "git", "clone", repo_url, target_directory]
    if branch:
        cmds.extend(["--branch", branch])
    result = subprocess.run(
        cmds,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to clone repository: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def git_status(directory,*args):
    result = subprocess.run(
        ["git", "status"]+list(args),
        cwd=directory,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to get git status: {result.stderr.strip()}"
        )
    return result.stdout.strip()

def git_commit(directory, message,*args):
    result = subprocess.run(
        ["git", "commit", "-m", message]+list(args),
        cwd=directory,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to commit changes: {result.stderr.strip()}"
        )
    return result.stdout.strip()

def git_push(directory,*args):
    """
    Push changes to the remote repository.
    """
    result = subprocess.run(
        ["git", "push"]+list(args),
        cwd=directory,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to push changes: {result.stderr.strip()}"
        )
    return result.stdout.strip()

def git_add(directory, *args):
    """
    Add files to the staging area.
    """
    result = subprocess.run(
        ["git", "add"]+list(args),
        cwd=directory,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to add files: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def read_url(url, timeout=0.1):
    """
    Read content from a URL.
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()  # Raise an error for bad responses
        return response.text
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to read from URL {url}: {e}")


def xmltodict(filename):
    root = ET.parse(filename).getroot()
    return etree_to_dict(root)

def etree_to_dict(elem):
    """Turn an ElementTree into a dict.
    """
    # convert an Element and its children into a dict
    d = {elem.tag: {} if elem.attrib else None}
    children = list(elem)
    if children:
        dd = {}
        for dc in map(etree_to_dict, children):
            for k, v in dc.items():
                if k in dd:
                    # group multiple children with same tag into a list
                    if not isinstance(dd[k], list):
                        dd[k] = [dd[k]]
                    dd[k].append(v)
                else:
                    dd[k] = v
        d = {elem.tag: dd}
    if elem.attrib:
        d[elem.tag].update(('@' + k, v) for k, v in elem.attrib.items())
    if elem.text and elem.text.strip():
        text = elem.text.strip()
        if children or elem.attrib:
            d[elem.tag]['#text'] = text
        else:
            d[elem.tag] = text
    return d


def read_knob_structure(filename_or_path_or_dict):
    if isinstance(filename_or_path_or_dict, dict):
        return filename_or_path_or_dict
    filename = str(filename_or_path_or_dict)
    if filename.endswith(".yaml"):
        return read_yaml(filename)
    elif filename.endswith(".json"):
        return read_json(filename)
    else:
        raise ValueError(f"Unknown knob_structure file format: {filename}")


def find_comparable_values(a, b, tol=1e-6):
    """
    Return the indices of values that are comparable between two lists a and b ascending values, not necessarily of the same length.

    Args:
        a (list of float): First list of values.
        b (list of float): Second list of values.
        tol (float): Tolerance for comparison.
    Returns:
        list of int: Indices of comparable values.
    """
    a=sorted(a)
    b=sorted(b)
    ia=0
    ib=0
    indices_a=[]
    indices_b=[]
    while ia < len(a) and ib < len(b):
        if abs(a[ia]-b[ib]) < tol:
            indices_a.append(ia)
            indices_b.append(ib)
            ia += 1
            ib += 1
        elif a[ia] < b[ib]:
            ia += 1
        else:
            ib += 1
    return np.array(indices_a), np.array(indices_b)




