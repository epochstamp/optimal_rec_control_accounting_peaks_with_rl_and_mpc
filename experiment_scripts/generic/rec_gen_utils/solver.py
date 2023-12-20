import argparse
import json
import lzma
import subprocess
import sys
import tarfile
import tempfile
import os.path
import shutil
from pathlib import Path


def solve(dirpath:Path, hashname:str):
    filename = f"{hashname}.instance.tar.xz"
    gboml_main = "/Users/gderval/PycharmProjects/gboml/src/main.py"
    fullpath = dirpath / filename
    
    full_solution_path = dirpath / f"{hashname}.solution.json"
    if not os.path.exists(full_solution_path):
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdirname = Path(tmpdirname)
            with tarfile.open(fullpath) as f:
                f.extractall(tmpdirname)

            command = [sys.executable, gboml_main, *json.load(open(tmpdirname / hashname / "parameters.json")), f"--gurobi", "--json", "--output", str(tmpdirname / hashname / "out"), str(tmpdirname / hashname / "main.gboml")]
            print("RUNNING", command)
            subprocess.run(command)
            shutil.copy(tmpdirname / hashname / "out.json", full_solution_path)
    return json.load(open(full_solution_path))
    
def gen_and_solve(rec, dirpath:Path=Path("tmp"), **kwargs):
    from rec_generator import gen_instance
    return solve(dirpath, gen_instance(dirpath, rec, **kwargs))