# global imports
import argparse
import os
import numpy as np
import pandas as pd
import re
import synapseclient
from subprocess import PIPE
from subprocess import run

# strong typing
from pandas import DataFrame
from pathlib import Path
from synapseclient.evaluation import Submission
from synapseclient import Synapse
from typing import Dict
from typing import List


def main(str_user: str, str_pass: str, int_id: int):
    # docker_login(str_user=str_user, str_pass=str_pass)
    syn_client: Synapse = synapse_client_login(str_user=str_user, str_pass=str_pass)

    dt_tasks: Dict[str, int] = {"task1": 9615415, "task2": 9615416, "task3": 9615417}
    print(f"Looking for submission id={int_id} with status='RECEIVED'.")
    for str_task in dt_tasks:
        for syn_sub, syn_status in syn_client.getSubmissionBundles(dt_tasks[str_task], status="RECEIVED"):
            if int(syn_sub["id"]) == int_id:
                print(f"Found submission {syn_sub['id']}, checking...")
                check_name(str_task=str_task, syn_sub=syn_sub)
                check_docker(syn_sub=syn_sub)
                check_run(syn_sub=syn_sub)
                check_results(str_task=str_task)
        print("Run completed.")


def docker_login(str_user: str, str_pass: str):
    process = run(
        ["docker", "login", "docker.synapse.org", f"-u {str_user}", f"-p {str_pass}"],
        stdin=PIPE,
        stderr=PIPE,
        stdout=PIPE,
        universal_newlines=True,
    )
    print(f"Docker download output: {process.stdout}")
    print(f"Docker download error: {process.stderr}")


def synapse_client_login(str_user: str, str_pass: str) -> Synapse:
    syn_client: Synapse = synapseclient.Synapse()
    syn_client.login(str_user, str_pass)
    return syn_client


def check_name(str_task: str, syn_sub: Synapse) -> bool:
    """"Checking submission name is of the correct form"""
    print(f"Checking if submission name is of the form the correct form...")
    pitvis, task, team_name_version = syn_sub['name'].lower().split("_")
    team, version = team_name_version.split(":")
    team_match = re.match("^[a-zA-Z]+", team, flags=re.I)
    version_match = re.match("^v[0-9]+", version, flags=re.I)
    matched = (pitvis == "pitvis" and task == str_task and bool(team_match) and bool(version_match))
    if bool(matched):
        print("Name is of the correct form!")
        return True
    else:
        print("Name is not of the correct form!")
        return False


def check_docker(syn_sub: Submission) -> bool:
    """download docker"""
    print("Checking docker download...")

    process = run(
        ["docker", "pull", f"{syn_sub['dockerRepositoryName'].lower()}:{syn_sub['name'].lower().split(':')[-1]}"],
        stdin=PIPE,
        stderr=PIPE,
        stdout=PIPE,
        universal_newlines=True,
    )

    print(f"Docker download output: {process.stdout}")
    print(f"Docker download error: {process.stderr}")
    if process.stderr:
        print("Docker download failed - please check docker error above.")
        return False
    print("Docker downloaded!")
    return True


def check_run(syn_sub: Submission) -> bool:
    """run test script"""
    print("Checking docker run...")
    pt_home: Path = Path().absolute()
    pt_csv: Path = pt_home.joinpath('data', 'outputs', '01.csv')
    if pt_csv.exists():
        os.remove(pt_csv)
    df_output = pd.DataFrame()
    df_output.to_csv(pt_csv, index=False)
    process = run(
        [
            "docker",
            "run",
            "--gpus=all",
            "-t",
            "--rm",
            f"-v={pt_home.parent.joinpath('inputs')}:{pt_home.joinpath('data', 'inputs')}",
            f"-v={pt_home.parent.joinpath('outputs')}:{pt_home.joinpath('data', 'outputs')}",
            f"{syn_sub['dockerRepositoryName'].lower()}:{syn_sub['name'].lower().split(':')[-1]}",
            f"{pt_home.joinpath('data', 'inputs', '01')}",
            f"{pt_csv}",
        ],
        stdin=PIPE,
        stderr=PIPE,
        stdout=PIPE,
        universal_newlines=True,
    )

    print(f"Docker run output: {process.stdout}")
    print(f"Docker run error: {process.stderr}")
    if process.stderr:
        print("Docker run failed - please check docker error above.")
        return False
    print("Docker ran successfully!")
    return True


def check_results(str_task):
    df_results: DataFrame = pd.read_csv(Path().absolute().parent.joinpath("outputs", "01.csv"))
    ls_headers: List[str] = df_results.columns.to_list()
    ls_headers.sort()

    if not np.array_equal(df_results["int_time"], df_results["int_time"].astype(int)):
        print("Results are in the incorrect form: in_time values are non-integers.")
        return False

    if str_task == "task1":
        if ls_headers != ["int_step", "int_time"]:
            print("Results are in the incorrect form: header values are not ['int_step', 'int_time']")
            return False
        if not np.array_equal(df_results["int_step"], df_results["int_step"].astype(int)):
            print("Results are in the incorrect form: int_step values are non-integers.")
            return False
    elif str_task == "task2":
        if ls_headers != ["int_instrument1", "int_instrument2", "int_time"]:
            print("Results are in the incorrect form: header values are not "
                  "['int_instrument1', 'int_instrument2', 'int_time']")
            return False
        if not np.array_equal(df_results["int_instrument1"], df_results["int_instrument1"].astype(int)):
            print("Results are in the incorrect form: int_instrument1 values are non-integers.")
            return False
        if not np.array_equal(df_results["int_instrument2"], df_results["int_instrument2"].astype(int)):
            print("Results are in the incorrect form: int_instrument2 values are non-integers.")
            return False
    else:
        if ls_headers != ["int_instrument1", "int_instrument2", "int_step", "int_time"]:
            print("Results are in the incorrect form: header values are not "
                  "['int_time', 'int_step', 'int_instrument1', 'int_instrument2']")
            return False
        if not np.array_equal(df_results["int_step"], df_results["int_step"].astype(int)):
            print("Results are in the incorrect form: int_step values are non-integers.")
            return False
        if not np.array_equal(df_results["int_instrument1"], df_results["int_instrument1"].astype(int)):
            print("Results are in the incorrect form: int_instrument1 values are non-integers.")
            return False
        if not np.array_equal(df_results["int_instrument2"], df_results["int_instrument2"].astype(int)):
            print("Results are in the incorrect form: int_instrument2 values are non-integers.")
            return False
    print("Results are of the correct form!")
    return True


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--str_user", type=str, help="synapse username", required=True)
    arg_parser.add_argument("--str_pass", type=str, help="synapse password", required=True)
    arg_parser.add_argument("--int_id", type=int, help="id number, check tables if unsure", required=True)
    args = vars(arg_parser.parse_args())
    SystemExit(main(str_user=str(args["str_user"]), str_pass=str(args["str_pass"]), int_id=int(args["int_id"])))
