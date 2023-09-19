# global imports
import argparse
import datetime
import json
import os
import numpy as np
import pandas as pd
import re
import synapseclient
from dateutil import parser
from subprocess import PIPE
from subprocess import run

# strong typing
from pandas import DataFrame
from pathlib import Path
from synapseclient.evaluation import Submission
from synapseclient import Synapse
from typing import Dict
from typing import List


def main(str_user: str, str_pass: str, bl_send: bool = False):
    docker_login(str_user=str_user, str_pass=str_pass)
    syn_client: Synapse = synapse_client_login(str_user=str_user, str_pass=str_pass)

    dt_tasks: Dict[str, int] = {"task1": 9615415, "task2": 9615416, "task3": 9615417}
    for str_task in dt_tasks:
        print("Checking {} queue.".format(str_task))
        for syn_sub, syn_status in syn_client.getSubmissionBundles(dt_tasks[str_task], status="RECEIVED"):
            syn_values: Dict = {"syn_client": syn_client, "syn_sub": syn_sub, "syn_status": syn_status}
            while True:  # allows for break clauses
                print(f"Time={datetime.datetime.now()}.")
                print(f"Checking submission {syn_sub['id']}.")

                if bl_send:
                    bl_new: bool = check_new(str_task=str_task, syn_sub=syn_sub)
                    if not bl_new:
                        set_status(syn_values=syn_values, str_reason="Old submission", str_out="closed", bl_s=bl_send)
                        break

                bl_name: bool = check_name(str_task=str_task, syn_sub=syn_sub)
                if not bl_name:
                    set_status(syn_values=syn_values, str_reason="Invalid name", str_out="rejected", bl_s=bl_send)
                    break

                bl_docker: bool = check_docker(syn_sub=syn_sub)
                if not bl_docker:
                    set_status(syn_values=syn_values, str_reason="Failed pull", str_out="rejected", bl_s=bl_send)
                    break

                bl_run: bool = check_run(syn_sub=syn_sub)
                if not bl_run:
                    set_status(syn_values=syn_values, str_reason="Failed run", str_out="rejected", bl_s=bl_send)
                    break

                bl_results: bool = check_results(str_task=str_task)
                if not bl_results:
                    set_status(syn_values=syn_values, str_reason="Incorrect output", str_out="rejected", bl_s=bl_send)
                    break

                set_status(syn_values=syn_values, str_reason="All tests have passed", str_out="accepted", bl_s=bl_send)
                break


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
    print("Looking for new submissions...")
    return syn_client


def check_new(str_task: str, syn_sub: Submission) -> bool:
    """checking if submission is new"""
    print(f"Checking if submission is new...")
    str_sub: str = syn_sub["name"].lower()
    str_json: str = f"{syn_sub['id']}.json"
    pt_history: Path = Path().absolute().joinpath("history", f"{str_sub.split('_')[-1].split(':')[0]}", f"{str_task}")
    pt_history.mkdir(parents=True, exist_ok=True)

    if not pt_history.joinpath(str_json).exists():
        with open(pt_history.joinpath(str_json), "x") as out_diary:
            json.dump(syn_sub, out_diary)
        print("New submission!")
        return True
    else:
        with open(pt_history.joinpath(str_json), "r") as in_diary:
            bl_old_submission = json.load(in_diary)
        flt_new_date: float = parser.parse(syn_sub["createdOn"]).timestamp()
        flt_old_date: float = parser.parse(bl_old_submission["createdOn"]).timestamp()
        if flt_new_date > flt_old_date:
            print("New submission!")
            return True
        else:
            return False


def check_name(str_task: str, syn_sub: Synapse) -> bool:
    """"Checking submission name is of the correct form"""
    print(f"Checking if submission name is of the form the correct form...")
    matched = re.match(f"^pitvis_{str_task}_[a-zA-Z]+:v{syn_sub['versionNumber']}", syn_sub['name'].lower(), flags=re.I)
    if bool(matched):
        print("Name is of the correct form!")
        return True
    else:
        return False


def check_docker(syn_sub: Submission) -> bool:
    """download docker"""
    print("Checking docker download...")

    process = run(
        ["docker", "pull", f"{syn_sub['dockerRepositoryName'].lower()}:v{syn_sub['versionNumber']}"],
        stdin=PIPE,
        stderr=PIPE,
        stdout=PIPE,
        universal_newlines=True,
    )

    print(f"Docker download output: {process.stdout}")
    print(f"Docker download error: {process.stderr}")
    if process.stderr:
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
            f"{syn_sub['dockerRepositoryName'].lower()}:v{syn_sub['versionNumber']}",
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
        return False
    print("Docker ran successfully!")
    return True


def check_results(str_task):
    df_results: DataFrame = pd.read_csv(Path().absolute().parent.joinpath("outputs", "01.csv"))
    ls_headers: List[str] = df_results.columns.to_list()
    ls_headers.sort()

    if not np.array_equal(df_results["int_time"], df_results["int_time"].astype(int)):
        return False

    if str_task == "task1":
        if ls_headers != ["int_step", "int_time"]:
            return False
        if not np.array_equal(df_results["int_step"], df_results["int_step"].astype(int)):
            return False
    elif str_task == "task2":
        if ls_headers != ["int_instrument1", "int_instrument2", "int_time"]:
            return False
        if not np.array_equal(df_results["int_instrument1"], df_results["int_step"].astype(int)):
            return False
        if not np.array_equal(df_results["int_instrument2"], df_results["int_step"].astype(int)):
            return False
    else:
        if ls_headers != ["int_instrument1", "int_instrument2", "int_step", "int_time"]:
            return False
        if not np.array_equal(df_results["int_step"], df_results["int_step"].astype(int)):
            return False
        if not np.array_equal(df_results["int_instrument1"], df_results["int_step"].astype(int)):
            return False
        if not np.array_equal(df_results["int_instrument2"], df_results["int_step"].astype(int)):
            return False
    print("Results are of the correct form!")
    return True


def create_message(syn_sub: Submission, str_reason: str, str_status: str, bl_valid: bool) -> str:
    if bl_valid:
        str_valid: str = "passed"
    else:
        str_valid: str = "failed"
    str_message: str = f"<br/><br/>This is an automatic message from the PitVis Evaluation Queue. \
                        <br/><br/>Thank you for the interest in our challenge. \
                        <br/><br/>Your recent submission has {str_valid}. \
                        <br/><br/>The status has been set to: {str_status}. \
                        <br/><br/>The reason for this is: {str_reason}. \
                        <br/><br/>The details of your submission are: {syn_sub}."

    return str_message


def send_message(syn_client: Synapse, str_userid: str, str_id: str, str_message: str):
    syn_client.sendMessage(
        userIds=[str_userid],
        messageSubject=f"PitVis: Your submission [{str_id}]",
        messageBody=str_message,
        contentType="text/html",
    )


def set_status(syn_values: Dict, str_reason: str, str_out: str, bl_s=False):
    syn_client = syn_values["syn_client"]
    syn_sub = syn_values["syn_sub"]
    syn_status = syn_values["syn_status"]
    if str_out == "accepted":
        print("Accepted!")
        syn_status.status = "OPEN"
        bl_valid: bool = True
    elif str_out == "closed":
        print("Closing old submission.")
        syn_status.status = "CLOSED"
        bl_valid: bool = False
    else:
        print("Rejected!")
        syn_status.status = "INVALID"
        bl_valid: bool = False

    str_message: str = create_message(
        syn_sub=syn_sub,
        str_status=syn_status.status,
        str_reason=str_reason, bl_valid=bl_valid
    )
    print(str_message)

    if bl_s:
        send_message(syn_client=syn_client, str_userid=syn_sub["userId"], str_message=str_message, str_id=syn_sub["id"])
        syn_client.store(syn_status)
    else:
        print("STATUS WILL NOT BE UPDATED!!!")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--str_user", type=str, help="synapse username", required=True)
    arg_parser.add_argument("--str_pass", type=str, help="synapse password", required=True)
    arg_parser.add_argument("--int_send", type=int, help="1 if you wish to send a message")
    args = vars(arg_parser.parse_args())
    SystemExit(main(str_user=str(args["str_user"]), str_pass=str(args["str_pass"]), bl_send=bool(args["int_send"])))
