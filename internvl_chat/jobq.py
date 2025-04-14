#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2023-12-28 11:37:41
# @Author  : Shangyu.Xing (starreeze@foxmail.com)
"""
A simple gpu job queue. This will maintain a queue of jobs to be executed on gpus, and will run them sequentially,
always beginning from the job with highest orders. Among the same order,
those jobs with higher gpu demands will be prioritized. If there is no free GPU available,
it wait until there are enough free GPUs to execute the job.
It will automatically set the environment variable `CUDA_AVAILABLE_DEVICES` to the free GPU IDs when run the job.
If `-s` is set, the program will automatically split the job into multiple jobs with 1 gpu for each.
Make sure that your program accepts `start_pos` and `end_pos`, and your output filename is dependent on them.
The output of the jobs will be both logged to the console and a file.

It support the following operations:
1. add: `python jobq.py add [job_cmd] [-n n_gpus_required] [-o job_order] [-s total_samples]`.
    Add a job to the queue with command to be executed with os.system(job_cmd).
2. list: `python jobq.py ls`. List all jobs in the queue, including an ID
    (which is a unique integer, starting from 0 and adds 1 for each job),
    the datetime upon creation, job command, and the gpu requirement.
3. delete: `python jobq.py del [job_id]`. Delete a job from the queue by its ID.
4. start: `python jobq.py start [-g gpus_to_be_used] [-i wait_time_before_using_a_free_gpu] [-j jobs included]
    [-f pool_frequency_for_checking_gpu_availability]`.
    Start a daemon to run the jobs in the queue. The daemon should run indefinitely until stopped manually.
    Note that even after the daemon is started, you can still modify the job queue,
    such as adding or deleting jobs to the queue.
"""

from __future__ import annotations

import datetime
import glob
import json
import logging
import os
import subprocess
import sys
import time
from argparse import ArgumentParser
from copy import deepcopy
from multiprocessing import Process
from typing import Any, Iterable

from natsort import natsorted
from rich.logging import RichHandler

# package info
__version__ = "0.1.6"
__author__ = "Starreeze"
__license__ = "GPLv3"
__url__ = "https://github.com/starreeze/server-tools"

logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])
_logger = logging.getLogger("rich")

# default path
output_dir = os.path.expanduser("/home/nfs05/xiangch/logs/jq_output")
status_path = os.path.expanduser("/home/nfs05/xiangch/logs/jq_status.json")


class GPUManager:
    def __init__(self, gpus: Iterable[int], max_used_mem_mb: int) -> None:
        self.gpu_avail: set[int] = set()
        self.gpu_pending: set[int] = set(gpus)
        self.gpu_inuse: dict[int, set[int]] = {}  # job_id -> {gpu_id}
        self.gpu_free_since: dict[int, datetime.datetime] = {}  # gpu_id -> timestamp
        self.max_used_mem_mb = max_used_mem_mb

    def update_gpu_status(self, interval: int) -> None:
        if not self.gpu_pending:
            _logger.debug(
                f"gpu available: {self.gpu_avail}, pending: {self.gpu_pending}, inuse: {self.gpu_inuse}"
            )
            return
        try:
            gpustat_output = subprocess.check_output(["gpustat", "--json"], text=True)
        except subprocess.CalledProcessError:
            _logger.error("Error running gpustat.")
            return
        gpu_data = json.loads(gpustat_output)
        current_time = datetime.datetime.now()
        for gpu_id, gpu in enumerate(gpu_data["gpus"]):
            if gpu["memory.used"] < self.max_used_mem_mb and gpu["utilization.gpu"] == 0:
                if gpu_id in self.gpu_pending:
                    _logger.info(f"find free gpu: {gpu_id}")
                    if interval == 0:
                        self.gpu_avail.add(gpu_id)
                        self.gpu_pending.remove(gpu_id)
                    elif gpu_id not in self.gpu_free_since:
                        self.gpu_free_since[gpu_id] = current_time
                    elif (current_time - self.gpu_free_since[gpu_id]).total_seconds() >= 60 * interval:  # type: ignore
                        # logger.info(f"starting job after waited...")
                        self.gpu_avail.add(gpu_id)
                        self.gpu_pending.remove(gpu_id)
            else:
                self.gpu_free_since.pop(gpu_id, None)
                if gpu_id in self.gpu_avail:
                    self.gpu_avail.remove(gpu_id)
                    self.gpu_pending.add(gpu_id)
        _logger.debug(
            f"gpu available: {self.gpu_avail}, pending: {self.gpu_pending}, inuse: {self.gpu_inuse}"
        )

    def allocate(self, job_id: int, ngpu: int) -> list[int] | None:
        "Allocate gpus according to number of gpus needed. Return None if no available."
        if len(self.gpu_avail) < ngpu:
            return None
        allocated = [self.gpu_avail.pop() for _ in range(ngpu)]
        self.gpu_inuse[job_id] = set(allocated)
        return allocated

    def free(self, job_id: int) -> set[int]:
        freed = self.gpu_inuse.pop(job_id)
        self.gpu_avail |= freed
        return freed


class JobQueue:
    def __init__(self, filename=status_path):
        self.filename = filename
        self.job_id_counter = 0
        self.running_jobs: dict[int, Process] = {}
        self.gpu_manager: GPUManager

    def load(self) -> list[dict[str, Any]]:
        if os.path.exists(self.filename):
            with open(self.filename, "r") as file:
                return json.load(file)
        return []

    def save(self, job_queue: list[dict[str, Any]]):
        job_queue = sorted(job_queue, key=lambda x: (x["order"], -x["ngpu"]))
        with open(self.filename, "w") as file:
            json.dump(job_queue, file)

    @staticmethod
    def add_single(args, job_queue: list[dict[str, Any]]):
        job_id = max([job["id"] for job in job_queue], default=-1) + 1 if args.id == -1 else args.id
        current_dir = os.getcwd()
        job = {
            "id": job_id,
            "datetime": datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
            "command": args.cmd,
            "ngpu": args.ngpu,
            "order": args.order,
            "dir": current_dir,
        }
        job_queue.append(job)
        _logger.info(f"Added job {job_id}.")

    def add(self, args):
        job_queue = self.load()
        if args.separate_total:
            chunk_size = (args.separate_total - 1 + args.ngpu) // args.ngpu
            for i in range(args.ngpu):
                separate_args = deepcopy(args)
                add_args = [
                    "--start_pos",
                    str(i * chunk_size),
                    "--end_pos",
                    str(min((i + 1) * chunk_size, args.separate_total)),
                ]
                separate_args.cmd.extend(add_args)
                separate_args.ngpu = 1
                self.add_single(separate_args, job_queue)
        else:
            self.add_single(args, job_queue)
        self.save(job_queue)

    def list(self):
        job_queue = self.load()
        for job in job_queue:
            print(
                f"{job['id']: 03d} ({job['datetime']}): {job['command']}, "
                f"gpus required: {job['ngpu']}, job order: {job['order']}"
            )

    def delete(self, job_id):
        job_queue = self.load()
        job_queue = [job for job in job_queue if job["id"] != job_id]
        self.save(job_queue)
        print(f"Deleted job {job_id}.")

    def clear(self):
        self.save([])
        print("Cleared all jobs.")

    def execute(self, job: dict[str, Any], gpus: list[int]):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus)) if gpus else "-1"
        output_file = os.path.join(output_dir, f"job_{job['id']}_output.txt")
        os.chdir(job["dir"])  # chdir in subprocess will not change the dir of main process
        try:
            with open(output_file, "w") as file, subprocess.Popen(
                job["command"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
                text=True,
            ) as proc:
                while True:
                    char = proc.stdout.read(1)  # type: ignore
                    if not char:
                        break
                    print(char, end="", flush=True)
                    file.write(char)
                if proc.wait() != 0:
                    raise subprocess.CalledProcessError(proc.returncode, job["command"])

            _logger.info(f"Job {job['id']} executed successfully.")
        except subprocess.CalledProcessError as e:
            _logger.error(f"Job {job['id']} failed to execute. Error: {e}")

    def daemon(self, args):
        while True:
            # check complete jobs and release gpus
            for job_id, process in self.running_jobs.copy().items():
                if not process.is_alive():
                    freed = self.gpu_manager.free(job_id)
                    _logger.info(f"job {job_id} finished, gpu {freed} is now available.")
                    process.join()
                    self.running_jobs.pop(job_id)

            job_queue = self.load()
            if len(job_queue) == 0:
                time.sleep(args.frequency)
                continue
            if len(args.jobs) == 0:
                args.jobs = [job["id"] for job in job_queue]

            # start new jobs
            self.gpu_manager.update_gpu_status(args.interval)
            started_job_idx = []
            for i, job in enumerate(job_queue):
                if job["id"] not in args.jobs:
                    continue
                gpus = self.gpu_manager.allocate(job["id"], job["ngpu"])
                if gpus is None:
                    continue
                _logger.info(f"starting job {job['id']} on gpu {gpus}")
                process = Process(target=self.execute, args=(job, gpus))
                self.running_jobs[job["id"]] = process
                process.start()
                started_job_idx.append(i)
            if started_job_idx:
                started_job_idx.reverse()
                for i in started_job_idx:
                    del job_queue[i]
                self.save(job_queue)
            else:
                _logger.info(f"No available gpu.")

            time.sleep(args.frequency)

    def start(self, args):
        os.makedirs(output_dir, exist_ok=True)
        self.gpu_manager = GPUManager(args.gpus, args.max_used_mem_mb)
        _logger.info("Daemon started.")
        self.daemon(args)


class ArgParser:
    def __init__(self, args: list[str] | None):
        self.args = args

    def parse_start(self):
        parser = ArgumentParser()
        parser.add_argument("--gpus", "-g", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
        parser.add_argument(
            "--jobs", "-j", type=int, nargs="+", default=[], help="select which jobs to include"
        )
        parser.add_argument("--interval", "-i", type=int, default=0, help="wait time before using a gpu")
        parser.add_argument("--frequency", "-f", type=int, default=30, help="frequency to check for gpu")
        parser.add_argument(
            "--max_used_mem_mb", "-m", type=int, default=1000, help="if memory usage > this, gpu is not used"
        )
        return parser.parse_args(self.args)

    def parse_add(self):
        parser = ArgumentParser()
        parser.add_argument("cmd", nargs="+", help="the full command for this job")
        parser.add_argument("--ngpu", "-n", type=int, default=1, help="number of gpus needed for this job")
        parser.add_argument(
            "--separate_total",
            "-s",
            type=int,
            default=0,
            help="total size of samples. If >0, run the job in separate processes, each allocating a gpu",
        )
        parser.add_argument("--order", "-o", type=int, default=0, help="the order of this job")
        parser.add_argument("--id", "-i", type=int, default=-1, help="the id of this job")
        known_args, unknown_args = parser.parse_known_args(self.args)
        known_args.cmd.extend(unknown_args)
        return known_args

    def parse_merge(self):
        parser = ArgumentParser()
        parser.add_argument("--input", "-i", type=str, nargs="+", default=[], help="input files to be merged")
        parser.add_argument("--output", "-o", type=str, required=True, help="the output file name")
        args = parser.parse_args(self.args)
        input_files = natsorted(sum([glob.glob(input_file) for input_file in args.input], []))
        args.input = input_files
        return args


def print_help(invalid: bool):
    if invalid:
        print("Invalid arguments.")
    print("Usage:")
    print(__doc__)


def merge(args):
    with open(args.output, "w") as file:
        for input_file in args.input:
            with open(input_file, "r") as f:
                content = f.read()
                # content = content.strip("\n") + "\n"
                file.write(content)


def main():
    if len(sys.argv) < 2:
        print_help(invalid=True)
        return
    queue = JobQueue()
    op, args = sys.argv[1], sys.argv[2:]
    parser = ArgParser(args)
    if op == "add" and args:
        queue.add(parser.parse_add())
    elif op == "ls":
        queue.list()
    elif op == "del" and args:
        for id in args:
            if id == "all":
                queue.clear()
                break
            else:
                queue.delete(int(id))
    elif op == "start":
        queue.start(parser.parse_start())
    elif op == "merge":
        merge(parser.parse_merge())
    elif op == "clear":
        queue.clear()
    else:
        print_help(invalid=not ("-h" in sys.argv[1] or "help" in sys.argv[1]))


if __name__ == "__main__":
    main()
