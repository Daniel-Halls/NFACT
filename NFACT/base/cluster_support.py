from .utils import colours, error_and_exit
import os
import subprocess
import itertools
import time
from tqdm import tqdm
import threading


class Cluster_parameters:
    """
    Class to process cluster parameters.

    Usage
    -----
    arg = Cluster_parameters().process_parameters(arg)

    Parameters
    ----------
    arg: dict
        command line arguments
    """

    def __init__(self, arg: dict):
        self.arg = arg
        self.col = colours()

    def process_parameters(self):
        """
        Method to process parameters

        Parameters
        ----------
        None

        Returns
        ------
        args: dict

        dictionary of processed
        command line arguments
        """
        queues = self.has_queues()
        queues_avail = queues["stdout"] != "No"
        if not queues_avail:
            raise NoClusterQueuesException
        self.cluster_ram()
        self.cluster_time()
        if self.arg["cluster_qos"]:
            self.cluster_qos()
        self.print_cluster_queues()
        return self.arg

    def has_queues(self) -> dict:
        """
        Method to check if queues
        available.
        """
        return run_fsl_sub(
            [os.path.join(os.environ["FSLDIR"], "bin", "fsl_sub"), "--has_queues"]
        )

    def config(self):
        return run_fsl_sub(
            [os.path.join(os.environ["FSLDIR"], "bin", "fsl_sub"), "--show_config"]
        )

    def cluster_ram(self):
        """Method to assign ram amount"""
        self.arg["cluster_ram"] = (
            self.arg["cluster_ram"] if self.arg["cluster_ram"] else "40"
        )

    def cluster_time(self):
        """Method to assign cluster time"""
        self.arg["cluster_time"] = (
            self.arg["cluster_time"]
            if self.arg["cluster_time"]
            else "160"
            if self.arg["gpu"]
            else "600"
        )

    def cluster_queue_check(self):
        """Method to assign cluster queue"""
        if self.arg["cluster_queue"]:
            queue_config = self.config()
            return (
                True if self.arg["cluster_queue"] in queue_config["stdout"] else False
            )

    def cluster_qos(self):
        """Method to assign cluster qos"""
        qos = self.arg["cluster_qos"]
        self.arg["cluster_qos"] = f'--extra "--qos={qos}"'

    def print_cluster_queues(self):
        """Method to print cluster queues"""
        print_string = (
            "No queue given will assign"
            if not self.arg["cluster_queue"]
            else self.arg["cluster_queue"]
        )
        is_queue_config = self.cluster_queue_check()
        if not is_queue_config:
            print_string = (
                print_string + " given but not configured. Queue may not work."
            )
        print(
            f"{self.col['darker_pink']}Cluster queue:{self.col['reset']} {print_string}"
        )


class NoClusterQueuesException(Exception):
    """Custom exception for when no cluster queues are available."""

    def __init__(self):
        super().__init__()


def base_command(
    cluster_time: str, cluster_ram: str, log_directory: str, log_name: str
) -> list:
    """
    Function to return base command

    Parameters
    ----------
    cluster_time: str
        Time job will take
    cluster_ram: str
        Amount of ram job will take
    log_directory: str
        Path to log directory
    log_name: str
        Name of log file

    Returns
    -------
    list: list object
        list of base cluster_command
    """
    return [
        os.path.join(os.environ["FSLDIR"], "bin", "fsl_sub"),
        "-T",
        str(cluster_time),
        "-R",
        str(cluster_ram),
        "-N",
        log_name,
        "-l",
        log_directory,
    ]


def fsl_sub_cluster_command(
    cluster_command: list,
    command_to_run: list,
    queue: str = False,
    qos: str = False,
    gpu: bool = False,
):
    """
    Function to
    """
    if qos:
        cluster_command.append(str(qos))
    if queue:
        cluster_command.extend(["-q", str(queue)])
    if gpu:
        cluster_command.extend(["-c", "cuda"])
    cluster_command.extend([" ".join(command_to_run)])
    return cluster_command


class Queue_Monitoring:
    """
    Class to Monitor cluster queue
    for jobs completion.

    Usage
    ----
    queue = Queue_Monitoring()
    queue.monitor(list_of_job_ids)
    """

    def __init__(self):
        self.spinner_running = True

    def monitor(self, job_id: list) -> None:
        """
        Main method to monitor queue.

        Parameters
        ----------
        job_id: list
            list of job_ids
        """
        print("\nStarting Queue Monitoring")

        self.spinner_running = True  # Control variable for the spinner thread
        spinner_thread = threading.Thread(target=self.__spinner, daemon=True)
        spinner_thread.start()

        try:
            with tqdm(
                total=len(job_id), desc="Jobs completed", unit="job", colour="magenta"
            ) as pbar:
                completed_jobs = []
                time.sleep(100)
                while True:
                    for job in job_id:
                        if job not in completed_jobs:
                            running = self.__check_job(job)
                            if not running:
                                pbar.update(1)
                                completed_jobs.append(job)

                    if len(completed_jobs) == len(job_id):
                        print("All jobs have finihsed")
                        break
                    time.sleep(300)

        except KeyboardInterrupt:
            pass
        finally:
            self.spinner_running = False
            spinner_thread.join()

    def __spinner(self):
        spinner_cycle = itertools.cycle(["|", "\\", "-", "/"])
        while self.spinner_running:
            print(f"\r{next(spinner_cycle)}", end="")
            time.sleep(0.1)

    def __check_job(self, job_id):
        output = run_fsl_sub(
            [os.path.join(os.environ["FSLDIR"], "bin", "fsl_sub_report"), job_id]
        )
        if "Finished" in output["stdout"]:
            return False
        if "Failed" in output["stdout"]:
            col = colours()
            print(f"{col['red']}JOB FAILED. CHECK LOGS{col['reset']}")
            return False
        return True


def no_cluster_queues():
    """
    Function to print
    and return None if no
    cluster support.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    col = colours()
    print(
        f"{col['plum']}Cluster:{col['reset']} No queues detected. Not running on cluster"
    )
    return None


def run_fsl_sub(command: list) -> dict:
    """
    Function wrapper around fsl_sub calls.

    Parameters
    ----------
    command: list
        list of command to run
    report: bool
        to use fsl_sub_report

    Returns
    -------
    output: dict
        output of fsl_sub call
    """
    try:
        run = subprocess.run(
            command,
            capture_output=True,
        )

    except subprocess.CalledProcessError as error:
        error_and_exit(False, f"Error in calling fsl_sub due to: {error}")
    except KeyboardInterrupt:
        run.kill()
    output = {
        key: value.decode("utf-8").strip() if isinstance(value, bytes) else value
        for key, value in vars(run).items()
    }
    if output["stderr"]:
        error_and_exit(False, f"FSL sub failed due to {output['stderr']}")
    return output
