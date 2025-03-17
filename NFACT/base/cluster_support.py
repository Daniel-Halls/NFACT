from NFACT.base.utils import colours, error_and_exit
import os
import subprocess
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
        """Method to assign ram amount. Largely Reduant"""
        self.arg["cluster_ram"] = (
            self.arg["cluster_ram"] if self.arg["cluster_ram"] else "60"
        )

    def cluster_time(self):
        """Method to assign cluster time"""
        self.arg["cluster_time"] = (
            self.arg["cluster_time"] if self.arg["cluster_time"] else "600"
        )

        self.arg["cluster_time"] = (
            "160" if self.arg["gpu"] else self.arg["cluster_time"]
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
    Function to build base command for fsl sub

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
    queue: str = None,
    qos: str = None,
    gpu: bool = False,
    parallel: int = None,
) -> list:
    """
    Function to build out fsl sub command.
    Adds in queue, cuda, qos and command
    to run.

    Parameters
    ----------
    cluster_command: list
        base cluster command
    command_to_run: list
        command to run on cluster
    queue: str = None
        Queue to send command to.
        Can be None as fsl sub
        can assign queue.
    qos: str = None
        SLURM qos. Can be None
    gpu: bool = False
        To use GPU.
    parallel: int = None
        Assigns parallel slots
        for fsl_sub

    Returns
    -------
    cluster_command: list
        built out command
        to run on cluster.
    """
    if qos:
        cluster_command.append(str(qos))
    if queue:
        cluster_command.extend(["-q", str(queue)])
    if gpu:
        cluster_command.extend(["-c", "cuda"])
    if parallel:
        cluster_command.extend(["-s", str(parallel)])
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

    def __init__(self) -> None:
        self.__spinner_running = True
        self.__col = colours()
        print(f"{self.__col['pink']}\nStarting Queue Monitoring{self.__col['reset']}")

    def monitor(self, job_id: list) -> None:
        """
        Main method to monitor queue.

        Parameters
        ----------
        job_id: list
            list of job_ids

        Returns
        -------
        None
        """

        self.__spinner_running = True
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
                        pbar.close()
                        print("All jobs have finihsed")
                        break
                    time.sleep(300)

        except KeyboardInterrupt:
            pbar.close()
        finally:
            self.__spinner_running = False
            spinner_thread.join()

    def __spinner(self) -> None:
        """
        Method to run spinner bar
        while monitoring the queue.
        """
        hash_line = ""
        max_hashes = 50
        adding_hash = True

        while self.__spinner_running:
            if adding_hash:
                hash_line += "#"
                if len(hash_line) >= max_hashes:
                    adding_hash = False
            else:
                hash_line = hash_line[:-1]
                if len(hash_line) == 0:
                    adding_hash = True

            print(
                f"{self.__col['deep_pink']}\033[1B\r{hash_line.ljust(max_hashes)}\033[1A{self.__col['reset']}",
                end="",
            )
            time.sleep(0.1)

    def __check_job(self, job_id: str) -> bool:
        """
        Method to check job progress.

        Parameters
        ----------
        job_id: str
            job ID of fsl sub job

        Returns
        -------
        bool: boolean
            True if job is still running
            or False if completed.
        """
        output = run_fsl_sub(
            [os.path.join(os.environ["FSLDIR"], "bin", "fsl_sub_report"), job_id]
        )
        if "Finished" in output["stdout"]:
            return False
        if "Failed" in output["stdout"]:
            tqdm.write(
                f"{self.__col['red']}JOB {job_id} FAILED. CHECK LOGS{self.__col['reset']}"
            )
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
        f"{col['darker_pink']}Cluster:{col['reset']} No queues detected. Not running on cluster"
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
        error_and_exit(False, f"Error in calling FSL sub due to: {error}")
    except KeyboardInterrupt:
        run.kill()
    output = {
        key: value.decode("utf-8").strip() if isinstance(value, bytes) else value
        for key, value in vars(run).items()
    }
    if output["stderr"]:
        error_and_exit(False, f"FSL sub failed due to {output['stderr']}")
    return output


def cluster_submission(
    command: list,
    cluster_time: str,
    cluster_ram: str,
    cluster_queue: str,
    log_name: str,
    log_location: str,
    cluster_qos: str,
    gpu: bool,
    parallel: int = None,
) -> str:
    """
    Function to submit jobs to cluster.

    Parameters
    ----------
    command: list
        command to run on cluster
    cluster_time: str,
        time job will take
    cluster_ram: str
        amount of ram needed
    cluster_queue: str = None
        Queue to send command to.
        Can be None as fsl sub
        can assign queue.
    log_name: str
        name of process and lo file
    log_location: str
        location of where log should be
        sent
    cluster_qos: str = None
        SLURM qos. Can be None
    gpu: bool = False
        To use GPU.
    parallel: int = None
        Assigns parallel slots
        for fsl_sub

    Returns
    -------
    str: string
        job id of cluster
        submission
    """
    bcluster_command = base_command(
        cluster_time,
        cluster_ram,
        log_location,
        log_name,
    )
    cluster_command = fsl_sub_cluster_command(
        bcluster_command, command, cluster_queue, cluster_qos, gpu, parallel
    )
    fsl_sub_rout = run_fsl_sub(cluster_command)
    return fsl_sub_rout["stdout"]


def processing_cluster(arg: dict) -> dict:
    """
    Function to process cluster.

    Parameters
    ----------
    arg: dict
        dictionary of command line arguments

    Returns
    -------
    arg: dict
       processed command line arguments
    """
    col = colours()
    print(f"{col['deep_pink']}Checking:{col['reset']} Cluster Availability\n")
    try:
        arg = Cluster_parameters(arg).process_parameters()
        print(f"{col['amethyst']}Using:{col['reset']} Cluster")
    except NoClusterQueuesException:
        arg["cluster"] = no_cluster_queues()
    return arg
