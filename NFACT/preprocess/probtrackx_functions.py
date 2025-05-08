from NFACT.base.filesystem import get_current_date
from NFACT.base.utils import colours, error_and_exit
from NFACT.base.cluster_support import (
    cluster_submission,
    Queue_Monitoring,
)
import os
import subprocess
import multiprocessing
import signal


def fail_safe_for_nfact_pp(warp: str, bpx: str):
    """
    Fail safe function to check that
    warps and bedpostx directory given.

    Parameters
    ----------
    warp: str
        str og warp files
    bpx: str
        str of bedpostX directory

    Returns
    -------
    None
    """
    arg_names = ["warps", "bedpostX directory"]
    for idx, arg in enumerate([warp, bpx]):
        error_and_exit(
            arg,
            f"{arg_names[idx]} not given. If using nfact pipeline check the json file",
        )


def check_bpx_dir(bpx_path: str, nodiff_mask: str) -> None:
    """
    Function to check bedpostX dir

    Parameters
    ----------
    bpx_path: str
        path to bedpost directory
    nodiff_mask: str
        path to nodiff mask

    Returns
    -------
    None
    """
    error_and_exit(
        os.path.exists(os.path.dirname(bpx_path)), "BedpostX directory does not exist."
    )
    error_and_exit(
        os.path.isfile(nodiff_mask + ".nii.gz"),
        "No diff mask in Bedpost X directory does not exist.",
    )


def process_command_arguments(arg: dict, sub: str) -> dict:
    """
    Function to process command line
    arguments

    Parameters
    -----------
    arg: dict
        dictionary of arguments from
        command line
    sub: str
        subjects full path

    Returns
    -------
    dict: dictonary oject
        dict of processed
        command line arguments
    """
    fail_safe_for_nfact_pp(
        arg["warps"],
        arg["bpx_path"],
    )

    return {
        "warps": [os.path.join(sub, warp.lstrip("/")) for warp in arg["warps"]],
        "seed": os.path.join(
            arg["outdir"], "nfact_pp", os.path.basename(sub), "seeds.txt"
        ),
        "bpx_path": os.path.join(sub, arg["bpx_path"].lstrip("/")),
    }


def build_warp_options(warps: list) -> list:
    """
    Function to build warp options

    Parameters
    -----------
    warps: list
        list of warps
    Returns
    -------
    warp_options: list
        lsit of warp options
    """
    warp_options = [f"--xfm={warps[0]}"]
    if len(warps) > 1:
        warp_options.extend([f"--invxfm={warps[1]}"])
    return warp_options


def get_target_mask(sub: str, arg: dict) -> str:
    """
    Parameters
    ----------
    arg: dict
        dictionary of arguments from
        command line
    sub: str
        subjects full path
    output_dir: str
        path to output directory

    Returns
    -------
    str: string object
        string option to target mask
    """
    return (
        os.path.join(sub, arg["target2"])
        if arg["target2"]
        else os.path.join(arg["outdir"], "nfact_pp", "target2.nii.gz")
    )


def get_probtrackx_bin(use_gpu: bool) -> str:
    """
    Function to get probtrackx binary

    Parameters
    ----------
    use_gpu: bool
        boolean to

    Returns
    -------
    str: string object
        path to probtrackx bin
    """
    prob_bin = "probtrackx2_gpu" if use_gpu else "probtrackx2"
    return os.path.join(os.environ["FSLDIR"], "bin", prob_bin)


def exclsuion_mask(exclusion_path: str) -> list:
    """
    Function to add exclusion
    mask to probtrackx

    Parameters
    ----------
    exclusion_path: str
        path to exclusion mask

    Returns
    -------
    list: list obect
        list with avoid argument
    """
    col = colours()
    print(f"{col['pink']}Processing:{col['reset']} Exclusion mask {exclusion_path}")
    return [f"--avoid={exclusion_path}"]


def add_stoppage_args(arg: dict, nfactpp_diretory: str, sub: str, sub_id: str):
    """
    Function to add in stoppage masks

    arg: dict,
       cmd processes
    nfactpp_diretory: str
        path to nfactpp_directory
    sub: str
        path to sub dirs
    sub_id: str
        subject id

    Returns
    -------
    list: list object
        list of stop and wtstop
        arguments
    """
    # Lazy import needed to stop circular imports
    from NFACT.preprocess.nfactpp_functions import stop_masks

    col = colours()
    print(f"{col['pink']}Processing:{col['reset']} stop and wtstop files")
    return stop_masks(arg, nfactpp_diretory, sub, sub_id)


def build_probtrackx2_arguments(arg: dict, sub: str, ptx_options=False) -> list:
    """
    Function to build out probtrackx2 arguments

    Parameters
    ----------
    arg: dict
        dictionary of arguments from
        command line
    sub: str
        subjects full path
    output_dir: str
        path to output directory

    Returns
    -------
    list: list object
        list of probtrackx2 arguements
    """
    command_arguments = process_command_arguments(arg, sub)
    prob_bin = "probtrackx2_gpu" if arg["gpu"] else "probtrackx2"
    binary = os.path.join(os.environ["FSLDIR"], "bin", prob_bin)
    warps = command_arguments["warps"]
    seeds = command_arguments["seed"]
    mask = os.path.join(command_arguments["bpx_path"], "nodif_brain_mask")
    target_mask = get_target_mask(sub, arg)
    bpx = os.path.join(command_arguments["bpx_path"], "merged")
    check_bpx_dir(bpx, mask)
    output_dir = os.path.join(
        arg["outdir"], "nfact_pp", os.path.basename(sub), "omatrix2"
    )
    warp_options = build_warp_options(warps)

    command = [
        binary,
        "-x",
        seeds,
        "-s",
        bpx,
        f"--mask={mask}",
        f"--seedref={arg['seedref']}",
        "--omatrix2",
        f"--target2={target_mask}",
        "--loopcheck",
        "--forcedir",
        "--opd",
        "--pd",
        f"--nsamples={arg['nsamples']}",
        f"--dir={output_dir}",
    ]

    command.extend(warp_options)
    if arg["exclusion"]:
        command.extend(exclsuion_mask(arg["exclusion"]))
    if arg["stop"]:
        command.extend(
            add_stoppage_args(
                arg, os.path.dirname(output_dir), sub, os.path.basename(sub)
            )
        )
    if "waypoints" in arg.keys():
        command.extend([f'--waypoints={os.path.join(sub, arg["waypoints"])}'])
    if ptx_options:
        command.extend(ptx_options)
    return command


def downsample_target2(
    target_img: str,
    output_dir: str,
    resolution: str,
    reference_img: str,
    interpolation_strategy: str,
) -> None:
    """
    Function to create target2 image

    Parameters
    ----------
    target_img: str
        string to target image
    output: str
        string to output directory
    resolution: str
        resolution of target2
    reference_img: str
        reference input
    interpolation_strategy: str
        interpolation, either
        trilinear,
        nearestneighbour,
        sinc,
        spline

    Returns
    -------
    None
    """

    try:
        run = subprocess.run(
            [
                "flirt",
                "-in",
                target_img,
                "-out",
                output_dir,
                "-applyisoxfm",
                str(resolution),
                "-ref",
                reference_img,
                "-interp",
                interpolation_strategy,
            ],
            capture_output=True,
        )
    except FileNotFoundError:
        error_and_exit(False, "Unable to find reference image. Please check it exists")
    except subprocess.CalledProcessError as error:
        error_and_exit(False, f"Error in calling FSL flirt: {error}")
    except KeyboardInterrupt:
        run.kill()

    if run.returncode != 0:
        error_and_exit(
            False, f"FSL FLIRT failure due to {run.stderr}. Unable to build target2"
        )


def seeds_to_ascii(surfin: str, roi: str, surfout: str) -> None:
    """
    Function to create seeds from
    surfaces.

    Parameters
    ----------
    surfin: str
        input surface
    roi: str,
        roi to restrict seeding
    surfout: str
        name of output surface.
        Needs to be full path

    Returns
    -------
    None
    """

    try:
        run = subprocess.run(
            [
                "surf2surf",
                "-i",
                surfin,
                "-o",
                surfout,
                f"--values={roi}",
                "--outputtype=ASCII",
            ],
            capture_output=True,
        )

    except subprocess.CalledProcessError as error:
        error_and_exit(False, f"Error in calling surf2surf: {error}")
    except KeyboardInterrupt:
        run.kill()

    if run.returncode != 0:
        error_and_exit(
            False,
            f"FSL surf2surf failure due to {run.stderr}. Unable to create asc surface",
        )


def get_probtrack2_arguments(bin: bool = False) -> None:
    """
    Function to get probtrack2
    arguments to check that user input
    is valid

    Parameters
    ----------
    bin: bool
        get the arguments for the
        gpu binary. Needed for NFACT
        pipeline

    Returns
    -------
    help_arguments: str
        string of help arguments
    """
    prob_bin = "probtrackx2" if not bin else "probtrackx2_gpu"
    binary = os.path.join(os.environ["FSLDIR"], "bin", prob_bin)
    try:
        help_arguments = subprocess.run([binary, "--help"], capture_output=True)
    except subprocess.CalledProcessError as error:
        error_and_exit(False, f"Error in calling probtrackx2: {error}")

    return help_arguments.stderr.decode("utf-8")


class Probtrackx:
    """
    Class to run probtrackx

    Usage
    -----
    probtrackx = Probtrackx(command: list,
        cluster_time: int,
        cluster_queue: str,
        cluster_ram: int,
        parallel: bool = False,
        dont_cluster: bool = False)
    probtrackx.run()
    """

    def __init__(
        self,
        command: list,
        cluster: bool,
        cluster_time: int,
        cluster_queue: str,
        cluster_ram: int,
        cluster_qos: str,
        gpu: bool,
        parallel: bool = False,
    ) -> None:
        self.col = colours()
        self.command = command
        self.parallel = int(parallel)
        self.cluster = cluster
        self.cluster_time = cluster_time
        self.cluster_queue = cluster_queue
        self.cluster_ram = cluster_ram
        self.cluster_qos = cluster_qos
        self.gpu = gpu

    def run(self):
        """
        Method to run probtrackx
        """
        if self.parallel:
            self.__parallel_mode()
        if not self.parallel:
            self.__single_subject_run()

    def __single_subject_command(self):
        """
        Method to get single subjects
        """
        return {
            "command": (self.__cluster if self.cluster else self._run_probtrackx),
            "print_str": "on cluster" if self.cluster else "locally",
        }

    def __single_subject_run(self) -> None:
        """
        Method to do single subject mode
        Loops over all the subject
        """
        run_probtractkx = self.__single_subject_command()
        print(
            f"{self.col['pink']}\nRunning subjects {run_probtractkx['print_str']}{self.col['reset']}"
        )

        submitted_jobs = []
        for sub_command in self.command:
            subject = self.__subject_id(self.__nfact_dir(sub_command))
            print(
                "Running",
                os.path.basename(sub_command[0]),
                f"on subject {subject}",
            )

            job = run_probtractkx["command"](sub_command)
            submitted_jobs.append(job)

        submitted_jobs = [job for job in submitted_jobs if job is not None]
        if submitted_jobs:
            queue = Queue_Monitoring()
            queue.monitor(submitted_jobs)

    def __cluster(self, command: list):
        """
        Method to submit jobs to cluster
        """
        return cluster_submission(
            command,
            self.cluster_time,
            self.cluster_ram,
            self.cluster_queue,
            f"nfact_pp_{os.path.basename(os.path.dirname(command[2]))}",
            os.path.join(self.__nfact_dir(command), "logs"),
            self.cluster_qos,
            self.gpu,
        )

    def __check_number_of_cores(self):
        """
        Method to check number of cores
        and number of subjects
        """

        number_of_subject = len(self.command)
        if self.parallel > number_of_subject:
            self.parallel = number_of_subject
            if self.parallel == 1:
                print(f"{self.col['red']}Only single subject given")
                print(
                    f"This might take longer. Consider removing --number_of_cores{self.col['reset']}"
                )

    def __parallel_mode(self) -> None:
        """
        Method to parallell process
        multiple subjects
        """
        self.__check_number_of_cores()
        print(
            f"{self.col['pink']}Parallel processing with {self.parallel} cores{self.col['reset']}"
        )
        pool = multiprocessing.Pool(processes=self.parallel)

        def kill_pool(sig, frame):
            """
            Method to kill pool safely.
            Also prints kill message so that
            the singit doesn't print it 100x
            times
            """

            pool.terminate()
            print(
                f"\n{self.col['darker_pink']}Recieved kill signal (Ctrl+C). Terminating..."
            )
            print(f"Exiting...{self.col['reset']}\n")
            exit(0)

        signal.signal(signal.SIGINT, kill_pool)
        pool.map(self._run_probtrackx, self.command)

    def __log_name(self):
        return "PP_log_" + get_current_date()

    def __log_path(self, nfactpp_directory):
        return os.path.join(nfactpp_directory, "logs", self.__log_name())

    def __nfact_dir(self, command):
        return os.path.dirname(command[2])

    def __subject_id(self, nfactpp_directory: str):
        return os.path.basename(nfactpp_directory)

    def _run_probtrackx(self, command: list) -> None:
        """
        Method to run probtrackx

        Parameters
        ----------
        command: list
            command in list form to run
        log_path: str
            Log path

        Returns
        -------
        None
        """
        nfactpp_directory = self.__nfact_dir(command)
        try:
            with open(self.__log_path(nfactpp_directory), "w") as log_file:
                run = subprocess.run(
                    command,
                    stdout=log_file,
                    stderr=log_file,
                    universal_newlines=True,
                )
        except subprocess.CalledProcessError as error:
            error_and_exit(False, f"Error in calling probtrackx2: {error}")
        except KeyboardInterrupt:
            run.kill()
        except Exception as e:
            error_and_exit(False, f"The following error occured: {e}")
        if run.returncode != 0:
            error_and_exit(False, f"Error in {command[0]} please check log files")
