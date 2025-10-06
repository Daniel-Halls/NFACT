from NFACT.base.imagehandling import get_cifti_data, save_volume
from NFACT.base.utils import colours
from NFACT.base.setup import return_list_of_subjects_from_file
import nibabel as nib
from glob import glob
import numpy as np
import os


def splash():
    col = colours()
    return f"""{col['pink']}
 _____  _           _                               _____                     _                
/  ___|| |         | |                             /  __ \                   | |               
\ `--. | |_   __ _ | |_   _ __ ___    __ _  _ __   | /  \/ _ __   ___   __ _ | |_   ___   _ __ 
 `--. \| __| / _` || __| | '_ ` _ \  / _` || '_ \  | |    | '__| / _ \ / _` || __| / _ \ | '__|
/\__/ /| |_ | (_| || |_  | | | | | || (_| || |_) | | \__/\| |   |  __/| (_| || |_ | (_) || |   
\____/  \__| \__,_| \__| |_| |_| |_| \__,_|| .__/   \____/|_|    \___| \__,_| \__| \___/ |_|   
                                           | |                                               
                                            -
{col['reset']}
"""


def subject_variability_map(group_comp, sub_data):
    return (sub_data - np.mean(group_comp)) / np.std(group_comp)


def get_subjects(path, img_type):
    return glob(os.path.join(path, f"{img_type}_*nii*"))


def merge_components(data_to_merge, comp, vol: bool = True) -> np.ndarray:
    if vol:
        return np.sum(data_to_merge[:, :, :, comp], axis=3)
    return np.sum(data_to_merge[:, comp], axis=1)


def create_vol_map(vol_path, comp):
    vol_data = nib.load(vol_path).get_fdata()
    return merge_components(vol_data, comp)


def merge_volumes(subjects, comp):
    subject_maps = [create_vol_map(subj, comp) for subj in subjects]
    return np.stack(subject_maps, axis=3)


def get_group_maps(group_path, comp):
    group = nib.load(os.path.join(group_path, "W_NMF_dim100.nii.gz")).get_fdata()
    group_comp = merge_components(group, comp)
    cifit_data = process_cifti(
        os.path.join(group_path, "G_NMF_dim100.dscalar.nii"), comp
    )
    cifit_data["wm"] = group_comp
    return cifit_data


def sub_variance(group_map, subject_map):
    val = (subject_map - group_map) * 2
    return normalization(val)


def normalization(data):
    normalized = np.zeros_like(data, dtype=np.float64)
    mask = data != 0
    nonzero_vals = data[mask]
    vmin = nonzero_vals.min()
    vmax = nonzero_vals.max()
    range_val = vmax - vmin
    if range_val == 0:
        normalized[mask] = 1
    else:
        normalized[mask] = (nonzero_vals - vmin) / range_val

    return normalized


def get_variance_maps(group_data, subject_data, vol=True):
    if vol:
        return np.stack(
            [
                sub_variance(group_data, subject_data[:, :, :, sub])
                for sub in range(subject_data.shape[3])
            ],
            axis=3,
        )
    return np.stack(
        [
            sub_variance(group_data, subject_data[:, sub])
            for sub in range(subject_data.shape[1])
        ],
        axis=1,
    )


def create_4d_vol(subjects, comp):
    return merge_volumes(subjects, comp)


def save_volume_wrapper(meta_data_nifit_path, vol_to_save, outdir, cifti=False):
    if cifti:
        vol_info = get_cifti_data(meta_data_nifit_path)["vol"]
    else:
        vol_info = nib.load(meta_data_nifit_path)
    save_volume(vol_info, vol_to_save, outdir)


def save_gm_surf(darrays, file_name):
    nib.GiftiImage(darrays=darrays).to_filename(f"{file_name}.func.gii")


def process_cifti(cifti_path, comp) -> dict:
    cifit_data = get_cifti_data(cifti_path)
    l_surf = merge_components(cifit_data["L_surf"], comp, vol=False)
    r_surf = merge_components(cifit_data["R_surf"], comp, vol=False)
    vol_comp = merge_components(cifit_data["vol"].get_fdata(), comp)
    return {"l_surf": l_surf, "r_surf": r_surf, "vol": vol_comp}


def create_darray(gii_data):
    return [
        nib.gifti.GiftiDataArray(
            data=gii_data, datatype="NIFTI_TYPE_FLOAT32", intent=2001
        )
    ]


def create_gm_maps(subjects, comp):
    results = [process_cifti(sub, comp) for sub in subjects]
    return {
        "l_surf": np.stack([dat["l_surf"] for dat in results], axis=1),
        "r_surf": np.stack([dat["r_surf"] for dat in results], axis=1),
        "vol": np.stack([dat["vol"] for dat in results], axis=3),
    }


def save_cifit_component(subjects, outdir, gm_data, prefix):
    ldarray = create_darray(gm_data["l_surf"])
    rdarray = create_darray(gm_data["r_surf"])
    save_volume_wrapper(
        subjects[0],
        gm_data["vol"],
        os.path.join(outdir, f"{prefix}_subcortical_stat_map.nii.gz"),
        cifti=True,
    )
    save_gm_surf(ldarray, os.path.join(os.path.join(outdir, f"{prefix}_left_stat_map")))
    save_gm_surf(
        rdarray, os.path.join(os.path.join(outdir, f"{prefix}_right_stat_map"))
    )


def process_gm(subjects, comp):
    return create_gm_maps(subjects, comp)


def extract_id(path):
    """Extract the subject ID (e.g., 'BANDA123') from the file path."""
    filename = path.split("/")[-1]
    parts = filename.split("_")
    if len(parts) >= 2:
        return parts[1]
    return None


def get_sort_index(path, order_dict):
    """Return the index of the subject ID from the order_dict, or inf if not found."""
    sub_id = extract_id(path)
    return order_dict.get(sub_id, float("inf"))


def sort_paths_by_subject_order(file_paths, subject_order):
    """
    Sort a list of file paths according to a desired subject ID order.

    Parameters:
    - file_paths: list of str, full file paths
    - subject_order: list of str, subject IDs like

    Returns:
    - list of str, sorted file paths
    """
    order_dict = {sub_id: idx for idx, sub_id in enumerate(subject_order)}
    return sorted(file_paths, key=lambda path: get_sort_index(path, order_dict))


def main():
    col = colours()
    args = stat_map_args()
    subjects = return_list_of_subjects_from_file(args["list_of_subjects"])
    print(f"{col['darker_pink']}Merging components:{col['reset']}", *args["components"])
    folder_path = os.path.join(args["folder_path"], "nfact_dr", "NMF")
    print(f"{col['darker_pink']}Folder Path:{col['reset']} {folder_path}")

    print(f"{col['darker_pink']}Number of Subjects:{col['reset']} {len(subjects)}")
    print(f"\n{col['plum']}Creating and Saving 4D white matter volume{col['reset']}")
    print("-" * 100)
    subjects_w = sort_paths_by_subject_order(get_subjects(folder_path, "W"), subjects)

    subject_W_maps = create_4d_vol(subjects_w, args["components"])
    save_volume_wrapper(
        subjects_w[0],
        subject_W_maps,
        os.path.join(args["outdir"], "R_W_stat_map.nii.gz"),
    )

    print(f"\n{col['plum']}Creating Grey matter files{col['reset']}")
    print("-" * 100)
    subjects_g = sort_paths_by_subject_order(get_subjects(folder_path, "G"), subjects)
    gm_data = process_gm(subjects_g, args["components"])
    save_cifit_component(subjects_g, args["outdir"], gm_data, "R")

    print(f"\n{col['plum']}Calculating variance maps{col['reset']}")
    group_maps = get_group_maps(
        os.path.join(
            args["folder_path"], "nfact_decomp", "components", "NMF", "decomp"
        ),
        args["components"],
    )
    wm = get_variance_maps(group_maps["wm"], subject_W_maps)
    save_volume_wrapper(
        subjects_w[0], wm, os.path.join(args["outdir"], "V_W_stat_map.nii.gz")
    )
    cifit_var = {
        "r_surf": get_variance_maps(group_maps["r_surf"], gm_data["r_surf"], vol=False),
        "l_surf": get_variance_maps(group_maps["l_surf"], gm_data["l_surf"], vol=False),
        "vol": get_variance_maps(group_maps["vol"], gm_data["vol"]),
    }
    save_cifit_component(subjects_g, args["outdir"], cifit_var, "V")
    print(f"\n{col['plum']}Finished{col['reset']}")
    exit(0)


if __name__ == "__main__":
    main()
