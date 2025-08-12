from NFACT.base.imagehandling import imaging_type, get_cifti_data
import numpy as np
import nibabel as nb
import os
import glob
from tqdm import tqdm


class Component_loading:
    def __init__(self, white_group_component_path, grey_group_component_path, dim):
        self.white_group_component_path = white_group_component_path
        self.grey_group_component_path = grey_group_component_path
        self.dim = dim

    def run(self, subject_paths):
        self.__load_group_components()
        w_corr = []
        g_corr = []
        for subject in tqdm(
            subject_paths, desc="Component Loadings", colour="magenta", unit=" Subject"
        ):
            component_loadings = self._subject_correlations(subject)
            w_corr.append(component_loadings["w"])
            g_corr.append(component_loadings["g"])
            breakpoint()
        return {
            "w_correlations": np.vstack(w_corr),
            "g_correlations": np.vstack(g_corr),
        }

    def __load_group_components(self):
        self.group_white = self.__volume(self.white_group_component_path)
        self.group_grey = self.__process_grey(self.grey_group_component_path)

    def __process_grey(self, grey_paths):
        if imaging_type(grey_paths[0]) == "cifti":
            return self.__cifti(grey_paths[0])
        loaders = {"nifti": self.__volume, "gifti": self.__gifti}
        grey_component = []
        for grey_img in grey_paths:
            grey_comp = loaders[imaging_type(grey_img)](grey_img)
            grey_component.append(grey_comp)
        return np.vstack(grey_component)

    def __volume(self, vol_path):
        vol_data = nb.load(vol_path).get_fdata()
        return vol_data.reshape(-1, vol_data.shape[-1])

    def __gifti(gifti_path):
        gifti_img = nb.load(gifti_path)
        return np.column_stack([darray.data for darray in gifti_img.darrays])

    def __cifti(self, cifti_path):
        gm_dat = get_cifti_data(cifti_path)
        component_data = np.concatenate([gm_dat["L_surf"], gm_dat["R_surf"]], axis=0)
        if "vol" in gm_dat.keys():
            vol_flat = gm_dat["vol"].get_fdata().reshape(-1, gm_dat["vol"].shape[-1])
            component_data = np.concatenate([component_data, vol_flat], axis=0)
        return component_data

    def __get_subject_img(self, subject_path):
        basename = os.path.basename(subject_path)
        return {
            "grey_components": glob.glob(
                os.path.join(
                    os.path.dirname(subject_path), f"G_{basename}_dim{self.dim}*"
                )
            ),
            "white_component": glob.glob(
                os.path.join(
                    os.path.dirname(subject_path), f"W_{basename}_dim{self.dim}*"
                )
            ),
        }

    def __correlating(self, subject_data, group_data):
        return [
            float(np.corrcoef(group_data[:, comp], subject_data[:, comp])[0, 1])
            for comp in range(subject_data.shape[1])
        ]

    def _subject_correlations(self, subject_path):
        subject_images = self.__get_subject_img(subject_path)
        w_subject = self.__volume(subject_images["white_component"][0])
        w_subject_correlations = self.__correlating(w_subject, self.group_white)
        del w_subject
        g_subject = self.__process_grey(subject_images["grey_components"])
        g_subject_correlations = self.__correlating(g_subject, self.group_grey)
        del g_subject
        return {"w": w_subject_correlations, "g": g_subject_correlations}
