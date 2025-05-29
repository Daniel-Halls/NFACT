# NFACT

*This is a repo merging NFacT (Shaun Warrington, Ellie Thompson, and Stamatios Sotiropoulos) and ptx_decomp (Saad Jbabdi and Rogier Mars).*

## What is NFACT

NFACT (Non-negative matrix Factorisation of Tractography data) is a set of modules (as well as an end to end pipeline) that decomposes 
tractography data using NMF/ICA.

It consists of three "main" decomposition modules:
    
    - nfact_pp (Pre-process data for decomposition)
    
    - nfact_decomp (Decomposes a single or average group matrix using NMF or ICA)
    
    - nfact_dr (Dual regression on group matrix)

as well as two axillary "modules":
    
    - nfact_config (creates config files for the pipeline and changing any hyperparameters)
    
    - nfact_Qc (Creates hitmaps to check for bias in decomposition)

and a pipeline wrapper
    
    - nfact (runs nfact_pp, nfact_decomp, nfact_Qc and nfact_dr. nfact_pp, nfact_Qc and nfact_dr can all individually be skipped)

-----------------------------------------------------------------------------------------------------------------------------------------------------------

``` 
 _   _ ______   ___   _____  _____    
| \ | ||  ___| /   \ /  __ \|_   _|    
|  \| || |_   / /_\ \| /  \/  | |      
|     ||  _|  |  _  || |      | |    
| |\  || |    | | | || \__/\  | |    
\_| \_/\_|    \_| |_/ \____/  \_/ 
```

## NFACT pipeline

This pipeline runs nfact_pp, nfact_decomp and nfact_dr on tractography data that has been processed by bedpostx. 

The pipeline first creates the omatrix2 before running decompostion, quality control and if multiple subjects provided, then dual regression.

Please see further down in readme for further details on modules.

### Usage:

```
usage: nfact [-h] [-l LIST_OF_SUBJECTS] [-s SEED [SEED ...]] [-o OUTDIR] [-n FOLDER_NAME] [-c CONFIG] [-P] [-Q] [-D] [-O] [-w WARPS [WARPS ...]] [-b BPX_PATH] [-r ROI [ROI ...]] [-f FILE_TREE] [-sr SEEDREF] [-t TARGET2] [-d DIM] [-a ALGO] [-rf ROI]
             [--threshold THRESHOLD]

options:
  -h, --help            Shows help message and exit

Pipeline inputs:
  -l LIST_OF_SUBJECTS, --list_of_subjects LIST_OF_SUBJECTS
                        Absolute filepath to a text file containing absolute path to subjects. Consider using nfact_config to create subject list
  -s SEED [SEED ...], --seed SEED [SEED ...]
                        Relative path to either a single or multiple seeds. If multiple seeds given then include a space between paths. Must be the same across subjects.
  -o OUTDIR, --outdir OUTDIR
                        Absolute path to a directory to save results in.
  -n FOLDER_NAME, --folder_name FOLDER_NAME
                        Name of output folder. That contains within it the nfact_pp, nfact_decomp and nfact_dr folders. Default is nfact
  -c CONFIG, --config CONFIG
                        Provide an nfact_config file instead of using command line arguements. Configuration files provide control over all parameters of modules and can be created using nfact_config -C. If this is provided no other arguments are needed to run nfact
                        as arguments are taken from config file rather than command line.
  -P, --pp_skip         Skips nfact_pp. Pipeline still assumes that data has been pre-processed with nfact_pp before. If data hasn't been pre-processed with nfact_pp consider runing modules seperately
  -Q, --qc_skip         Skips nfact_qc.
  -D, --dr_skip         Skips nfact_dr so no dual regression is performed.
  -O, --overwrite       Overwirte existing file structure

nfact_pp inputs:
  -w WARPS [WARPS ...], --warps WARPS [WARPS ...]
                        Relative path to warps inside a subjects directory. Include a space between paths. Must be the same across subjects.
  -b BPX_PATH, --bpx BPX_PATH
                        Relative path to Bedpostx folder inside a subjects directory. Must be the same across subjects
  -r ROI [ROI ...], --roi ROI [ROI ...]
                        REQUIRED FOR SURFACE MODE: Relative path to a single ROI or multiple ROIS to restrict seeding to (e.g. medial wall masks). Must be the same across subject. ROIS must match number of seeds.
  -f FILE_TREE, --file_tree FILE_TREE
                        Use this option to provide name of predefined file tree to perform whole brain tractography. nfact_pp currently comes with HCP filetree. See documentation for further information.
  -sr SEEDREF, --seedref SEEDREF
                        Absolute path to a reference volume to define seed space used by probtrackx. Default is MNI space ($FSLDIR/data/standard/MNI152_T1_2mm.nii.gz).
  -t TARGET2, --target TARGET2
                        Absolute path to a target image. If not provided will use the seedref. Default is human MNI ($FSLDIR/data/standard/MNI152_T1_2mm.nii.gz).

nfact_decomp/nfact_dr inputs:
  -d DIM, --dim DIM     This is compulsory option. Number of dimensions/components to retain after running NMF/ICA.
  -a ALGO, --algo ALGO  Which decomposition algorithm to run. Options are: NMF (default), or ICA. This is case insensitive
  -rf ROI, --rf_decomp ROI
                        Absolute path to a text file containing the absolute path ROI(s) paths to restrict seeding to (e.g. medial wall masks). This is not needed if seeds are not surfaces. If used nfact_pp then this is the roi_for_decomp.txt file in the nfact_pp
                        directory. This option is not needed if the pipeline is being ran from nfact_pp onwards.

nfact_Qc inputs:
  --threshold THRESHOLD
                        Z score value to threshold hitmaps.

```

example call:

```
nfact --list_of_subject /absolute path/sub_list \
--seed thalamus.nii.gz \
--algo NMF \
--dim 100 \
--outdir /absolute path/save directory \
--warps standard2acpc_dc.nii.gz acpc_dc2standard.nii.gz \
--ref $FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz \
--bpx Diffusion.bedpostX 
```

With a config file:
```
nfact –config /absolute path/nfact_config.config  
```
-----------------------------------------------------------------------------------------------------------------------------------------------------------

```
 _   _ ______   ___   _____  _____     ______ ______
| \ | ||  ___| /   \ /  __ \|_   _|    | ___ \| ___ \
|  \| || |_   / /_\ \| /  \/  | |      | |_/ /| |_/ /
|     ||  _|  |  _  || |      | |      |  __/ |  __/
| |\  || |    | | | || \__/\  | |      | |    | |
\_| \_/\_|    \_| |_/ \____/  \_/      \_|    \_|

```
## NFACT PP
Pre-processing module of NFACT.

Under the hood NFACT PP is probtrackx2 omatrix2 option to get a seed by target connectivity matrix 

### Input for nfact_preproc

Required before running NFACT PP:
    - Crossing-fibre diffusion modelled data (bedpostX)
    - Seeds (either surfaces or volumes)

NFACT PP has three modes: surface , volume, and filestree.

Required input:
    - List of subjects (absolute path)
    - Output directory (absolute path)

Input needed for filetree mode:
    - .tree file (NFACT_PP comes with some defaults such as hcp)

Input needed for both surface and volume mode:
    - Seeds path inside folder (relative path, must be same across subjects)
    - Warps path inside a subjects folder (relative path, must be same across subjects)
    - bedpostx folder path inside a subjects folder (relative path, must be same across subjects)
   
Input for surface seed mode:
    - Seeds as surfaces (relative path, must be same across subjects)
    - ROI as surfaces. This is files to restrict seeding to (for example surface files that exclude medial wall, this is a relative path, must be same across subjects)

Input needed for volume mode:
    - Seeds as volumes (relative path, must be same across subjects)

Warps must be ordered Standard2diff and Diff2standard. If your target fdt paths doesn't match up to the template then it is most likely the warps being the wrong way around.

#### Other NFACT_PP inputs

NFACT_PP needs a seed reference space to define seed space used by probtrackx. This is optional however this default is Human MNI. 

NFACT_PP needs a target2 img (a target for the seeds). This can be a whole brain mask or an ROIs mask, and it is recommended that they are downsampled. If the target2 is not given then the seedrefernce will be used. 


### NFACT PP input folder 

NFACT pp can be used in a folder agnostic way by providing the paths to seeds/bedpostX/target inside a subject folder (i.e --seeds seeds/amygdala.nii.gz).
However, NFACT pp does have an  --absolute option which will treat the seeds (and rois) as absolute paths. This way one set of seeds and rois can be passed to all subjects

The other way is to use the --file_tree command with the name of a file tree (see https://open.win.ox.ac.uk/pages/fsl/file-tree/index.html for further details on filetree).
In this case seeds/rois/bedpostx do not need to be specified as nfact_pp will try and find the appropriate files.

```
nfact_pp --file_tree hcp --list_of_subjects /home/study/list_of_subjects
```

Filetrees are saved in filetrees folder in nfact, so custom filetrees can be put there and called similar to the command above. NFACT_PP currently has a built in a filetree for HCP and a hcp_qunex (HCP in qunex format) to perform full brain tractography. 

Use of custom filetree
-----------------------
seed files are aliased as (seed), roi as (roi), warps as (diff2std, std2diff) and bedpostX as (bedpostX). Two seeds are supported if the seeds are bilateral indicated with {hemi}.seed, with the actual seed names being L.seed.nii.gz/R.seed.nii.gz. A singe seed can be given as well.

CIFTI support
--------------
NFACT can save files as cifti dscalars. However, seeds must be in the following order: left_hemisphere.gii, right hemisphere.nii, follwed by optional nifti files as subcortical structures. Left and right surfaces are needed however the subcortical structures are optional, and you can put as many or as few as you would (or none at all).

Subcortical structures must be named as standard cifti structures (i.e CIFTI_STRUCTURE_ACCUMBENS_LEFT.nii.gz) or subcortical data is put as the CIFTI structure OTHER. For further details on naming conventions please see the hcp_cifti in the NFACT/filetree folder in this repo

### Usage:

```
usage: nfact_pp [-h] [-hh] [-O] [-l LIST_OF_SUBJECTS] [-o OUTDIR] [-G] [-f FILE_TREE] [-s SEED [SEED ...]] [-w WARPS [WARPS ...]] [-b BPX_PATH] [-r ROI [ROI ...]] [-sr SEEDREF] [-t TARGET2] [-ns NSAMPLES] [-mm MM_RES] [-p PTX_OPTIONS] [-e EXCLUSION] [-S [STOP ...]]
                [-A] [-n N_CORES] [-C] [-cq CLUSTER_QUEUE] [-cr CLUSTER_RAM] [-ct CLUSTER_TIME] [-cqos CLUSTER_QOS]

options:
  -h, --help            show this help message and exit
  -hh, --verbose_help   Verbose help message. Prints help message and example usages
  -O, --overwrite       Overwrites previous file structure
  -G, --gpu             To use the GPU version of probtrackx2.

Set Up Arguments:
  -l LIST_OF_SUBJECTS, --list_of_subjects LIST_OF_SUBJECTS
                        Filepath to a list of subjects
  -o OUTDIR, --outdir OUTDIR
                        Path to output directory

Filetree option:
  -f FILE_TREE, --file_tree FILE_TREE
                        Use this option to provide name of a predefined file tree to perform whole brain tractography. nfact_pp currently comes with a number of HCP filetrees. See documentation for further information.

Tractography options:
  -s SEED [SEED ...], --seed SEED [SEED ...]
                        Relative path to either a single or multiple seeds. If multiple seeds given then include a space between paths. Path to file must be the same across subjects.
  -w WARPS [WARPS ...], --warps WARPS [WARPS ...]
                        Relative path to warps inside a subjects directory. Include a space between paths. Path to file must be the same across subjects. Expects the order as Standard2diff and Diff2standard.
  -b BPX_PATH, --bpx BPX_PATH
                        Relative path to Bedpostx folder inside a subjects directory. Path to file must be the same across subjects
  -r ROI [ROI ...], --roi ROI [ROI ...]
                        REQUIRED FOR SURFACE MODE: Relative path to a single ROI or multiple ROIS to restrict seeding to (e.g. medial wall masks). Must be the same across subject. ROIS must match number of seeds.
  -sr SEEDREF, --seedref SEEDREF
                        Absolute path to a reference volume to define seed space used by probtrackx. Default is MNI space ($FSLDIR/data/standard/MNI152_T1_2mm.nii.gz).
  -t TARGET2, --target TARGET2
                        Absolute path to a target image. If not provided will use the seedref. Default is human MNI ($FSLDIR/data/standard/MNI152_T1_2mm.nii.gz).
  -ns NSAMPLES, --nsamples NSAMPLES
                        Number of samples per seed used in tractography. Default is 1000
  -mm MM_RES, --mm_res MM_RES
                        Resolution of target image. Default is 2 mm
  -p PTX_OPTIONS, --ptx_options PTX_OPTIONS
                        Path to ptx_options file for additional options. Currently doesn't override defaults
  -e EXCLUSION, --exclusion EXCLUSION
                        Absolute path to an exclusion mask. Will reject pathways passing through locations given by this mask
  -S [STOP ...], --stop [STOP ...]
                        Use wtstop and stop in the tractography. Takes an absolute file path to a json file containing stop and wtstop masks, JSON keys must be stopping_mask and wtstop_mask. Argument can be used with the --filetree, in that case no json file is
                        needed.
  -A, --absolute        Treat seeds and rois as absolute paths, providing one set of seeds and rois for tractography across all subjects.

Parallel Processing arguments:
  -n N_CORES, --n_cores N_CORES
                        If should parallel process locally and with how many cores. This parallelizes the number of subjects. If n_cores exceeds subjects nfact_pp sets this argument to be the number of subjects. If nfact_pp is being used on one subject then this may
                        slow down processing.

Cluster Arguments:
  -C, --cluster         Use cluster enviornment
  -cq CLUSTER_QUEUE, --queue CLUSTER_QUEUE
                        Cluster queue to submit to
  -cr CLUSTER_RAM, --cluster_ram CLUSTER_RAM
                        Ram that job will take. Default is 60
  -ct CLUSTER_TIME, --cluster_time CLUSTER_TIME
                        Time that job will take. nfact_pp will assign a time if none given
  -cqos CLUSTER_QOS, --cluster_qos CLUSTER_QOS
                        Set the qos for the cluster

Example Usage:
    Surface mode:
           nfact_pp --list_of_subjects /absolute_path/study/sub_list
               --outdir absolute_path/study
               --bpx_path /relative_path/.bedpostX
               --seeds /relative_path/L.surf.gii /path_to/R.surf.gii
               --roi /relative_path/L.exclude_medialwall.shape.gii /path_to/R.exclude_medialwall.shape.gii
               --warps /relative_path/stand2diff.nii.gz /relative_path/diff2stand.nii.gz
               --n_cores 3

    Volume mode:
            nfact_pp --list_of_subjects /absolute_path/study/sub_list
                --outdir /absolute_path/study
                --bpx_path /relative_path/.bedpostX
                --seeds /relative_path/L.white.nii.gz /relative_path/R.white.nii.gz
                --warps /relative_path/stand2diff.nii.gz /relative_path/diff2stand.nii.gz
                --seedref absolute_path/MNI152_T1_1mm_brain.nii.gz
                --target absolute_path/dlpfc.nii.gz

    Filestree mode:
        nfact_pp --filestree hcp
            --list_of_subjects /absolute_path/study/sub_list
            --outdir /absolute_path/study


```

------------------------------------------------------------------------------------------------------------------------------------------

```
 _   _ ______   ___   _____  _____  ______  _____  _____  _____ ___  ___ _____
| \ | ||  ___| / _ \ /  __ \|_   _| |  _  \|  ___|/  __ \|  _  ||  \/  || ___ \
|  \| || |_   / /_\ \| /  \/  | |   | | | || |__  | /  \/| | | || .  . || |_/ /
| . ` ||  _|  |  _  || |      | |   | | | ||  __| | |    | | | || |\/| ||  __/
| |\  || |    | | | || \__/\  | |   | |/ / | |___ | \__/\\ \_/ /| |  | || |
\_| \_/\_|    \_| |_/ \____/  \_/   |___/  \____/  \____/ \___/ \_|  |_/\_|

```
## NFACT decomp
This is the main decompoisition module of NFACT. Runs either ICA or NMF and saves the components in the nfact_decomp directory. Components can also be normalised with the zscore maps saved, which is useful for visualization. Winner takes all maps can be created with the brain represented by which 
components are the "winner" in that region

### Usage
```
usage: nfact_decomp [-h] [-hh] [-O] [-l LIST_OF_SUBJECTS] [-o OUTDIR] [--seeds SEEDS] [--roi ROI] [-n CONFIG] [-d DIM] [-a ALGO] [-C] [-D] [-W] [-z WTA_ZTHR] [-N] [-t THRESHOLD] [-c COMPONENTS] [-p PCA_TYPE] [-S]

options:
  -h, --help            show this help message and exit
  -hh, --verbose_help   Verbose help message. Prints help message and example usages
  -O, --overwrite       Overwrites previous file structure

Set Up Arguments:
  -l LIST_OF_SUBJECTS, --list_of_subjects LIST_OF_SUBJECTS
                        Filepath to a list of subjects
  -o OUTDIR, --outdir OUTDIR
                        Path to output directory

Decomposition inputs:
  --seeds SEEDS, -s SEEDS
                        Absolute path to a text file of seed(s) used in nfact_pp/probtrackx. If used nfact_pp this is the seeds_for_decomp.txt in the nfact_pp directory.
  --roi ROI, -r ROI     Absolute path to a text file containing the absolute path ROI(s) paths to restrict seeding to (e.g. medial wall masks). This is not needed if seeds are not surfaces. If used nfact_pp then this is the roi_for_decomp.txt file in the nfact_pp
                        directory.
  -n CONFIG, --nfact_config CONFIG
                        Absolute path to a configuration file. Congifuration file provides available hyperparameters for ICA and NMF. Use nfact_config -D to create a config file. Please see sckit learn documentation for NMF and FASTICA for further details

Decomposition options:
  -d DIM, --dim DIM     This is compulsory option. Number of dimensions/components to retain after running NMF/ICA.
  -a ALGO, --algo ALGO  Which decomposition algorithm. Options are: NMF (default), or ICA. This is case insensitive

Output options:
  -C, --cifti           Option to save GM as a cifti dscalar. Seeds must be left and right surfaces with an optional nifti for subcortical structures.
  -D, --disk            Save the decompistion matricies directly to disk rather than as nii/gii files.
  -W, --wta             Option to create and save winner-takes-all maps.
  -z WTA_ZTHR, --wta_zthr WTA_ZTHR
                        Winner-takes-all threshold. Default is 0
  -N, --normalise       Convert component values into Z scores and saves map. This is useful for visualization
  -t THRESHOLD, --threshold THRESHOLD
                        Value at which to threshold Components at. Set to 0 to do no thresholding.

ICA options:
  -c COMPONENTS, --components COMPONENTS
                        Number of component to be retained following the PCA. Default is 1000
  -p PCA_TYPE, --pca_type PCA_TYPE
                        Which type of PCA to do before ICA. Options are 'pca' which is sckit learns default PCA or 'migp' (MELODIC's Incremental Group-PCA dimensionality). Default is 'pca' as for most cases 'migp' is slow and not needed. Option is case insensitive.
  -S, --sign_flip       nfact_decomp by default sign flips the ICA distribution to reduce the number of negative values. Use this option to stop the sign_flip


Basic NMF with volume seeds usage:
    nfact_decomp --list_of_subjects /absolute path/sub_list \
                 --seeds /absolute path/seeds.txt \
                 --dim 50

Basic NMF usage with surface seeds:
    nfact_decomp --list_of_subjects /absolute path/sub_list \
                 --seeds /absolute path/seeds.txt \
                 --roi /absolute path/rois
                 --dim 50

ICA with config file usage:
    nfact_decomp --list_of_subjects /absolute path/sub_list \
                 --seeds /absolute path/seeds.txt \
                 --outdir /absolute path/study_directory \
                 --algo ICA \
                 --nfact_config /absolute path/nfact_config.decomp

Advanced ICA Usage:
    nfact_decomp --list_of_subjects /absolute path/sub_list \
                 --seeds /absolute path/seeds.txt \
                 --outdir /absolute path/study_directory \
                 --algo ICA \
                 --components 1000 \
                 --pca_type mipg
                 --dim 100 \
                 --normalise \
                 --wta \
                 --wta_zthr 0.5


```
------------------------------------------------------------------------------------------------------------------------------------------

```
_   _ ______   ___   _____  _____  ______ ______
| \ | ||  ___| / _ \ /  __ \|_   _| |  _  \| ___ \
|  \| || |_   / /_\ \| /  \/  | |   | | | || |_/ /
| . ` ||  _|  |  _  || |      | |   | | | ||    /
| |\  || |    | | | || \__/\  | |   | |/ / | |\ \
\_| \_/\_|    \_| |_/ \____/  \_/   |___/  \_| \_|

```

## NFACT Dr

This is the dual regression module of NFACT. Depending on which decompostion method was used depends on which 
dual regression technique will be used. If NMF was used then non-negative least squares regression will be used, if ICA
then it will be standard regression.

### Usage
```
options:
  -h, --help            show this help message and exit
  -l LIST_OF_SUBJECTS, --list_of_subjects LIST_OF_SUBJECTS
                        REQUIRED: Filepath to a list of subjects
  -o OUTDIR, --outdir OUTDIR
                        REQUIRED: Path to output directory
  -a ALGO, --algo ALGO  REQUIRED: Which NFACT algorithm to perform dual regression on
  --seeds SEEDS, -s SEEDS
                        REQUIRED: File of seeds used in NFACT_PP/probtrackx
  --roi ROI, -r ROI     RECOMMENDED FOR SURFACE SEEDS: Txt file with ROI(s) paths to restrict seeding to (e.g. medial wall masks).
  -n NFACT_DECOMP_DIR, --nfact_decomp_dir NFACT_DECOMP_DIR
                        REQUIRED IF NFACT_DECOMP: Filepath to the NFACT_decomp directory. Use this if you have ran NFACT decomp
  -d DECOMP_DIR, --decomp_dir DECOMP_DIR
                        REQUIRED IF NOT NFACT_DECOMP: Filepath to decomposition components. WARNING NFACT decomp expects components to be named in a set way. See documentation for further info.
  -N, --normalise       normalise components by scaling
  -hh, --verbose_help   Prints help message and example usages


Dual regression usage:
    nfact_dr --list_of_subjects /path/to/nfact_config_sublist \
        --seeds /path/to/seeds.txt \
        --nfact_decomp_dir /path/to/nfact_decomp \
        --outdir /path/to/output_directory \
        --algo NMF

ICA Dual regression usage:
    nfact_dr --list_of_subjects /path/to/nfact_config_sublist \
        --seeds /path/to/seeds.txt \
        --nfact_decomp_dir /path/to/nfact_decomp \
        --outdir /path/to/output_directory \
        --algo ICA

Dual regression with roi seeds usage:
    nfact_dr --list_of_subjects /path/to/nfact_config_sublist \
        --seeds /path/to/seeds.txt \
        --nfact_decomp_dir /path/to/nfact_decomp \
        --outdir /path/to/output_directory \
        --roi /path/to/roi.txt \
        --algo NMF
```

nfact_dr is independent from nfact_decomp however, nfact_decomp expects a strict naming convention of files. If nfact_decomp has not been ran then group average files and components must all be in the same folder. Components must be named W_dim* and G_dim* with group average files named coords_for_fdt_matrix2, lookup_tractspace_fdt_matrix2.nii.gz. 

------------------------------------------------------------------------------------------------------------------------------------------

```
 _   _ ______   ___   _____  _____     ___     ____
| \ | ||  ___| / _ \ /  __ \|_   _|   / _ \   / ___|
|  \| || |_   / /_\ \| /  \/  | |    | | | | | |
| . ` ||  _|  |  _  || |      | |    | | | | | |
| |\  || |    | | | || \__/\  | |    | |_| | | |___
\_| \_/\_|    \_| |_/ \____/  \_/     \__\_\  \____|

```
## NFACT Qc

This is a quality control module that creates a number of hitmaps that can be used to check for bias in decomposition.

Each map contains the number of times that voxel/vertex appears in the decomposition. 

## Output:

Prefix:
- hitmap_*.nii.gz: Volume nii component. Components are thresholded by zscoring to remove noise
- hitmap_*_raw.nii.gz: Volume nii component. Components are not thresholded
- mask_*.nii.gz: Volume nii component. Binary mask of thresholded components
- mask_*_raw.nii.gz: Volume nii component. Binary mask of unthresholded components     
- *.gii: Surface gii component. Components are thresholded by zscoring to remove noise
- *_raw.gii: Surface gii component. Components are not thresholded   

## Usage:

```
usage: nfact_Qc [-h] [-n NFACT_FOLDER] [-d DIM] [-a ALGO] [-t THRESHOLD] [-O]

options:
  -h, --help            show this help message and exit
  -n NFACT_FOLDER, --nfact_folder NFACT_FOLDER
                        REQUIRED: Absolute path to nfact_decomp output folder. nfact_Qc folder is also saved within this folder.
  -d DIM, --dim DIM     REQUIRED: Number of dimensions/components that was used to generate nfact_decomp image
  -a ALGO, --algo ALGO  REQUIRED: Which algorithm to run qulatiy control on. Options are: NMF (default), or ICA.
  -t THRESHOLD, --threshold THRESHOLD
                        Threshold value for z scoring the number of times a component comes up in a voxel in the image. Values below this z score are treated as noise and discarded in the non raw image.
  -O, --overwrite       Overwite previous QC

```

------------------------------------------------------------------------------------------------------------------------------------------

```
 _   _______ ___  _____ _____                    __ _
| \ | |  ___/ _ \/  __ \_   _|                  / _(_)
|  \| | |_ / /_\ \ /  \/ | |     ___ ___  _ __ | |_ _  __ _
| . ` |  _||  _  | |     | |    / __/ _ \| '_ \|  _| |/ _` |
| |\  | |  | | | | \__/\ | |   | (_| (_) | | | | | | | (_| |
\_| \_|_|  \_| |_/\____/ \_/    \___\___/|_| |_|_| |_|\__, |
                                                       __/ |
                                                      |___/

```
## NFACT config

NFACT config is a util tool for nfact, that creates a variety of config files to be used in nfact.

NFACT config can create:
1) nfact_config.pipeline overview. This config JSON file is used in the nfact pipeline to have greater control over parameters.  
2) nfact_config.decomp. A config JSON file to control the hyper-parameters of the ICA and NMF functions.
3) nfact_config.sublist. A list of subjects(text file) in a folder. 


## Usage:
```
usage: nfact_config [-h] [-C] [-D] [-s SUBJECT_LIST] [-o OUTPUT_DIR] [-f FILE_NAME]

options:
  -h, --help            Shows help message and exit
  -C, --config          Creates a config file for NFACT pipeline
  -D, --decomp_only     Creates a config file for hyperparameters for the NMF/ICA
  -s SUBJECT_LIST, --subject_list SUBJECT_LIST
                        Creates a subject list from a given directory Needs path to subjects directory. If ran inside an nfact_pp directory will make a subject list for decompoisition (adds on omatrix2 to file paths)
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        File path of where to save config file
  -f FILE_NAME, --file_name FILE_NAME
                        Name of the nfact config filename. Defaults is nfact_config
```
Altering a Boolean value in a JSON is done by giving then everything has to be lower case i.e true, false. It is advised that unless you are familiar with JSON 
files to use a JSON linter to check they are valid. 

### nfact_config_pipeline.config overview

This is the config file for the nfact pipeline. Please check the individual modules for further details on arguments.

```
{
    "global_input": {
        "list_of_subjects": "Required",
        "outdir": "Required",
        "seed": [
            "Required unless file_tree specified"
        ],
        "overwrite": false,
        "pp_skip": false,
        "dr_skip": false,
        "qc_skip": false,
        "folder_name": "nfact"
    },
    "cluster": {
        "cluster": false,
        "cluster_queue": "None",
        "cluster_ram": "60",
        "cluster_time": false,
        "cluster_qos": false
    },
    
    "nfact_pp": {
        "gpu": false,
        "file_tree": false,
        "warps": [],
        "bpx_path": false,
        "roi": [],
        "seedref": false,
        "target2": false,
        "nsamples": "1000",
        "mm_res": "3",
        "ptx_options": false,
        "exclusion": false,
        "stop": false,
        "absolute": false,
        "n_cores": false
    },

    "nfact_decomp": {
        "dim": "Required",
        "algo": "NMF",
        "roi": false,
        "config": false,
        "cifti": false,
        "wta": false,
        "wta_zthr": "0.0",
        "normalise": false,
        "threshold": "3",
        "components": "1000",
        "pca_type": "pca",
        "sign_flip": true
    },
    
    "nfact_dr": {
        "normalise": false,
        "n_cores": false
    },

    "nfact_qc": {
        "threshold": "2"
    }

```

Everything that has says is required must be given. rois, warps and seed must be given in python list format like this
```
"seed": ["l_seed.nii.gz", "r_seed.nii.gz"]
```

### nfact_config_decomp.config 

This is the nfact_config_decomp.config file.

NFACT does its decomposition using sckit-learn's FastICA (https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html#sklearn.decomposition) and NFM (https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html) so any of the hyperparameters of these functions can be altered by changing the values in the JSON file.

```
{
    "ica": {
        "algorithm": "parallel",
        "whiten": "unit-variance",
        "fun": "logcosh",
        "fun_args": null,
        "max_iter": 200,
        "tol": 0.0001,
        "w_init": null,
        "whiten_solver": "svd",
        "random_state": null
    },
    "nmf": {
        "init": null,
        "solver": "cd",
        "beta_loss": "frobenius",
        "tol": 0.0001,
        "max_iter": 200,
        "random_state": null,
        "alpha_W": 0.0,
        "alpha_H": "same",
        "l1_ratio": 0.0,
        "verbose": 0,
        "shuffle": false
    }
}
```

### nfact_config_sublist

NFACT config will attempt to given a directory work out and write to a file all the subjects in that file. Though nfact will try and filter out 
folders that aren't subjects, it isn't perfect so please check the subject list. 
