# ChangeLog

## NFACT V2

This is will be the first and last new edition. Moving to rolling releases after v2

### Bug fixes
- Fixed nfact dual regression multicore processing on SLURM so dual regression multicore processing works correctly
- Fixed nfact_pp cifti filetree seeds. Seeds now always refelect that in the filetree

### New Features
- nfact_stats. New Module to creating statistical maps for PALM/randomise and component loadings
- nfact_config. Now can compress fdt_matrix2.dot to save space as lz4 files.
- nfact_decomp & nfact_dr can handle compressed lz4 fdt_matrix2.dot