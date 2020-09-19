```
usage: interaction_score_run.py [-h] [-o OUTDIR] [-j N_JOBS]
                                adata_path radata_path groupby constraint
                                ligand_file

positional arguments:
  adata_path            AnnData object for ligand expression, loaded into
                        adata
  radata_path           AnnData object for receptor score, loaded into radata
  groupby               Column common to adata.obs and radata.obs, usually
                        corresponding to cell phenotypes
  constraint            Column common to adata.obs and radata.obs, denotes
                        groups of cells to constrain the interactions
  ligand_file           path to a pickled dictionary holding the ligand pairs
                        with receptors as the keys and ligands are stored in a
                        list

optional arguments:
  -h, --help            show this help message and exit
  -o OUTDIR, --outdir OUTDIR
  -j N_JOBS, --n_jobs N_JOBS
```