```
usage: score_genelists_run.py [-h] [--out OUT] [-j N_JOBS]
                              genelist_dir adata_path groupby

positional arguments:
  genelist_dir          A directory containing gene lists in newline delimited
                        *.txt format
  adata_path            AnnData object holding gene expression
  groupby               A column in adata.obs to use as cell groups for
                        background gene expression

optional arguments:
  -h, --help            show this help message and exit
  --out OUT             an h5ad file to stash results. if not specified, it
                        defaults to converting the genelist_dir path into a
                        file name
  -j N_JOBS, --n_jobs N_JOBS
                        Number of parallel jobs to launch. default=8
```