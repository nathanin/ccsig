#!/usr/bin/env python

import numpy as np
import pandas as pd
import scanpy as sc

import itertools
import argparse
import pickle
import os

from utils import make_logger
from build_c_table import build_c_table

from scipy.stats import fisher_exact

"""
We're going for a command line interface like this:

./score_fisher.py adata.h5ad radata.h5ad \
  -s Cell_1 -r Cell_2 --col SubType \
  --samples Patient

so we can call it (or generate calls) with parallel

parallel --dry-run ./score_fisher.py adata.h5ad radata.h5ad -s {1} -r {2} ... ::: senders.txt ::: receivers.txt 

radata should have an entry in `uns` that is a dictionary listing
the active receptors for each subtype, i.e.:

receptor_lists = radata.uns['active_receptors']
receptor_lists['Phagocytic_Macs'] # ['FCGR1', ...]

we can probably assume these receptors are significantly expressed on 
that receiving population but in case, the default behavior is to perform the 
same expression test on receptor activity as we apply to determine ligand expression


NOTES:
- generally assume that adata and radata have a common obs, or 
  at least that celltype_col and sample columns are harmonized

- deal with 99% of the gene expression being dead weight in a very lazy way that works fine on 
  a HPC but not so good on local machines running jobs with parallel.
  . Each time the script is called, whe whole of adata_file is read , and only after 
    we learn what receptors to score can the out of scope gene columns be dropped.

    For this it might be worth looking into a random startup delay ~30s - a few mins to give
    jobs a chance to carousel the RAM 

"""

def get_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    'adata_file', 
    type = str,
    help = 'path to gene expression AnnData saved as h5ad'
  )
  parser.add_argument(
    'radata_file', 
    type = str,
    help = 'path to receptor score AnnData saved as h5ad'
  )

  parser.add_argument(
    'output_dir', 
    type = str,
    help = 'path save outputs'
  )

  parser.add_argument(
    '--clobber', 
    action = 'store_true',
    help = 'whether to overwrite existing output files'
  )

  parser.add_argument(
    '-s', 
    dest = 'sender',
    type = str,
    help = 'sending cell population'
  )
  parser.add_argument(
    '-r', 
    dest = 'receiver',
    type = str,
    help = 'sending cell population'
  )
  parser.add_argument(
    '--col', 
    type = str,
    dest = 'celltype_col',
    help = 'column of adata.obs and radata.obs to extract the celltypes'
  )

  parser.add_argument(
    '--sample_col', 
    type = str,
    dest = 'sample_col',
    help = 'column of adata.obs and radata.obs to treat as samples'
  )

  parser.add_argument(
    '--active_receptor_key',
    type = str,
    default = 'active_receptors',
    help = 'key in radata.uns'
  )

  parser.add_argument(
    '--receptor_ligand_dict',
    type = str,
    help = 'path to a pickled python dictionary where keys are receptor gene names '+\
           'and values are matched ligands'

  )

  parser.add_argument( 
    '--significance', 
    default = 0.05,
    type = float,
    help = 'pvalue to consider significant and write out receptor names'
  )
  parser.add_argument( 
    '--reps', 
    default = 1000,
    type = int,
    help = 'number of repeated draws from background to determine enriched expression'
  )
  parser.add_argument( '--verbose', action = 'store_true' )

  return parser







def main(ARGS):
  RL = pickle.load(open("cabello_aguilar_dict.pkl", "rb"))

  adata = sc.read_h5ad(ARGS.adata_file)
  radata = sc.read_h5ad(ARGS.radata_file)

  sc.pp.normalize_total(adata, target_sum=10000)

  if adata.shape[0] != radata.shape[0]:
    logger.warning('adata {adata.shape} != radata {radata.shape}')

  assert ARGS.celltype_col in adata.obs.columns
  assert ARGS.celltype_col in radata.obs.columns

  # More checks -- sender and receiver types in obs, etc.
  use_receptors = radata.uns[ARGS.active_receptor_key][ARGS.receiver]
  all_ligands = []
  for k in use_receptors:
    v = RL[k]
    for l in v:
      if l not in all_ligands: 
        all_ligands.append(l)

  adata = adata[:, adata.var_names.isin(all_ligands)]
  radata = radata[:, radata.var_names.isin(use_receptors)]
  logger.info(f'Cut down adata to relevant ligands ~ {adata.shape}')
  logger.info(f'Cut down radata to relevant receptors ~ {radata.shape}')

  outf = open(os.path.join(ARGS.output_dir, f'{ARGS.sender}__to__{ARGS.receiver}.txt'), "w+")

  for r in use_receptors:
    c_table = build_c_table(r_scores = radata, gene_expr = adata, 
                            r_group = ARGS.receiver, 
                            s_group = ARGS.sender,
                            r_samples = ARGS.sample_col,
                            s_samples = ARGS.sample_col,
                            receptor = r, 
                            ligands = RL[r],
                            r_celltypes = ARGS.celltype_col, 
                            s_celltypes = ARGS.celltype_col,
                            reps = ARGS.reps,
                            verbose = ARGS.verbose
                        )
    fisher_pval = fisher_exact(c_table, alternative='greater')[1]
    logger.info(f'{ARGS.sender} --> {ARGS.receiver} {r}: {fisher_pval:3.3e}')

    if fisher_pval < ARGS.significance:
      outf.write(f'{r}\t{fisher_pval:3.5f}\n')


  outf.close()

if __name__ == '__main__':
  parser = get_parser()

  ARGS = parser.parse_args()

  # Check as early as possible for sender == receiver stuff 
  # Decide whether or not to allow that, i don't see why not.

  logger = make_logger()

  main(ARGS)
