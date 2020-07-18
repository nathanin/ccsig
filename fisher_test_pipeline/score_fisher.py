#!/usr/bin/env python

import numpy as np
import pandas as pd
import scanpy as sc

import itertools
import argparse
import pickle
import time
import sys
import os

from utils import make_logger
from build_c_table import build_c_table, build_expressed_table

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
    '--all_receptors',
    action = 'store_true',
    help = 'run all annotated receptors. ignores active_receptor_key'
  )

  parser.add_argument(
    '--separate_ligands',
    action = 'store_true',
    help = 'run each ligand separately'
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
    '--delay_start_interval',
    nargs = '+',
    default = [5, 30],
    help = 'I cant believe am doing this. Delay startup (data loading) by a random number '+\
           'of seconds to avoid memory issues where the front end of the script blows up '+\
           'in memory upon loading the full adata, then drops significantly once we drop '+\
           'unused genes/receptors.'
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
  RL = pickle.load(open(ARGS.receptor_ligand_dict, "rb"))

  adata = sc.read_h5ad(ARGS.adata_file)
  radata = sc.read_h5ad(ARGS.radata_file)

  sc.pp.normalize_total(adata, target_sum=10000)

  if adata.shape[0] != radata.shape[0]:
    logger.warning('adata {adata.shape} != radata {radata.shape}')

  assert ARGS.celltype_col in adata.obs.columns
  assert ARGS.celltype_col in radata.obs.columns

  # More checks -- sender and receiver types in obs, etc.
  if ARGS.all_receptors:
    use_receptors = list(radata.var_names)
  else:
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

  logger.info('Performing permutation test on all receptors and ligands')
  r_table, l_table = build_expressed_table(r_scores = radata, gene_expr = adata, 
                                   r_group = ARGS.receiver, 
                                   s_group = ARGS.sender,
                                   r_samples = ARGS.sample_col,
                                   s_samples = ARGS.sample_col,
                                   r_celltypes = ARGS.celltype_col, 
                                   s_celltypes = ARGS.celltype_col,
                                   reps = ARGS.reps,
                                   verbose = ARGS.verbose
                                   )
  del radata, adata
  r_nonzero = (r_table.values > 0).sum()
  l_nonzero = (l_table.values > 0).sum()
  logger.info(f'Got receptor expression table: {r_table.shape} with {r_nonzero} '+\
              f'({r_nonzero/np.prod(r_table.shape)}) nonzero entries')
  logger.info(f'Got ligand expression table: {l_table.shape} with {l_nonzero} '+\
              f'({l_nonzero/np.prod(l_table.shape)}) nonzero entries')

  n_hits = 0
  for r in use_receptors:
    r_ligands = RL[r]
    for l in r_ligands:
      if l not in l_table.columns:
        logger.warning(f'Ligand {l} not found in adata var_names')
        continue

      # c_table, r_data, l_data = build_c_table(r_scores = radata, gene_expr = adata, 
      #                                         r_group = ARGS.receiver, 
      #                                         s_group = ARGS.sender,
      #                                         r_samples = ARGS.sample_col,
      #                                         s_samples = ARGS.sample_col,
      #                                         receptor = r, 
      #                                         ligands = [l],
      #                                         r_celltypes = ARGS.celltype_col, 
      #                                         s_celltypes = ARGS.celltype_col,
      #                                         reps = ARGS.reps,
      #                                         verbose = ARGS.verbose
      #                                         )

      c_table = build_c_table(r_table, l_table, r, l)
      fisher_pval = fisher_exact(c_table, alternative='greater')[1]
      logger.info(f'{ARGS.sender} ({l}) --> {ARGS.receiver} ({r}): {fisher_pval:3.3e}')

      if fisher_pval < ARGS.significance:
        n_hits += 1
        outf.write(f'{l}\t{r}\t{fisher_pval:3.5f}\n')

  outf.close()
  logger.info(f'Finished. Got {n_hits} hits')



if __name__ == '__main__':
  parser = get_parser()

  ARGS = parser.parse_args()

  # Check as early as possible for sender == receiver stuff 
  # Decide whether or not to allow that, i don't see why not.

  logger = make_logger()

  # Check for output file existence, decide to continue or not
  r_data_f = os.path.join(ARGS.output_dir, f'{ARGS.sender}__to__{ARGS.receiver}_R.csv')
  if os.path.exists(r_data_f):
    if not ARGS.clobber:
      logger.info(f'Output {r_data_f} found. Exiting. Set --clobber to continue anyway')
      sys.exit(1)

  logger.info('Starting')
  for k, v in ARGS.__dict__.items():
    logger.info(f'{k}\t{v}')

  delay = np.random.randint(low=ARGS.delay_start_interval[0], high=ARGS.delay_start_interval[1])
  logger.info(f'Delay startup for {delay} seconds')
  time.sleep(delay)

  main(ARGS)

