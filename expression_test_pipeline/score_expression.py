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
from association_test import group_values, association_test

"""
We're going for a command line interface like this:

./score_expression.py adata.h5ad radata.h5ad \
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

  # parser.add_argument( 
  #   '--reps', 
  #   default = 1000,
  #   type = int,
  #   help = 'number of repeated draws from background to determine enriched expression'
  # )
  parser.add_argument( '--verbose', action = 'store_true' )

  return parser







def main(ARGS):
  RL = pickle.load(open(ARGS.receptor_ligand_dict, "rb"))

  adata = sc.read_h5ad(ARGS.adata_file)
  radata = sc.read_h5ad(ARGS.radata_file)
  all_samples = np.unique(adata.obs[ARGS.sample_col])
  logger.info(f'Working with samples {all_samples}')

  logger.info('Count normalize and log the expression data')
  sc.pp.normalize_total(adata, target_sum=10000)
  sc.pp.log1p(adata)

  if adata.shape[0] != radata.shape[0]:
    logger.warning('adata {adata.shape} != radata {radata.shape}')

  assert ARGS.celltype_col in adata.obs.columns
  assert ARGS.celltype_col in radata.obs.columns

  use_receptors = list(radata.var_names)

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

  # In this setup we can also discard other celltypes:
  adata = adata[adata.obs[ARGS.celltype_col] == ARGS.sender]
  radata = radata[radata.obs[ARGS.celltype_col] == ARGS.receiver]
  logger.info(f'Cut down adata to the sending population only ~ {adata.shape}')
  logger.info(f'Cut down radata to the receiving population only ~ {radata.shape}')

  outf = open(os.path.join(ARGS.output_dir, f'{ARGS.sender}__to__{ARGS.receiver}.txt'), "w+")

  # Each subytpe gets its own grouping vector
  groupby_gex = np.array(adata.obs[ARGS.sample_col])
  groupby_r = np.array(radata.obs[ARGS.sample_col])

  grouped_gex = group_values(adata.X.toarray(), groupby_gex, all_samples, agg='sum')
  grouped_rscore = group_values(radata.X.toarray(), groupby_r, all_samples)
  logger.info(f'Grouped gene expression {grouped_gex.shape}')
  logger.info(f'Grouped receptor scores {grouped_rscore.shape}')


  n_hits , total_tests = 0, 0
  for r in use_receptors:
    # rscore = np.squeeze(radata.X[:, radata.var_names == r].toarray())
    # rscore = group_values(rscore, groupby_r, all_samples)
    rscore = grouped_rscore[:, radata.var_names.values == r]

    # Test all ligands for this receptor independently
    r_ligands = RL[r]
    for l in r_ligands:
      if l not in adata.var_names:
        logger.warning(f'Ligand {l} not found in adata var_names')
        continue
      # gex = np.squeeze(adata.X[:, adata.var_names==l].toarray())
      # gex = group_values(gex, groupby_gex, all_samples, agg='sum')
      total_tests += 1
      gex = grouped_gex[:, adata.var_names.values == l]
      pval, message = association_test(rscore, gex)

      logger.info(f'{ARGS.sender} ({l})\t-->\t{ARGS.receiver} ({r}):\t{pval:3.3e}\t{message}')

      if pval < ARGS.significance:
        outf.write(f'{l}\t{r}\t{pval:3.5f}\n')
        n_hits += 1

  outf.close()
  logger.info(f'Finished with {n_hits}/{total_tests} predicted interactions')



if __name__ == '__main__':
  parser = get_parser()

  ARGS = parser.parse_args()

  # Check as early as possible for sender == receiver stuff 
  # Decide whether or not to allow that, i don't see why not.

  logger = make_logger()

  # Check for output file existence, decide to continue or not
  outf = os.path.join(ARGS.output_dir, f'{ARGS.sender}__to__{ARGS.receiver}.txt')
  if os.path.exists(outf):
    if not ARGS.clobber:
      logger.info(f'Output {outf} found. Exiting. Set --clobber to continue anyway')
      sys.exit(1)

  logger.info('Starting')
  for k, v in ARGS.__dict__.items():
    logger.info(f'{k}\t{v}')

  delay = np.random.randint(low=ARGS.delay_start_interval[0], high=ARGS.delay_start_interval[1])
  logger.info(f'Delay startup for {delay} seconds')
  time.sleep(delay)

  main(ARGS)

