#!/usr/bin/env python

import numpy as np
import pandas as pd
import scanpy as sc
import os

import pickle
import logging

import ray
# import multiprocessing
from argparse import ArgumentParser
from interaction_score_def import interaction_test

parser = ArgumentParser()
parser.add_argument(
  'adata_path',
  help='AnnData object for ligand expression, loaded into adata'
)
parser.add_argument(
  'radata_path',
  help='AnnData object for receptor score, loaded into radata'
)
parser.add_argument(
  'groupby',
  help='Column common to adata.obs and radata.obs, usually corresponding to cell phenotypes'
)
parser.add_argument(
  'constraint',
  help='Column common to adata.obs and radata.obs, denotes groups of cells to '+\
       'constrain the interactions, e.g. samples'
)
parser.add_argument(
  'ligand_file',
  help='path to a pickled dictionary holding the ligand pairs with receptors as the keys '+\
       'and ligands are stored in a list'
)
parser.add_argument(
  '--use_receptor_expression',
  default = False,
  action = 'store_true',
  help = 'whether to ignore `radata_path` and use receptor expression instead of ' +\
         'receptor scores. still give a dummy input for `radata_path`.'
)
parser.add_argument('-o', '--outdir')
parser.add_argument('-j', '--n_jobs', default=4, type=int)
parser.add_argument('-p', '--permutations', default=100, type=int)
ARGS = parser.parse_args()

logger = logging.getLogger('ITX POTENTIAL')
logger.setLevel('INFO')
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

for k, v in ARGS.__dict__.items():
  logger.info(f'{k}:\t{v}')

if not os.path.isdir(ARGS.outdir):
  os.makedirs(ARGS.outdir)

# // loading data
# Receptor ligand annotations
rl_pairs = pickle.load(open( ARGS.ligand_file, "rb" ))

# ligand adata with gene expression
adata = sc.read_h5ad(ARGS.adata_path)
# Do this to probably reduce memory 
sc.pp.filter_genes(adata, min_cells=10)

# ------- Take care of receptor expression instead of score
if not ARGS.use_receptor_expression:
  radata = sc.read_h5ad(ARGS.radata_path)
  receptors = list(radata.var_names)
  receptors = [r for r in receptors if r in rl_pairs.keys()]
elif ARGS.use_receptor_expression:
  radata = None
  receptors = [r for r in list(adata.var_names) if r in rl_pairs.keys()]

logger.info(f'Ligand expression dataset: {adata.shape}')
# logger.info(f'Receptor score dataset: {radata.shape}')
logger.info(f'Got {len(receptors)} with paired annotated ligands')

logger.info(f'Count normalize and ligand expression...')
sc.pp.normalize_total(adata, target_sum=10000)
# sc.pp.log1p(adata)

logger.info('Starting ray')
ray.init(num_cpus=ARGS.n_jobs)


# // All of these will go to shared memory
adata_in = adata.X
adata_var = np.array(adata.var_names)

yL = np.array(adata.obs[ARGS.groupby])
constraints_L = np.array(adata.obs[ARGS.constraint])

# ------- Take care of receptor expression instead of score
if radata is None:
  r_adata_in = None
  r_adata_var = adata_var
  yR = yL
  constraints_R = constraints_L
else:
  r_adata_in = radata.X
  r_adata_var = np.array(radata.var_names) # same as our list of receptors above
  yR = np.array(radata.obs[ARGS.groupby])
  constraints_R = np.array(radata.obs[ARGS.constraint])

# // make data available to multiple workers
adata_in_id = ray.put(adata_in)
r_adata_in_id = ray.put(r_adata_in)
adata_var_id = ray.put(adata_var)
r_adata_var_id = ray.put(r_adata_var)
yL_id = ray.put(yL)
yR_id = ray.put(yR)
constraints_L_id = ray.put(constraints_L)
constraints_R_id = ray.put(constraints_R)


# // construct the compute tasks to be distributed
futures = []
interaction_channels = []
for r in receptors:
  ligands = rl_pairs[r]
  # Here, filter ligands for matching columns in the gene expression
  ligands = [l for l in ligands if l in adata_var]
  for l in ligands:
    job_id = interaction_test.remote(adata_in_id, r_adata_in_id, 
                                     adata_var_id, r_adata_var_id,
                                     yL_id, yR_id,
                                     constraints_L_id, constraints_R_id,
                                     permutations = ARGS.permutations,
                                     ligand=l, receptor=r, outdir=ARGS.outdir)
    interaction_channels.append(f'{l}__{r}')
    futures.append(job_id)

logger.info(f'Set up {len(futures)} jobs')
logger.info('Running...')
ret = ray.get(futures)

logger.info(f'Finished: {len(ret)}')

