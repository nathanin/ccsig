#!/usr/bin/env python

import numpy as np
import pandas as pd
import scanpy as sc
import os

import pickle
import logging

# import multiprocessing
from argparse import ArgumentParser
from interaction_score_def import interaction_test_base

"""
Process a single ligand-receptor interaction in a set of cells.
the reason to spin this out is to have a standalone function
that can be qsubbed, to process longer permutations -- to estimate
p more accurately.
"""

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
  '-g', '--groupby',
  help='Column common to adata.obs and radata.obs, usually corresponding to cell phenotypes'
)
parser.add_argument(
  '-c', '--constraint',
  help='Column common to adata.obs and radata.obs, denotes groups of cells to '+\
       'constrain the interactions, e.g. samples'
)
parser.add_argument(
  '--ligand',
  help='name of the ligand to test. must exist in adata.var_names'
)
parser.add_argument(
  '--receptor',
  help='name of the ligand to test. must exist in adata.var_names'
)
parser.add_argument(
  '--use_receptor_expression',
  default = False,
  action = 'store_true',
  help = 'whether to ignore `radata_path` and use receptor expression instead of ' +\
         'receptor scores. Make sure to still give a dummy input for `radata_path`, any string will do.'
)
parser.add_argument('-o', '--outdir')
parser.add_argument('-p', '--permutations', default=1000, type=int)
parser.add_argument('--signif', default=0.05, type=float, 
  help = 'signifcance level' 
)
ARGS = parser.parse_args()

if not os.path.isdir(ARGS.outdir):
  os.makedirs(ARGS.outdir, exist_ok=True)

logger = logging.getLogger('ITX POTENTIAL')
logger.setLevel('INFO')
ch = logging.StreamHandler()
fh = logging.FileHandler(f'{ARGS.outdir}/{ARGS.receptor}_{ARGS.ligand}log.txt', 'w+')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)


for k, v in ARGS.__dict__.items():
  logger.info(f'{k}:\t{v}')


# // loading data
# Receptor ligand annotations
rl_pairs = pickle.load(open( ARGS.ligand_file, "rb" ))

# ligand adata with gene expression
adata = sc.read_h5ad(ARGS.adata_path)

logger.info(f'Count normalize ligand expression...')
sc.pp.normalize_total(adata, target_sum=10000)

# Do this to probably reduce memory 
sc.pp.filter_genes(adata, min_cells=10)
logger.info(f'Dropped genes observed in < 10 cells: {adata.shape}')

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


# // All of these will go to shared memory
adata_in = adata.X
adata_var = np.array(adata.var_names)

yL = np.array(adata.obs[ARGS.groupby])
constraints_L = np.array(adata.obs[ARGS.constraint])

# ------- Take care of receptor expression instead of score
if radata is None:
  r_adata_in = None
  r_adata_var = adata_var.copy()
  yR = yL
  constraints_R = constraints_L
else:
  r_adata_in = radata.X
  r_adata_var = np.array(radata.var_names) # same as our list of receptors above
  yR = np.array(radata.obs[ARGS.groupby])
  constraints_R = np.array(radata.obs[ARGS.constraint])

logger.info('Running...')
outf = interaction_test_base(adata_in, r_adata_in, 
                             adata_var, r_adata_var,
                             yL, yR,
                             constraints_L, constraints_R,
                             ligand=ARGS.ligand, receptor=ARGS.receptor, 
                             permutations = ARGS.permutations,
                             sig_level = ARGS.signif,
                             outdir=ARGS.outdir)

logger.info(f'Finished: {outf}')