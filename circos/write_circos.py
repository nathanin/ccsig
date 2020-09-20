#!/usr/bin/env python

import numpy as np

'''
/home/ing/miniconda3/envs/scrna/lib/python3.7/site-packages/anndata/_core/anndata.py:1094: \
  FutureWarning: is_categorical is deprecated and will be removed in a future version.  \
    Use is_categorical_dtype instead
'''
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import scanpy as sc

import seaborn as sns
from matplotlib.colors import rgb2hex

import pickle
import argparse
import sys
import os

import logging

from write_circos_utils import *

import shutil
import pathlib
parent_path = pathlib.Path(__file__).parent.absolute()

"""
Please install Circos from the source:
http://circos.ca/ 


This script will create the data files needed to create a circos plot
displaying a set of putative interactions you've already identified in single
cell data.

If you do not provide interactions then interactions will be identified for you.

At least, there needs to be single cell resolution cell type annotations, 
and a single sender cell type with one or several receiver cell types. 

Note that this version supports drawing a single sending cell type at a time.
This is an area with potential for immediate imporvement.

We will fill in many stylistic things like color maps automatically
unless specifics are provided.

The data type of choice here are pickled dictionaries.
"""

logger = make_logger()

parser = argparse.ArgumentParser()
parser.add_argument(
  'adata', type=str, 
  help = 'path to input scSeq data (AnnData)'
)
parser.add_argument(
  '-d', '--radata', default=None, type=str, 
  help = 'path to receptor activity score AnnData'
)
parser.add_argument(
  '-i', '--interactions', default=None, type=str, 
  help = 'path to a dictionary of interactions'
)
parser.add_argument(
  '-c', '--celltypes', default=None, type=str, 
  help = 'column to use for specific cell type annotations'
)
parser.add_argument(
  '-s', '--sender', default=None, type=str, 
  help = 'the sender subtype'
)
parser.add_argument(
  '-r', '--receivers', default=None, nargs='+', type=str, 
  help = 'the receiver subtypes'
)
parser.add_argument(
  '--norm_adata', action='store_true',
  help = 'apply library size normalization to the single cells'
)
parser.add_argument(
  '--log1p', action='store_true',
  help = 'apply log1p normalization to the single cells'
)
parser.add_argument(
  '--colors', default=None, type=str, 
  help = 'a dictionary of colors'
)
parser.add_argument(
  '--palette', default='tab20', type=str, 
  help = 'a named seaborn color palette'
)
parser.add_argument(
  '--conf', default=None, type=str, 
  help = 'circos config.conf file'
)
parser.add_argument(
  '-o', '--outdir', default='circos_data', type=str, 
  help = 'base directory for saving'
)

## Options for calling ligands if none are provided upfront
parser.add_argument('-b', '--broadtypes', default=None, type=str,
                    help = 'broad cell type column')
parser.add_argument('-p', '--percent', default=0.1, type=float,
                    help = 'percent expression cutoff')
parser.add_argument('--lr_table', default=None, type=str,
                    help = 'table to load the LR pairs')
parser.add_argument('--allow_interactions', default=None, type=str,
                    help = 'list of interaction channels to allow (see note)')


ARGS = parser.parse_args()


# First, decide where to stick the output
if os.path.isdir(ARGS.outdir):
  logger.warn(f'Output directory {ARGS.outdir} already exists. Contents will be overwritten.')
else:
  logger.info(f'Creating output directory {ARGS.outdir}')
  os.makedirs(ARGS.outdir)

sh = logging.FileHandler(f'{ARGS.outdir}/log.txt',  mode='w+')
sh.setFormatter(logger.handlers[0].formatter)
logger.addHandler(sh)

for k, v in ARGS.__dict__.items():
  logger.info(f'argument: {k:>20}\t{v}')

sender = ARGS.sender

# Allow file input for receivers
if (len(ARGS.receivers) == 1) and os.path.exists(ARGS.receivers[0]):
  logger.info(f'Reading receivers from a list {ARGS.receivers[0]}')
  receivers = [l.strip() for l in open(ARGS.receivers[0], 'r')]
else:
  receivers = ARGS.receivers

adata = sc.read_h5ad(ARGS.adata)
logger.info(f'adata: {adata.shape}')
if ARGS.norm_adata:
  logger.info('Normalizing adata')
  sc.pp.normalize_total(adata, target_sum=10000)

if ARGS.log1p:
  sc.pp.log1p(adata)

if ARGS.radata is not None:
  radata = sc.read_h5ad(ARGS.radata)
  logger.info(f'radata: {radata.shape}')
else:
  radata = None
  logger.info(f'radata: None')



# ----------------------- Get some interactions to plot

if ARGS.interactions is None:
  logger.info('No interactions provided, try looking for some...')
  assert ARGS.lr_table is not None
  lr_table = pd.read_csv(ARGS.lr_table, index_col=0, header=0)
  ligands = np.unique(lr_table.keys())

  if ARGS.allow_interactions is not None:
    allow_interactions = [l.strip() for l in open(ARGS.allow_interactions, 'r')]
  else:
    allow_interactions = None

  interactions = get_interactions(adata, 
                                  adata if radata is None else radata, 
                                  sender, receivers, ARGS.percent, 
                                  ARGS.celltypes, ARGS.broadtypes, 
                                  lr_table, allow_interactions)
else:
  interactions = pickle.load(open(ARGS.interactions, 'rb'))


# ----------------------- Get all receptors from the LR list

all_ligands = set()
all_receptors = set()
total_channels = 0
for k, vv in interactions.items():
  for v in vv:
    ligand, receptor = v.split('_')
    all_ligands = all_ligands.union(set([ligand]))
    all_receptors = all_receptors.union(set([receptor]))
    total_channels += 1

logger.info(f'Drawing {total_channels} total_channels')
        
sd = adata[adata.obs[ARGS.celltypes] == sender, adata.var_names.isin(list(all_ligands))]
# # Percent of cells with nonzero ligand expression in sender
# sdx = pd.DataFrame(sd.X.toarray() > 0, index=sd.obs_names, columns=sd.var_names)
# sdx['subtype'] = sd.obs.SubType_v3
# sdx = sdx.groupby('subtype').mean().loc[[sender]]
# logger.info(f'Ligand percent positive: {sdx.shape}')

# Mean ligand expression in sender 
sdxe = pd.DataFrame(sd.X.toarray(), index=sd.obs_names, columns=sd.var_names)
sdxe['subtype'] = sd.obs[ARGS.celltypes]
sdxe = sdxe.groupby('subtype').mean().loc[[sender]]
logger.info(f'Ligand expression: {sdxe.shape}')


if radata is not None:
  rd = radata[:, radata.var_names.isin(list(all_receptors))]
else:
  rd = adata[:, adata.var_names.isin(list(all_receptors))]
# # Percent of cells with non-zero receptor expression
# rdx = pd.DataFrame(rd.X.toarray() > 0, index=rd.obs_names, columns=rd.var_names)
# rdx['subtype'] = rd.obs.SubType_v3
# rdx = rdx.groupby('subtype').mean()
# rdx = rdx.loc[receivers, :]
# logger.info(f'Receptor percents: {rdx.shape}')

# Mean receptor score
rdxe = pd.DataFrame(rd.X.toarray(), index=rd.obs_names, columns=rd.var_names)
rdxe['subtype'] = rd.obs[ARGS.celltypes]
rdxe = rdxe.groupby('subtype').mean()
rdxe = rdxe.loc[receivers, :]
logger.info(f'Receptor means: {rdxe.shape}')

## Receptor/ligand heatmap colors
# reds = [tuple(int(ch * 255) for ch in c) for c in sns.color_palette('Reds', 10)]
# blues = [tuple(int(ch * 255) for ch in c) for c in sns.color_palette('Blues', 10)]

f = open(f'{ARGS.outdir}/karyotype.txt', 'w+')
hlf = open(f'{ARGS.outdir}/highlights.txt', 'w+')
txtf = open(f'{ARGS.outdir}/genes.txt', 'w+')
linkf = open(f'{ARGS.outdir}/links.txt', 'w+')

TOTAL_TICKS = 10000
SEMI_CIRCLE = int(TOTAL_TICKS / 2)

if ARGS.colors is None:
  n_colors = len(receivers) + 1 # TODO allow more than 1 sender
  colors = sns.color_palette(ARGS.palette, 20)
  # np.random.shuffle(colors)
  color_palette = {s: rgb2hex(c) for s, c in zip([sender]+receivers, colors[:n_colors])}

try:
  ligand_order, receptor_coords = write_receptor_karyotype(interactions, rdxe, f, hlf, txtf, 0, 
                                                           SEMI_CIRCLE, color_palette)

  write_sender_karyotype(f, sender, SEMI_CIRCLE, TOTAL_TICKS, color_palette)
  ligand_coords = write_ligands(sdxe, sender, hlf, txtf, SEMI_CIRCLE, TOTAL_TICKS, ligand_order)

  receiver_order = list(interactions.keys())[::-1]
  draw_links(interactions, sender, linkf, sdxe, rdxe, ligand_coords, 
             receptor_coords, color_palette, receiver_order=receiver_order)

except:
  logger.exception()

finally:
  f.close()
  hlf.close()
  txtf.close()
  linkf.close()

# Copy the conf and build files to the newly created output dir
if (ARGS.conf is None) or not os.path.exists(ARGS.conf):
  src = f'{parent_path}/config.conf'
else:
  src = ARGS.conf

dst = f'{ARGS.outdir}/config.conf'
logger.info(f'Copying config file {src} --> {dst}')
shutil.copyfile(src, dst)

# call build
logger.info('Calling circos')
os.chdir(ARGS.outdir)
basedir = os.path.basename(os.path.normpath(ARGS.outdir))
os.system(f'circos -conf config.conf -outputfile {basedir}.svg -debug_group textplace')

