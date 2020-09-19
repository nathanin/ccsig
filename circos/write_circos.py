#!/usr/bin/env python

import numpy as np
import pandas as pd
import scanpy as sc

import seaborn as sns

import pickle
import argparse
import os

import logging

from write_circos_utils import *

"""
Please install Circos from the source:
http://circos.ca/ 


This script will create the data files needed to create a circos plot
from a set of putative interactions you've already identified in single
cell data.

If you do not provide interactions then interactions will be identified for you.

At least there needs to be single cell resolution cell type annotations, 
and a __single__ sender cell type with one or several receiver cell types. 

Note that this version supports a single sending cell type at a time.
This is an area with potential for immediate imporvement.

We will fill in many stylistic things like color maps automatically
unless speciics are provided.

The data type of choice here are pickled dictionaries.
"""

logger = make_logger()

parser = argparse.ArgumentParser()
parser.add_argument('adata', type=str,
                    help = 'path to input scSeq data (AnnData)')
parser.add_argument('-d', '--radata', default=None, type=str,
                    help = 'path to receptor activity score AnnData')
parser.add_argument('-i', '--interactions', default=None, type=str,
                    help = 'path to a dictionary of interactions')
parser.add_argument('-c', '--celltypes', default=None, type=str,
                    help = 'column to use for cell type annotations')
parser.add_argument('-s', '--sender', default=None, type=str,
                    help = 'the sender subtype')
parser.add_argument('-r', '--receivers', default=None, nargs='+', type=str,
                    help = 'the receiver subtypes')
parser.add_argument('-p', '--percent', default=0.1, type=float,
                    help = 'percent expression cutoff')

parser.add_argument('-o', '--outdir', default='circos_data', type=str,
                    help = 'base directory for saving')
# parser.add_argument('-n', '--ntop', default=25, type=int,
#                     help = 'number of top receptors to allow, per receiver')

parser.add_argument('--colors', default=None, type=str,
                    help = 'a dictionary of colors')
parser.add_argument('--allow_interactions', default=None, type=str,
                    help = 'list of interaction channels to allow (see note)')

ARGS = parser.parse_args()



for k, v in ARGS.__dict__.items():
  logger.info(f'argument: {k}\t{v}')

adata = sc.read_h5ad(ARGS.adata)
logger.info(f'adata: {adata.shape}')
if ARGS.radata is not None:
  radata = sc.read_h5ad(ARGS.radata)
  logger.info(f'radata: {radata.shape}')
else:
  radata = None
  logger.info(f'radata: None')


if ARGS.interactions is None:
  pass
  # interactions = get_interactions(adata, radata, sender, receivers, percent, subytpe_col)
else:
  interactions = pickle.load(open(ARGS.interactions, 'rb'))

logger.info(f'Drawing {len(interactions)} interactions')


# Decide where to stick the output
if os.path.isdir(ARGS.outdir):
  logger.warn(f'Output directory {ARGS.outdir} already exists. Contents will be overwritten.')
else:
  logger.info(f'Creating output direcotyr {ARGS.outdir}')
  os.makdirs(ARGS.outdir)



# ----------------------- Get all receptors from the LR list -----------------------

all_ligands = set()
all_receptors = set()
for k, vv in interactions.items():
  for v in vv:
    ligand, receptor = v.split('_')
    all_ligands = all_ligands.union(set([ligand]))
    all_receptors = all_receptors.union(set([receptor]))
        
sd = adata[adata.obs.BroadCellType_v2.isin(background_set), adata.var_names.isin(list(all_ligands))]
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

reds = [tuple(int(ch * 255) for ch in c) for c in sns.color_palette('Reds', 10)]
blues = [tuple(int(ch * 255) for ch in c) for c in sns.color_palette('Blues', 10)]


f = open(f'{outdir}/karyotype.txt', 'w+')
hlf = open(f'{outdir}/highlights.txt', 'w+')
txtf = open(f'{outdir}/genes.txt', 'w+')
linkf = open(f'{outdir}/links.txt', 'w+')

TOTAL_TICKS = 10000
SEMI_CIRCLE = int(TOTAL_TICKS / 2)

ligand_order, receptor_coords = write_receptor_karyotype(interactions, f, hlf, txtf, 0, 
  SEMI_CIRCLE)

# write the sending semi-circle
color = ','.join([f'{v}' for v in hex2rgb(master_palette[sender])])
line = f'chr - {sender} {sender} {semi_circle} {total_ticks} {color}\n'
logger.info(line)
f.write(line)

ligand_coords = write_ligands(hlf, txtf, semi_circle, total_ticks, ligand_order)

receiver_order = list(interactions.keys())[::-1]
draw_links(interactions, linkf, sdxe, rdxe, ligand_coords, 
           receptor_coords, receiver_order=receiver_order)

f.close()
hlf.close()
txtf.close()
linkf.close()
