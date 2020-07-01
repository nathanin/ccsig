#!/usr/bin/env python

import numpy as np
import pandas as pd
import scanpy as sc

import os
from glob import glob
import argparse

import multiprocessing

from score_genelists_def import score_genelist, adata

'''
Score lists of genes 
--------------------

Read in AnnData and reset to raw counts if possible
subset cells if needed

Read in lists 
If the list has just one member, skip it
Otherwise run scoring and create a table for single cells
'''

parser = argparse.ArgumentParser()
parser.add_argument('genelist_dir')
parser.add_argument('--out', default=None, type=str)
parser.add_argument('-j', default=6, type=int)
args = parser.parse_args()

gene_lists = sorted(glob(f'{args.genelist_dir}/*.txt'))
print(f'Loaded {len(gene_lists)} gene lists')

p = multiprocessing.Pool(args.j)
ret = p.map(score_genelist, gene_lists)
p.close()
p.join()

list_scores = {}
for d in ret:
  if d is None:
    continue
  list_scores.update(d)

x = []
cols = []
for k,v in list_scores.items():
  print(k, v.shape)
  x.append(v)
  cols.append(k)

x = np.stack(x, axis=1)
print(f'Creating scores DataFrame from x={x.shape}')

list_scores = pd.DataFrame(x, index=adata.obs_names, columns=cols)
list_scores.index.name = 'barcodes'
print(list_scores.head())

if args.out is None:
  gld = args.genelist_dir
  if gld.endswith('/'):
    gld = gld[:-1]
  df_out = gld + '_scores.csv'
else:
  df_out = args.out

print(f'Writing to {df_out}')
list_scores.to_csv(df_out, float_format='%.5f')
