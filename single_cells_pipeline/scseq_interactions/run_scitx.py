#!/usr/bin/env python

import numpy as np
import pandas as pd
import scanpy as sc

import ray

import argparse

from itx_handler import itxHandler

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--lrmat', type=str, required=True)

  parser.add_argument('--adata', type=str, required=True)
  parser.add_argument('--outdir', type=str, required=True)
  parser.add_argument('--groupby', type=str, required=True)
  parser.add_argument('-j', type=int, default=3)

  ARGS = parser.parse_args()

  ad = sc.read_h5ad(ARGS.adata)
  sc.pp.normalize_total(ad, target_sum=10000)
  sc.pp.log1p(ad)

  s = np.load(ARGS.lrmat, allow_pickle=True)
  lr_matrix = s['lr_matrix']
  ligands = s['ligands']
  receptors = s['receptors']
  
  scitx = itxHandler()
  scitx.create(ad, lr_matrix, ligands, receptors)
  print(scitx)

  ray.init(num_cpus=ARGS.j)
  scitx.calc(ARGS.groupby, ARGS.outdir)