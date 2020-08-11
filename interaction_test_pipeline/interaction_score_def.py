#!/usr/bin/env python

import numpy as np
import pandas as pd
# import scanpy as sc
import pickle
import sys
import os

# import scrna
import ray
import time
import itertools
from tqdm import tqdm
from argparse import ArgumentParser
import logging

# import multiprocessing
# from nichenetpy import plot_interactions
from scipy.stats.mstats import rankdata
from interaction_util import group_cells, calc_interactions

# import logging

"""
Implements a function that calculates significant (i.e. enriched) interactions
along a given interaction channel

Read from a file containing receptor scores so as to avoid 
copying a large dense matrix unnecessarily to many workers.

Keep a full copy of adata and pass it around 

Each time the function is called, we have to assess the significance of
each interaction, within each sample. 
As in Vento-Tormo, et al. (2018), generate a null distribution by 
assessing interactions on shuffled data. 

Report interactions that pass a significance cutoff.

"""


def permute_labels(labels, constraints):
  """
  Find a permutation for labels grouped by categorical constraints
  i.e. Shuffle labels with a sample.

  If:
    labels = [1,2,3,4,5,6]
    constraints = [1,1,1,2,2,2]

  A valid permutation might be
    perm = [2,3,1,5,6,4]

  but not
    perm = [6,5,1,2,3,4]
  """
  u_constraints = np.unique(constraints)
  perm = np.zeros_like(labels)

  for u in u_constraints:
    ix = constraints == u
    l_ix = labels[ix]
    np.random.shuffle(l_ix)
    perm[ix] = l_ix

  return perm



# def _process_permutation(r_adata_in, adata_in, receptor, labels, samples, interaction_kw_args):
def _process_permutation(xL, xR, P, yL, yR, constraints_L, constraints_R, m_y_groups, min_cells):
  # # We need this because the random state in each thread is initialized identically
  # Maybe salt the time with process id, or some unique offset 
  # pid = multiprocessing.current_process()._identity[0]
  seed = int(time.time())
  np.random.seed(seed)

  shuffled_y_L = permute_labels(yL, constraints_L)
  shuffled_y_R = permute_labels(yR, constraints_R)

  m_y_L = np.array([f'{m};{y}' for m, y in zip(constraints_L, yL)])
  m_y_R = np.array([f'{m};{y}' for m, y in zip(constraints_R, yR)])

  L = group_cells(xL, m_y_L, u_y=m_y_groups, min_cells=min_cells)
  R = group_cells(xR, m_y_R, u_y=m_y_groups, min_cells=min_cells)

  I_perm = calc_interactions(R, L, P)

  return I_perm



def calc_pvals(I_test, null_interactions):

  # Stick the test value on the back of the null distribution
  null_stacked = np.concatenate(
    [null_interactions, np.expand_dims(I_test.values, -1)],
    axis = -1
  )

  # Flip the order because we want high values to be low-rank
  ranks = rankdata(-null_stacked, axis=-1)

  # Grab p-values
  pvals = ranks[:,:,-1] / ranks.shape[-1]
  pvals = pd.DataFrame(pvals, index=I_test.index, columns=I_test.columns)

  return pvals





# def interaction_test(receptor, return_null_data=False, permutations=50, 
#                          min_cells=10, threads=4, verbose=False):

@ray.remote
def interaction_test(adata_in, r_adata_in, 
                     adata_var, r_adata_var,
                    # # just feed in the annotation vectors now; no more pulling from the adata.obs
                     yL, yR,
                     constraints_L,
                     constraints_R,
                    # --- Everything above should be put in ray shared memory ---
                     ligand, receptor, 
                    #  groupby, constraints, 
                     return_null_data=False, permutations=50, sig_level=0.5,
                     min_cells=10, 
                     outdir=None,
                    #  threads=4, 
                    #  ALL_LIGANDS=None, ALL_RECEPTORS=None, 
                     calculate_pvals=False,
                     verbose=False):
  """
  Run an interaction test between receptor/ligands

  Annoyingly, some inputs come in kind of pairs as they are repeated for sending and receiving cells
    This is to take care of the situation where there might be slight differences in the cells included,
    or an annotation mis-match ? 

  adata_in and r_adata_in ~ scipy.sparse
  adata_var, r_adata_var ~ np.ndarrays for the position of vars along columns of adata_in and r_adata_in
  receptor and ligand ~ strings (no groups of LR pairs yet.. although the machinery is in place)
  yR, yL ~ np.ndarray representing the celltype/"groupby" annotation
  constraints_R, constraints_L ~ np.ndarray representing the sample/cell bucket annotation


  This function is a bit of a bear. Given a whole dataset made up of possible multiple
  samples, we want to find the interaction potential on the specified channel between
  each annotated celltype -- given in groupby -- within each patient (constraints)

  constraints are another annotation that act like buckets of cells within which 
    it makes sense to ask about interacting sub-populations.

  Say the inputs are M=4 buckets (samples/patients/plates/whatever) and the groupby cell annotation
  has Y=5 subtypes to ask about:

  We want to always return a matrix with size (M*Y x M*Y) = (20 x 20)
  That is, even when if particular cell bucket is missing cells of some annotation,
  we will fill in 0's.

  To make ray happy we need to pass in sparase matrices and np.ndarrays for the annotation
  of groupby and constraints, which should be PUT into shared memory before calling this function.


  Benchmarked with ~ 68k cells in 25 * 18 = 450 cell type buckets, 50 permutations --> 50s elapsed

  """

  logging.basicConfig(level='INFO')

  tstart = time.time()
  # ligands = rl_pairs[receptor]

  # we can also split ligands up and handle them individually
  # ligands = np.array(ligands)

  # logging.info(f'Ligand: {ligand} receptor: {receptor}')

  # Filter ligands for those included in the master LIGANDS list
  # ligands = ligands[pd.Series(ligands).isin(ALL_LIGANDS)]

  # Build P -- this is the trivial case, we want to expand this to the M receptor x N ligands 
  # general case at some point
  P = np.ones(shape=(1, 1), dtype='double')

  # Groupby array for receptor data and ligand data separately
  # yR = np.array(r_adata_in.obs[groupby])
  # yL = np.array(adata_in.obs[groupby])
  u_y = list(set(np.unique(yR)).union(set(np.unique(yL)))) # "Y"
  # constraints_R = np.array(r_adata_in.obs[constraints])
  # constraints_L = np.array(adata_in.obs[constraints])
  u_constraints = list(set(np.unique(constraints_R)).union(set(np.unique(constraints_L)))) # "M"

  logging.info(f'{ligand} {receptor} Working on {len(u_y)} groups within {len(u_constraints)} cell buckets')
  
  # m_y_groups acts as a scaffold, helping us keep track of indices throughout
  m_y_groups = np.array([f'{m};{y}' for m, y in itertools.product(u_constraints, u_y)])
  m_y_R = np.array([f'{m};{y}' for m, y in zip(constraints_R, yR)])
  m_y_L = np.array([f'{m};{y}' for m, y in zip(constraints_L, yL)])


  # working with dense arrays...
  # Sum all the expression for each ligand per cell... (NOTE: this precludes the P matrix.)
  # xL = np.sum(adata_in.X[:, adata_in.var_names.isin(ligands)].toarray(), axis=1, keepdims=True)
  # xR = r_adata_in.X[:, r_adata_in.var_names == receptor].toarray()

  xL = adata_in[:, adata_var == ligand].toarray()
  xR = r_adata_in[:, r_adata_var == receptor].toarray()

  logging.info(f'{ligand} {receptor} xL: {xL.shape} xR: {xR.shape}')

  # Keep xL and xR around for permuting later
  L = group_cells(xL, m_y_L, u_y=m_y_groups, min_cells=min_cells)
  R = group_cells(xR, m_y_R, u_y=m_y_groups, min_cells=min_cells)

  # give the whole group to be interaction scored, which returns a dense np.ndarray
  # there are going to be __many__ elements which represent invalid combinations i.e. across buckets.
  # We take care of those later.
  I_test = calc_interactions(R, L, P)

  logging.info(f'{ligand} {receptor} I_test: {I_test.shape}, {np.sum(I_test.nbytes)/(1024**2):0.2f}MB')
  logging.info(f'{ligand} {receptor} Original nonzero: {(I_test > 0).sum()}')
  I_test_original = pd.DataFrame(I_test, index=m_y_groups, columns=m_y_groups)


  perm_kwargs = dict(
    xL=xL,
    xR=xR,
    P=P,
    yL=yL,
    yR=yR,
    constraints_L=constraints_L,
    constraints_R=constraints_R,
    m_y_groups=m_y_groups,
    min_cells=min_cells
  )
  null_interactions = [_process_permutation(**perm_kwargs) for _ in range(permutations)]

  null_interactions = np.dstack(null_interactions)
  significance_vals = np.quantile(null_interactions, q=sig_level, axis=-1)

  # Compare I to the significance levels
  I_test[I_test < significance_vals] = 0

  # Construct a mask of valid within-cell-bucket comparisons 
  # This should be squares along the diagonal of the overall array
  valid_I = np.zeros_like(I_test, dtype='bool')
  m_expanded = np.array([m.split(';')[0] for m in m_y_groups])
  for u_bucket in u_constraints:
    v = (m_expanded == u_bucket).reshape(-1, 1)
    valid_I[np.matmul(v, v.T)] = 1

  I_test[np.logical_not(valid_I)] = 0

  # Use the original values to get p-values. This can help us sanity check also.
  # p vals for invalid (cross-bucket) interactions should be 1
  if calculate_pvals:
    logging.info(f'{ligand} {receptor} Calculating pvalues: {I_test.shape}, {null_interactions.shape}')
    pvals = calc_pvals(I_test_original, null_interactions)

  I_test = pd.DataFrame(I_test, index=m_y_groups, columns=m_y_groups)
  tend = time.time()
  logging.info(f'{ligand} {receptor} passing interactions: {(I_test.values > 0).sum()} '+\
               f'(of {np.prod(I_test.shape)}) {tend-tstart:3.3f}s')

  # TODO different output types
  # New: we want to return dictionaries
  # if return_null_data:
  #   return {receptor: [I_test_original, I_test, pvals, null_interactions]}
  # else:
  #   return {receptor: [I_test_original, I_test, pvals]}

  # Can we check that the outdir is actually available?
  if outdir is not None:
    outf = f'{outdir}/{receptor}_{ligand}_I.pkl'
    logging.info(f'Writing {I_test.shape} --> {outf}')
    I_test.to_pickle(outf)

    return outf

  else:
    return I_test


