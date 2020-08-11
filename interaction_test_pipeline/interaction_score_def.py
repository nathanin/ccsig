#!/usr/bin/env python

import numpy as np
import pandas as pd
import scanpy as sc
import pickle
import sys
import os

import scrna
import time
from nichenetpy import plot_interactions
from tqdm import tqdm

import multiprocessing
from argparse import ArgumentParser

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

# We want these to have SubType_v4, subtype_timepoint and
# sample_subtype_timepoint attributes in obs
adata_path = 'pembroRT_TNBC_AllCells_GEX.h5ad'
receptor_adata_path = 'pembroRT_TNBC_AllCells_ReceptorScores.h5ad'

# These are the database for receptor/ligand pairs. 
# For R receptors, and L ligands, 
# P should be an R x L matrix where element P_i,j is the 
# potential for Receptor i to interact with Ligand j.
P = np.load('POTENTIALS.npy')
RECEPTORS = np.array([l.strip() for l in open('RECEPTORS.txt')])
LIGANDS = np.array([l.strip() for l in open('LIGANDS.txt')])

# Load a dictionary with keys ~ Receptors, and values ~ ligands
rl_pairs = pickle.load(open("matched_ligands_all.p", "rb"))

# spike in some extra R/L or edit ones that are in there 
rl_pairs['CD40'] = ['CD40LG']
rl_pairs['CXCR5'] = ['CXCL13']
rl_pairs['CXCR4'] = ['CXCL12']
rl_pairs['LTBR'] = ['LTB']
rl_pairs['CD27'] = ['CD70']
rl_pairs['PDCD1'] = ['CD274']
rl_pairs['TIGIT'] = ['PVR']
rl_pairs['HAVCR2'] = ['LGALS9']

# Set some parameters
MIN_CELLS = 10
PERMUTATIONS = 20
GROUPBY = 'sample_subtype_E_ig_timepoint'
SIG_LEVEL = 0.5


# On an HPC environment when this is called as a script, 
# load the adata object here.
def _subset_data(receptor, ligands):
  adata = sc.read_h5ad(adata_path)
  r_adata = sc.read_h5ad(receptor_adata_path)

  # Keep these raw values and apply log to averaged values on the ligand/sender only.
  r_adata_in = r_adata[:, r_adata.var_names == receptor].copy()
  adata_in = adata[:, adata.var_names.isin(ligands)].copy()

  del adata
  del r_adata

  return adata_in, r_adata_in


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



def _process_permutation(r_adata_in, adata_in, receptor, labels, samples, interaction_kw_args):
  # # We need this because the random state in each thread is initialized identically
  pid = multiprocessing.current_process()._identity[0]
  seed = int(time.time())
  np.random.seed(seed+pid)

  shuffled_labels = permute_labels(labels, samples)
  r_adata_in.obs['shuffled_label'] = shuffled_labels.copy()
  adata_in.obs['shuffled_label'] = shuffled_labels.copy()

  interaction_kw_args['receiver_groupby'] = 'shuffled_label'
  interaction_kw_args['sender_groupby'] = 'shuffled_label'
  interaction_kw_args['verbose'] = False

  _, _, I_tmp = plot_interactions([receptor], r_adata_in, adata_in, 
    **interaction_kw_args
  )

  # print(I_tmp.iloc[:2,:2])
  
  return I_tmp


from scipy.stats.mstats import rankdata
def calc_pvals(I_test, null_interactions):
  print('Calculating pvalues', I_test.shape, null_interactions.shape)

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



def run_interaction_test(receptor, return_null_data=False, permutations=50, 
                         min_cells=10, threads=4, verbose=False):
  tstart = time.time()
  ligands = rl_pairs[receptor]
  print(f'Receptor {receptor} ligands {ligands}')
  ligands = np.array(ligands)
  # Filter ligands for those included in the master LIGANDS list
  ligands = ligands[pd.Series(ligands).isin(LIGANDS)]

  adata_in, r_adata_in = _subset_data(receptor, ligands)
  RM_GROUPS = [g for g in np.unique(adata_in.obs[GROUPBY]) if 'x' in g]

  interaction_kw_args = dict(
    P=P,
    all_receptors=pd.Series(RECEPTORS),
    all_ligands=pd.Series(LIGANDS),
    receiver_groupby=GROUPBY,
    sender_groupby=GROUPBY,
    rm_receivers=RM_GROUPS,
    rm_senders=RM_GROUPS,
    min_cells=min_cells,
    force_include=ligands,
    only_use_forced=True,
    show_weights=False,
    show_receptors=False,
    show_ligands=False,
    show_ix=False,
    verbose=verbose)

  _, _, I_test = plot_interactions([receptor], r_adata_in, adata_in, 
    **interaction_kw_args
  )
  print('I_test :', I_test.shape, np.sum(I_test.memory_usage())/(1024**2))
  print('Original nonzero:', (I_test.values > 0).sum())
  I_test_original = I_test.copy()

  labels = np.array(adata_in.obs[GROUPBY])
  samples = np.array(adata_in.obs.sample_id)

  def generate_args():
    for p in range(permutations):
      yield r_adata_in, adata_in, receptor, labels, samples, interaction_kw_args


  pool = multiprocessing.Pool(threads)
  arg_generator = generate_args()
  try:
    null_interactions = pool.starmap(_process_permutation, arg_generator)
  except Exception as e:
    print(e)
  finally:
    pool.close()
    pool.join()


  # Pad I's with 0
  for p, I_tmp in enumerate(null_interactions):
    I_tmp = I_tmp.loc[I_tmp.index.isin(I_test.index), I_tmp.columns.isin(I_test.columns)] 
    full_I = pd.DataFrame(index=I_test.index, columns=I_test.columns, dtype=np.float32)
    full_I.loc[:,:] = 0
    full_I.loc[I_tmp.index, I_tmp.columns] = I_tmp
    null_interactions[p] = full_I.copy()

  null_interactions = np.stack([v for v in null_interactions], axis=-1)
  significance_vals = np.quantile(null_interactions, q=SIG_LEVEL, axis=-1)

  # Compare I to the significance levels
  I_test[I_test < significance_vals] = 0

  # Use the original values to get p-values. This can help us sanity check also.
  pvals = calc_pvals(I_test_original, null_interactions)

  tend = time.time()
  print(f'{receptor} passing interactions: {(I_test.values > 0).sum()} (of {I_test.shape}) {tend-tstart:3.3f}s')

  if return_null_data:
    return I_test_original, I_test, pvals, null_interactions
  else:
    return I_test_original, I_test, pvals



if __name__ == '__main__':

  parser = ArgumentParser()
  parser.add_argument(
    'receptor',
    help = 'Must be a valid key in the `rl_pair` (receptor-ligand pair) dictionary.'
  )
  parser.add_argument(
    'out_dir'
  )
  parser.add_argument(
    '--save_null',
    action = 'store_true'
  )
  parser.add_argument(
    '--save_original',
    action = 'store_true'
  )
  parser.add_argument(
    '--clobber',
    action = 'store_true'
  )
  parser.add_argument(
    '--verbose',
    action = 'store_true'
  )
  parser.add_argument(
    '-j',
    default=4,
    type=int
  )
  parser.add_argument(
    '--min_cells',
    default = 10,
    type=int,
    help = 'The min number of cells per compartment to consider. \
            Compartments that do not pass this cutoff are treated as missing.'
  )
  parser.add_argument(
    '--permutations',
    default = 25,
    type=int,
    help = 'The number of cell permutations to run. A permutation consists of a shuffle of\
            cell type labels within each single sample. Fewer permutations are good for\
            quick filtering of enriched interactions. More permutations allows significance\
            calcultion.'
  )
  parser.add_argument(
    '--groupby',
    default = 'sample_subtype_timepoint',
    type=str
  )
  parser.add_argument(
    '--sig_level',
    default=0.5,
    type=float,
    help = 'The quantile value of the null distribution to accept. 0.5 makes sense for an\
            enrichment filter. 0.95 would correspond to a signifcicance threshold in\
            conjunction with a large number of permutations.'
  )

  ARGS = parser.parse_args()
  outf = os.path.join(ARGS.out_dir, f'{ARGS.receptor}.pkl')

  if os.path.exists(outf):
    if not ARGS.clobber:
      print(f'Found output file {outf}. Exiting')
      sys.exit(0)

  # res returns here as a pd.DataFrame
  if ARGS.save_null:
    original_res, res, null_vals, pvals = run_interaction_test(ARGS.receptor, 
      return_null_data=True, 
      min_cells=ARGS.min_cells,
      permutations=ARGS.permutations,
      threads=ARGS.j,
      verbose=ARGS.verbose)

    if null_vals is not None:
      nullf = os.path.join(ARGS.out_dir, f'{ARGS.receptor}_null.npy')
      print(f'saving permuted values {null_vals.shape} --> {nullf}')
      np.save(nullf, null_vals)


  else:
    original_res, res, pvals = run_interaction_test(ARGS.receptor, 
      return_null_data=False, 
      min_cells=ARGS.min_cells,
      permutations=ARGS.permutations,
      threads=ARGS.j,
      verbose=ARGS.verbose)

  # NOTE read it with pd.read_pickle
  res.to_pickle(outf)

  pval_outf = os.path.join(ARGS.out_dir, f'{ARGS.receptor}_pvals.pkl')
  pvals.to_pickle(pval_outf)

  if ARGS.save_original:
    ogf = os.path.join(ARGS.out_dir, f'{ARGS.receptor}_og.pkl')
    original_res.to_pickle(ogf)


