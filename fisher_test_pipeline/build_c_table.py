import numpy as np
from utils import get_logger

def is_expressed(foreground, background, 
                 significance_level=0.05, 
                 fallback_positive_cutoff=0.1, 
                 reps=1000,
                 expression_test='permutation',
                 foreground_majority_behavior='percent', # what to do when foreground > background
                 verbose=False
                ):
  """
  Determine if foreground set of cells has enriched expression relative to background

  Generally this will be done through a permutation test testing the mean 
  of each column of foreground against means of sets the same size from 
  the background set of cells.

  Where we hit a relatively small number of cells in the background set we can :
  (1) apply a straight percent non-zero cutoff
  (2) compare non-zero percents, weighted by size ?
  (3) straight up compare foreground vs background mean ? This would be the same
      as carrying out the full permutation test, since the whole background would 
      just be sampled each time through
  
  This is not uncommon, especially when passing in broad cell types
  """

  logger = get_logger()
  if foreground.shape[0] == 0:
    logger.debug('is_expressed got foreground shape=0')
    return False
  
  if expression_test == 'percent':
    pct = (foreground > 0).mean(axis=0)
    return (pct > fallback_positive_cutoff).sum() > 0

  if (background.shape[0] < foreground.shape[0]) :
    logger.debug('is_expressed got ncells(background) < ncells(foreground)')
    # If any of the columns are non-zero , return True
    if foreground_majority_behavior == 'percent':
      pct = (foreground > 0).mean(axis=0)
      return (pct > fallback_positive_cutoff).sum() > 0
    elif foreground_majority_behavior == 'compare_percent':
      pct_fg = (foreground > 0).mean(axis=0)
      pct_bg = (background > 0).mean(axis=0)
      return (pct_fg > pct_bg).sum() > 0
    elif foreground_majority_behavior == 'means':
      mean_fg = np.mean(foreground, axis=0)
      mean_bg = np.mean(background, axis=0)
      return (mean_fg > mean_bg).sum() > 0
  
  # We've covered the degenerate cases, do the real test
  # If multiple, test independently
  fg_mean = np.mean(foreground, axis=0)
  
  # Decide how many background points to sample
  # Usually sample the same size as foreground
  n_sample = foreground.shape[0]
  
  null_distrib = np.zeros((reps, background.shape[1]))
  for i in range(reps):
    idx = np.random.choice(background.shape[0], n_sample, replace=False)
    null_distrib[i, :] = np.mean(background[idx, :], axis=0)
      
  # Calculate enriched in foreground vs null-distribution
  q = np.quantile(null_distrib, 1-significance_level, axis=0)
      
  passing = fg_mean > q
  
  if verbose:
    with np.printoptions(suppress=True, precision=3):
      print('', fg_mean , '\nvs\n', q, '\n', passing)
  
  # Return true if any of the columns pass
  return passing.sum() > 0




## If we can get everything here into a numpy equivalent , we can use @numba.jit
def build_c_table(r_scores, gene_expr, 
                  r_group, s_group, 
                  r_samples, s_samples,
                  receptor, ligands, 
                  r_celltypes='SubType_v2', s_celltypes='SubType_v2', 
                  r_expression_test='permutation', 
                  l_expression_test='permutation', 
                  foreground_majority_behavior='percent', # what to do when foreground > background
                  reps=1000,
                  verbose=False):
  """
  ctable is sorting cases according to expression of ligand and receptor activity enrichment:

                        ligand not-expressed | ligand expressed 
  --------------------------------------------------------------
  receptor not-active |                      |                 |
  --------------------------------------------------------------
  receptor active     |                      |                 |
  --------------------------------------------------------------

  receptor_active is receptor score in [population] > 95-th percentile of background scores for that receptor
  ligand-expressed is as ligand(s) expression in [population] > 95-th percentile of background expression for that receptor

  background expression is sampled from combinations of cells NOT in [population], taking samples to be the same
  number of cells as $N_{population}$

  NOTES:
  - There is a question of how to interpret populations missing from a particular sample.

  Arguments:
  :param r_scores: anndata.AnnData object
  :param gene_expr: anndata.AnnData object
  :param r_group: string or list
  :param s_group: string or list
  :param r_samples: key of obs to use a samples
  :param s_samples: key of obs to use a samples
  :param receptor: the name of the receptor to test
  :param ligands: the name(s) of the ligand(s) to test
  :param r_celltypes: key of obs to use as celltypes 
  :param s_celltypes: key of obs to use as celltypes 
  """

  logger = get_logger()
  c_table = np.zeros((2,2), dtype=np.int) 
      
  if not isinstance(r_group, list):
    r_group = [r_group]
      
  if not isinstance(s_group, list):
    s_group = [s_group]

      
  r_scores = r_scores[:, r_scores.var_names == receptor]
  gene_expr = gene_expr[:, gene_expr.var_names.isin(ligands)]
  
  recv_idx = r_scores.obs[r_celltypes].isin(r_group)
  send_idx = gene_expr.obs[s_celltypes].isin(s_group)
  
  logger.debug(f'Checking for {ligands} --> {receptor} interactions ({s_group} --> {r_group})')

  u_samples_r = set(np.unique(r_scores.obs[r_samples]))
  u_samples_s = set(np.unique(gene_expr.obs[s_samples]))

  u_samples = u_samples_r.intersection(u_samples_s)
  logger.debug(f'Working on {len(u_samples)} samples common to both sample annotations')


  for sample in u_samples:
    r_sample_idx = r_scores.obs[r_samples] == sample
    s_sample_idx = gene_expr.obs[s_samples] == sample
    
    recv_r_score = r_scores[r_sample_idx & recv_idx].X.toarray()
    background_r_score = r_scores[r_sample_idx & ~recv_idx].X.toarray()
    
    send_gene_expr = gene_expr[s_sample_idx & send_idx].X.toarray()
    background_gene_expr = gene_expr[s_sample_idx & ~send_idx].X.toarray()
    
    # logger.info(f'{sample}')
    # logger.info(f'{receptor} recv_r_score {recv_r_score.shape} background {background_r_score.shape}')
    # logger.info(f'{len(ligands)} ligands: send_gene_expr {send_gene_expr.shape} background {background_gene_expr.shape}')
    
    # do expression of receiver
    r_expressed = is_expressed(recv_r_score, background_r_score, 
                               expression_test=r_expression_test, 
                               foreground_majority_behavior=foreground_majority_behavior,
                               reps=reps,
                               verbose=verbose)

    l_expressed = is_expressed(send_gene_expr, background_gene_expr, 
                               expression_test=l_expression_test, 
                               foreground_majority_behavior=foreground_majority_behavior,
                               reps=reps,
                               verbose=verbose)
    c_table[int(r_expressed), int(l_expressed)] += 1

      
  return c_table