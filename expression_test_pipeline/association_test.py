import numpy as np

from scipy.stats import ranksums

from utils import get_logger



def group_values(expr, groupby, u_groups, agg='mean', 
                 fill_value=0, filter_percent=True):
  """
  Assume expr is N x M and groupby is N x 1 

  Provide u_groups externally to not rely on the values in groupby

  fill_value: the default to use when a sample is in u_groups but missing from groupby
  """

  logger = get_logger()

  if agg == 'mean':
    group_fn = lambda x: np.mean(x, axis=0)
  elif agg == 'sum':
    group_fn = lambda x: np.sum(x, axis=0)

  pct_fn = lambda x: (x > 0).mean(axis=0)

  grouped = np.zeros((len(u_groups), expr.shape[1]), dtype=np.float32)
  for i, u in enumerate(u_groups):
    idx = groupby == u
    if idx.sum() == 0:
      continue
    vals = expr[idx, :]
    grp_vals = group_fn(vals)
    if filter_percent:
      pct_vals = pct_fn(vals)
      grp_vals[pct_vals < 0.1] = 0
      logger.info(f'Grouping {u}: filtered {(pct_vals < 0.1).sum()} values < 10%')

    grouped[i, :] = grp_vals

  return grouped


def association_test(rscore, gex, split_qs=[0.25, 0.75], min_nonzero_pct=0.5):
  """
  We have this situation: for a set of samples s=1,...,N , 
  determine if the expression of Ligand L in a subset of cells
  is significantly associated with the Receptor Activity R of
  a related receptor.

  We can approach a few ways, more than this too I'm sure:
  1. Correlate the expression of L and R across samples
  2. Determine if a divsion of samples according to quantiles of 
    L expression also predicts (associates with) differential R  

  Basically this function implements procedure #2:
  1. Split samples according to expression L by median, or specifically requested quantiles
    1a. Determine if L is significantly different according to the split ?
  2. Find R values for samples in groups L_hi and L_lo
  3. Compare R values with a statistical test

  :param rscore: a vector N x 1
  :param gex: a vector N x 1
  :param split_qs: a list of quantiles to split, default 0.25 , 0.75
  :param min_nonzero_pct: minimum percent of N samples that must be nonzero to do the test
  """

  logger = get_logger()

  min_n = int(rscore.shape[0] * min_nonzero_pct)

  if gex.sum() == 0:
    return 1., 'ligand sum = 0'

  qs = np.quantile(gex, split_qs)

  low_idx = gex <= qs[0]
  high_idx = gex >= qs[1]

  low_l = gex[low_idx]
  high_l = gex[high_idx]
  res = ranksums(low_l, high_l)
  if res.pvalue > 0.05:
    return 1, 'ligand change not significant'

  logger.debug(f'Low L : {low_idx.sum()}')
  logger.debug(f'High L : {high_idx.sum()}')

  low_r = rscore[low_idx]
  high_r = rscore[high_idx]

  # Require the change to be positive
  if np.mean(high_r) < np.mean(low_r):
    return 1., 'receptor changes down'

  res = ranksums(low_r, high_r)
  pval = res.pvalue

  return pval, ''

