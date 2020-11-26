import numpy as np
import pandas as pd
import time
from scipy.sparse import lil_matrix, csr_matrix, issparse, isspmatrix_lil

__all__ = [
  'group_cells',
  'calc_interactions'
]



# fns = {
#   'mean': np.mean,
#   'sum': np.sum,
#   'percent': lambda x: (x > 0).mean(),
# }
# def group_cells(x, y, u_y=None, min_cells=10, n=50, size=20, agg='mean', log=False):
#   t0 = time.time()
#   df = pd.DataFrame(x, columns=['gene_{k}' for k in range(x.shape[1])])
#   cols = df.columns
#   df['groups'] = y
#   counts = df.groups.value_counts()
#   if u_y is None:
#     u_y = np.unique(y)

#   df = df.loc[df.groups.isin(u_y)]
#   xout = df.groupby('groups').apply(fns[agg]).loc[u_y, cols].values
#   if log:
#     xout = np.log1p(xout)
#   t1 = time.time()

#   # print(f' ------- group time {t1-t0:3.3f} xout={xout.shape}')

#   return xout




def group_cells(x, y, u_y=None, min_cells=10, n=50, size=20, agg='mean', log=False):
  """
  build a summary of x (N x C) by grouping sets of cells (rows) by values in y (N x 1)
  according to the aggregation strategy (sum, mean, nonzero_mean, percent)
  
  - We usually want to give u_y , which lists uniques in y in case there's something
    missing, we'll want to have a 0 placeholder there

  :param x: np.ndarray with instances we want to aggregate
  :param y: vector indicating labels of x
  :param u_y: vector optionally giving an (ordered) list of uniques of y
  :param min_cells: groups with fewer than this number of instances are skipped (returned as 0's)
  :param n: (TODO)/NOT-USED when subsampling instead of all-sampling, this is how many reps to do
  :param size: (TODO)/NOT-USED when subsampling instad of all-sampling, this is how many instances 
                to take per rep
  :param agg: [mean, nonzero_mean, sum, percent] the aggregation type (TODO or a function)

  Returns
  group_x ~ (M x C)
  """
  t0 = time.time()
  if u_y is None:
    u_y = np.unique(y)

  group_x = np.zeros((len(u_y), x.shape[1]), dtype=np.float32)
  for i,u in enumerate(u_y):
    idx = y == u
    if idx.sum() < min_cells:
      continue
    x_u = x[idx, :]
    group_x[i, :] = agg_x(x_u, agg=agg)

  if log:
    group_x = np.log1p(group_x)

  # group_x = lil_matrix(group_x) 
  t1 = time.time()

  # print(f' ------- group time {t1-t0:3.3f} group_x={group_x.shape}')

  return group_x



def agg_x(x, agg='mean'):
  # aggregate expression in x , an n x c np.ndarray
  if agg=='mean':
    x_ = np.mean(x, axis=0)
  elif agg=='nonzero_mean':
    xs = x.copy()
    xs[xs == 0] = np.nan
    x_ = np.nanmean(xs, axis=0)
  elif agg=='sum':
    x_ = np.sum(x, axis=0)
  elif agg=='percent':
    x_ = np.mean((x > 0), axis=0)

  return x_ 




# def constrained_interaction_score(R, L, P, samples_R, samples_L):
#   """
#   Divide the R and L arrays according to groups in samples_R, and samples_L
#   """

def calc_interactions(R, L, P, verbose=False, as_np_array=False):
  """
  The sparse conversions let us balloon up to quite large matrices if we want to work on individual cells
  which , for now , we don't want to do.

  :param R: (M x C) (maybe sparse) matrix of receptor scores
  :param L: (M x D) (maybe sparse) matrix of ligand expression
  :param P: (C x D) (maybe sparse) matrix of receptor ligand interactivity
  """
  # Y = R.dot(P)
  Y = np.matmul(R, P)
  # print(f'calc_interacitons Y: {Y.shape}, {Y.dtype}')
  # I = csr_matrix(L).dot(csr_matrix(Y.T))
  I = np.matmul(L, Y.T)
  # print(f'calc_interacitons I: {I.shape}, {I.dtype}')
  I = csr_matrix(I)
  return I

# def calc_interactions(R, L, P, verbose=False, as_np_array=False):
#   """
#   The sparse conversions let us balloon up to quite large matrices if we want to work on individual cells
#   which , for now , we don't want to do.

#   :param R: (M x C) (maybe sparse) matrix of receptor scores
#   :param L: (M x D) (maybe sparse) matrix of ligand expression
#   :param P: (C x D) (maybe sparse) matrix of receptor ligand interactivity
#   """

#   ## Recovered ligand regulatory potential , per cell , given the receptor activity per cell
#   ## This is the sum of receptor-ligand interactions weighted by receptor activity in the cell
#   ## Receptor activity is the average of downstream genes * the expression of the receptor itself
#   ## Rows ~ cells , Columns ~ ligands

#   # print('input: R: ', R.shape)
#   # print('input: L: ', L.shape)
#   # print('input: P: ', P.shape)

#   # sometimes we want to normalize by the nubmer of receptor/ligands per cell:
#   # n_receptors_per_cell = np.squeeze(np.sum(R.toarray() > 0, axis=1))
#   # Y = np.matmul(R, P) #/ (n_receptors_per_cell)
#   Y = R.dot(P)
#   # Y = Y.tolil()

#   # print(f'sparse check Y: {isspmatrix_lil(Y)}')

#   # print(f'sparse Y: {issparse(Y)}')

#   # print('n_receptors_per_cell:', n_receptors_per_cell.shape)
#   # print('Y:', Y.shape)

#   # Y[:, is_0] = 0
#   # Y[np.isnan(Y)] = 0
#   # Y[n_receptors_per_cell == 0, :] = 0
#   # if verbose:
#   #   print('R:', R.shape, R.dtype)
#   #   print('P:', P.shape, P.dtype)
#   #   print('Y:', Y.shape, Y.dtype)
#   #   print('Converting to sparse...')

#   # sp_YT = csr_matrix(Y.T)
#   # sp_L = csr_matrix(L)

#   # if verbose:
#   #   print('Multiplying sparse matrices', sp_L.shape, sp_YT.shape)

#   ## Cell - to - cell interaction potential is the sum (or average) of matching ligands + ligand potential
#   ## We sum over the expressed ligands in each cell, and the ligands we previously determined to potentiate
#   ## a receptor response in each other cell.
#   ## Cell, cell pairs with high agreement receive high scores.
#   ## We need a normalization scheme, as the scores grow essentially without bound.
#   ## This method has a low specificity, favoring "sending" cells with --many-- expressed ligands.
#   ## Therefore, it makes sense to scale the score according to the number of non-zero ligands on the sending cell
#   #XL = np.matmul(L > 0 , Y.T) / np.sum(L > 0, axis=1, keepdims=True)


#   # n_ligands_per_cell = np.squeeze(np.sum(L.toarray() > 0, axis=1))
#   # sp_I = sp_L.dot(sp_YT) / (n_ligands_per_cell)

#   # Normalizing like this divides by 0, which gets fixed immediately, but it pops an error.
#   # I = np.matmul(L, Y.T) #/ (n_ligands_per_cell)
#   I = csr_matrix(L).dot(csr_matrix(Y.T))
#   # I = I.tolil()
#   # print(f'sparse I: {issparse(I)}')

#   # print(f'sparse check I: {isspmatrix_lil(I)}')

#   # sp_I[np.isnan(sp_I)] = 0
#   # I[n_ligands_per_cell == 0, :] = 0
#   # if verbose:
#   #   print('sp_I', sp_I.shape, sp_I.dtype)

#   # I = sp_I
#   # if verbose:
#   #   print('I:', I.shape, I.dtype)
#   # if as_np_array:

#   # This is here in case we've been working with sparse
#   # I = np.array(I)

#   # I = I.tocsr()
#   # I = I.tolil()

#   return I
