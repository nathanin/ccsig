import numpy as np
import pandas as pd
import scanpy as sc
import zarr
import glob
import shutil
import os

from anndata import AnnData
import itertools
import tqdm.auto as tqdm

import ray
import logging
import pickle

@ray.remote(num_cpus=1)
def process_itx_block(sender_x, receiver_x, ligands, receptors, 
                      channels,
                      sender_name, receiver_name,
                      I,
                      logger):
  sl = channels[0]
  rl = channels[1]
  shape = (sender_x.shape[0], receiver_x.shape[0], len(sl))
  # logger.info(f'Processing interactions for {sender_name} --> {receiver_name} (shape={shape})')
  # print(f'Processing interactions for {sender_name} --> {receiver_name} (shape={shape})') 
  # I = np.zeros(shape, dtype=np.float16)
  for i,(l,r) in enumerate(zip(sl,rl)):
    L = sender_x[:,ligands==l]
    R = receiver_x[:,receptors==r]
    # if R.shape[1] > 1:
    #   R = np.sum(R, axis=1, keepdims=True)
    
    I[:,:,i] = np.matmul(L, R.T).astype('f2')
  return (sender_name, receiver_name)#, I


class itxHandler:
  def __init__(self, logger=None, verbose=False):
    self._initialized = False
    self._using_receptor_scores = False

    if logger is None:
      self.logger = logging.getLogger('Single cell ITX')
      self.logger.setLevel('INFO')
      ch = logging.StreamHandler()
      formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
      ch.setFormatter(formatter)
      self.logger.addHandler(ch)
    else:
      self.logger = logger

    self.verbose = verbose
    self._needs_cleanup = False

  def close(self):
    if self._needs_cleanup:
      self.logger.info(f'Cleaning up ramdisk copy: {self.ramdisk_copy}')
      # what we really don't want is to delete an original copy of data
      shutil.rmtree(self.ramdisk_copy)

  def create(self, adata, lr_matrix, ligands, receptors, radata=None):
    L = np.array([l in adata.var_names for l in ligands])
    R = np.array([r in adata.var_names for r in receptors])
    self.lr_matrix = lr_matrix[L,:][:,R].astype(np.float16)
    self.ligands = ligands[L]
    self.receptors = receptors[R]

    self.ligand_adata = adata[:,self.ligands].copy()
    if isinstance(radata, AnnData):
      # TODO check if radata is given, that the order and everything is the same
      self._using_receptor_scores = True
      self.receptor_adata = radata[:,self.receptors].copy()
    else:
      self.receptor_adata = adata[:,self.receptors].copy()
    self.categorical_groups = adata.obs.select_dtypes(include='category').columns.tolist()
    self.obs = adata.obs.copy() 

    channels = []
    for k,l in enumerate(self.ligands):
        lr = self.lr_matrix[k] > 0
        #print(np.nonzero(lr)[0])
        for j in np.nonzero(lr)[0]:
            pair = (l, self.receptors[j])
            channels.append(pair)
        
    single_ligands = np.array([l for l,r in channels])
    single_receptors = np.array([r for l,r in channels])

    self.n_channels = len(single_ligands)
    self.channels = (single_ligands, single_receptors)
    self.single_ligands = single_ligands
    self.single_receptors = single_receptors

    self._initialized = True
    self.logger.info('Initialized itxHandler')


  def create_from_db(self, path, ramdisk='/dev/shm', mode='r'):
    self.logger.info(f'loading database from {path}')
    # initial tests - this may not actually be faster at all ? 
    if ramdisk is not None:
      base = '_'.join(path.split('/')[-2:])
      active_path = f'{ramdisk}/{base}'
      self.logger.info(f'copying database to ramdisk: {active_path}')
      shutil.copytree(path, active_path)
      self.ramdisk_copy = active_path # this attribute only exists if the copy succeeds
    else: 
      active_path = path

    signif_file = f'{active_path}/significance_0.01.pkl'
    self.alpha_levels = pickle.load(open(signif_file, 'rb'))

    zarr_path = f'{active_path}/interaction_store.zarr'
    # self.zstore = zarr.open(zarr_path, mode=mode)
    # self.zcache = zarr.LRUStoreCache(self.zstore, max_size = 2**28)
    # self.zroot = zarr.group(store=self.zcache)

    self.zroot = zarr.open(zarr_path, mode=mode)

    # we need the interaction data saved in the zarr store
    self.ligands = self.zroot['ligands'][:]
    self.receptors = self.zroot['receptors'][:]
    self.lr_matrix = self.zroot['lr_matrix'][:]

    self.single_ligands = self.zroot['single_ligands'][:]
    self.single_receptors = self.zroot['single_receptors'][:]

    self.ligand_adata = None
    self.receptor_adata = None
    self.obs = None
    self._initialized = True
    if ramdisk is not None:
      self._needs_cleanup = True


  def repack_interactions(self, disable_pbar=False):
    """
    The original zarr strategy is to store chunks of pieces of a much larger cell-cell
    interaction graph:
    
    Celltypes M and N have i and j cells apiece, and over K interaction channels, the
    chunk sizes are (i,j,1), resulting in K individual files.

    The problem is that by this chunking strategy, access to individual cells 
    is insanely slow. To read the data for all K channels for just 1 pair of cells, 
    we have to read the entire dataset!

    While this repacking won't solve that problem, per se, it will help us to do what
    we really want, which is to sum the pairwise cell-cell potentials for every 
    pair of cells.

    If we have C cells in a dataset, we'll repack the piecemeal data as chunks:
    (C, C, 1)

    This way, to get the cell-cell interactions we sum over K:
    I_cell-cell = \sum_k I_k

    We still have to access the whole dataset, of course - no way around that fact,
    but this way we only access it once instead of C times!
    """

    all_groups = self.zroot['groups'][:]
    group_borders = {}
    n_cells = 0
    start = 0
    for g in all_groups:
      bc = self.zroot[f'barcodes_{g}'][:]
      end = start + len(bc)
      group_borders[g] = (start,end)
      start = end
      n_cells += len(bc)

    n_channels = len(self.single_ligands)
    chunks = (n_cells, n_cells, 1)
    shape = (n_cells, n_cells, n_channels)
    I = self.zroot.create('full_cell_cell_interaction', shape=shape, dtype='f2', chunks=chunks)
    Isig = self.zroot.create('full_cell_cell_interaction_sig', shape=shape, dtype='f2', chunks=chunks)
    
    pbar = tqdm.trange(n_channels, disable=disable_pbar)
    for k in pbar:
      Ik = np.zeros((n_cells, n_cells), dtype='f2')
      Isigk = np.zeros((n_cells, n_cells), dtype='uint8')
      ch_name = f'{self.single_ligands[k]}_{self.single_receptors[k]}'
      ch_significance = self.alpha_levels[ch_name]

      for g in all_groups:
        # we have each group as the sender and as the receiver, including intra-group pairs
        gb = group_borders[g]
        for g2 in all_groups:
          if g==g2:continue # deal with these later
          gb2 = group_borders[g2]
          q = f'{g}__{g2}'
          v = self.zroot[q][:,:,k]
          Ik[gb[0]:gb[1], gb2[0]:gb2[1]] += v
          Isigk[gb[0]:gb[1], gb2[0]:gb2[1]] += v > ch_significance

        # intra-group pairs
        q = f'{g}__{g}'
        v = self.zroot[q][:,:,k]

        Ik[gb[0]:gb[1], gb[0]:gb[1]] = v
        Isigk[gb[0]:gb[1], gb[0]:gb[1]] = v > ch_significance

      I[:,:,k] = Ik 
      Isig[:,:,k] = Isigk 




  def _init_calc(self, workers):
    self.logger.info(f'Starting ray with {workers}')


  def calc(self, groupby, outbase, workers=4, zarr_kw=None):
    """ Calculate raw interactions and save to a backing file system

    zarr creating arrays seems to have problems if the structure already exists.
    """
    if not os.path.exists(outbase):
      os.makedirs(f'{outbase}/{groupby}')

    outf = f'{outbase}/{groupby}/interaction_store.zarr'
    self.logger.info(f'Initializing output zarray store at: {outf}')
    # if zarr_kw is None:
    # store = zarr.DirectoryStore(store=outf)
    # if os.path.exists(outf):
    #   os.removedirs(outf)
    self.zroot = zarr.group(store=outf)

    self.logger.info(f'Storing L/R matrix and Ligand/Receptor references')
    self.zroot.array('lr_matrix', self.lr_matrix)
    self.zroot.array('ligands', self.ligands, dtype=str)
    self.zroot.array('receptors', self.receptors, dtype=str)
    self.zroot.array('single_ligands', self.single_ligands, dtype=str)
    self.zroot.array('single_receptors', self.single_receptors, dtype=str)

    groups = np.array(self.obs[groupby])
    u_groups, i_groups = np.unique(groups, return_inverse=True)
    iu_groups = np.unique(i_groups)

    self.logger.info(f'Got {len(u_groups)} groups: {u_groups}')
    self.zroot.array('groups', u_groups, dtype=str)

    self.logger.info(f'Stashing barcode arrays for {len(u_groups)} groups')
    for g in u_groups:
      group_name = f'barcodes_{g}'
      self.logger.info(f'Stashing barcodes for group {group_name}')
      bc = np.array(self.ligand_adata[groups == g].obs_names.tolist())
      self.zroot.array(group_name, bc, dtype=str)

    self.logger.info(f'Processing interactions for {len(u_groups)} groups')
    futures = []
    for g1,g2 in itertools.product(iu_groups,iu_groups):
      sender_name = u_groups[g1]
      receiver_name = u_groups[g2]
      sender_x = self.ligand_adata[i_groups==g1].X.toarray().astype(np.float16)
      receiver_x = self.receptor_adata[i_groups==g2].X.toarray().astype(np.float16)

      # sender_bc = self.ligand_adata[i_groups==g1].obs_names.tolist()
      # receiver_bc = self.receptor_adata[i_groups==g1].obs_names.tolist()
      group_name = f'{sender_name}__{receiver_name}'
      shape = (sender_x.shape[0], receiver_x.shape[0], self.n_channels)
      chunks = (sender_x.shape[0], receiver_x.shape[0], 1)
      I = self.zroot.create(group_name, shape=shape, dtype='f2', chunks=chunks)

      task_id = process_itx_block.remote(sender_x, receiver_x, 
                                         self.ligands, self.receptors, self.channels,
                                         sender_name, receiver_name,
                                         I,
                                         self.logger)
      futures.append(task_id) 

    self.logger.info(f'Finished creating {len(futures)} remote jobs. Waiting for work to complete.')

    # Consume the results so that we can free up memory -- ?
    # This suffers from blocking, if there's a large cell type early on, the smaller ones
    # will not run until that large matrix is processed and saved. It's just a little
    # optimization that might be useful to reduce the max memory footprint
    for f in futures:
      # name_pair, I = ray.get(f)
      name_pair = ray.get(f)
      self.logger.info(f'Returned value for {name_pair}')
    #   #outf = f'{istore}/{groupby}/{name_pair[0]}__{name_pair[1]}.zarr'
    #   group_name = f'{name_pair[0]}__{name_pair[1]}'
    #   self.logger.info(f'{I.shape}, {I.dtype} --> {group_name}')
    #   self.zroot.array(group_name, I, dtype=I.dtype)
    #   # zarr.save(outf, I)
    #   del I


  def recall_I(self, sender, receiver, ligand, receptor):
    igroup = f'{sender}__{receiver}'
    k = self._lr2k(ligand, receptor)
    I = self.zroot[igroup][:,:,k]
    sbc = self.zroot[f'barcodes_{sender}'][:]
    rbc = self.zroot[f'barcodes_{receiver}']

    sig = self.alpha_levels[f'{ligand}_{receptor}']
    Isig = I > sig

    return I, Isig, sbc, rbc


  def summarize_channels(self, sender, receiver, disable_pbar=False):
    assert hasattr(self, 'alpha_levels'), 'No significance data found. Run the sc_background_test() method first.'
    gname = f'{sender}__{receiver}'
    sl = self.single_ligands
    sr = self.single_receptors
    channel_strs = [f'{l}_{r}' for l,r in zip(sl, sr)]
    channels = pd.DataFrame(index=channel_strs, 
                            columns=[
                              'ligand',
                              'receptor',
                              'avg_potential',
                              'pct_nonzero_itx',
                              'pct_nonzero_sender',
                              'pct_nonzero_receiver',
                              'pct_significant_itx',
                              'pct_significant_sender',
                              'pct_significant_receiver',
                              'significance_cutoff',
                              'avg_nonzero'
                            ])

    
    # can we make parallel ? would it help ?
    # seems to depend on the size and chunking - seeing rates 4 it/s ~ 13 it/s
    # cant rule out disky things
    pbar = tqdm.tqdm(zip(sl, sr), total=len(sl), disable=disable_pbar)
    for k,(l,r) in enumerate(pbar):
      id = f'{l}_{r}'
      sig = self.alpha_levels[id]

      channels.loc[id, 'ligand'] = l
      channels.loc[id, 'receptor'] = r

      i = self.zroot[gname][:,:,k]
      channels.loc[id, 'avg_potential'] = np.mean(i)
      p = np.mean(i > 0)
      channels.loc[id, 'pct_nonzero_itx'] = p
      avg_nz = np.mean(i[i > 0]) if p > 0 else 0.
      channels.loc[id, 'avg_nonzero'] = avg_nz

      channels.loc[id, 'pct_nonzero_sender'] = np.mean(np.sum(i, axis=1) > 0)
      channels.loc[id, 'pct_nonzero_receiver'] = np.mean(np.sum(i, axis=0) > 0)

      isig = i > sig
      channels.loc[id, 'pct_significant_itx'] = np.mean(isig > 0)
      channels.loc[id, 'pct_significant_sender'] = np.mean(np.sum(isig, axis=1) > 0)
      channels.loc[id, 'pct_significant_receiver'] = np.mean(np.sum(isig, axis=0) > 0)
      channels.loc[id, 'significance_cutoff'] = sig

      
    return channels


  def sc_background_test(self, alpha=0.01, disable_pbar=False):
    self.alpha_levels = {}
    pbar = tqdm.tqdm(zip(self.single_ligands, self.single_receptors), 
                     total = len(self.single_ligands),
                     disable = disable_pbar
                     )
    for k, (lig,rec) in enumerate(pbar):
      backgrounds = []
      all_groups = self.zroot['groups'][:]
      for s,r in itertools.product(all_groups, all_groups):
        g = f'{s}__{r}'
        I = self.zroot[g][:,:,k]
        backgrounds.append(I.ravel().copy())
      
      backgrounds = np.concatenate(backgrounds)
      cutoff = np.quantile(backgrounds, 1-alpha)
      self.alpha_levels[f'{lig}_{rec}'] = cutoff




  def test_channel(self, sender, receiver, ligand, receptor, alpha=0.01):
    gname = f'{sender}__{receiver}'
    k = self._lr2k(ligand, receptor)
    
    all_groups = self.zroot['groups'][:]
    Itest = self.zroot[gname][:,:,k]
    
    q = f'{ligand}_{receptor}'
    if hasattr(self, 'alpha_levels') and (q in self.alpha_levels.keys()):
      cutoff = self.alpha_levels[q]
    else:
      backgrounds = []
      for s,r in itertools.product(all_groups, all_groups):
        g = f'{s}__{r}'
        I = self.zroot[g][:,:,k]
        backgrounds.append(I.ravel().copy())
      
      backgrounds = np.concatenate(backgrounds)
      cutoff = np.quantile(backgrounds, 1-alpha)

    self.logger.info(f'{ligand} --> {receptor} potential cutoff: {cutoff:3.3f} (alpha={alpha})')
    Ipass = Itest > cutoff
    return Itest, Ipass


  # def _est_n_itx(self, groupby):
  #   groups = np.array(self.obs[groupby])
  #   u_groups, i_groups = np.unique(groups, return_inverse=True)
  #   iu_groups = np.unique(i_groups)
  #   total_itx = 0
  #   for g1, g2 in itertools.product(iu_groups,iu_groups):
  #     if g1==g2: continue
  #     Ng1 = np.sum(i_groups == g1)
  #     Ng2 = np.sum(i_groups == g2)
  #     total_itx += Ng1 * Ng2
  #   return len(self.ligands) * total_itx

  def __str__(self):
    if self._initialized:
      s = 'Initialized instance of itxHandler:\n'+\
          f'\tlr_matrix: {self.lr_matrix.shape}\n'+\
          f'\tligands: {self.ligands.shape}\n'+\
          f'\treceptors: {self.receptors.shape}\n'+\
          f'\tligand_adata: {self.ligand_adata.shape}\n'

      if self._using_receptor_scores:
        s += f'\treceptor_adata (receptor scores): {self.receptor_adata.shape}\n'
      else:
        s += f'\treceptor_adata: {self.receptor_adata.shape}\n'

      return s
    else:
      return 'Uninitialized instance of itxHandler'

  def _lr2k(self, ligand, receptor):
    k = (self.single_ligands == ligand) & (self.single_receptors == receptor)
    k = np.nonzero(k)[0][0] # is there a better way
    return k