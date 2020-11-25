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
import logging

import seaborn as sns


__all__ = [
  'make_logger',
  'get_logger',
  'get_interactions',
  'get_receptor_color',
  'write_sender_karyotype',
  'write_receptor_karyotype',
  'get_ligand_color',
  'write_ligands',
  'draw_links',
  'draw_links_contrast',
  'filter_interactions_by_contrast',
]

def make_logger():
  logger = logging.getLogger('CIRCOS')
  logger.setLevel('INFO')
  ch = logging.StreamHandler()
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  ch.setFormatter(formatter)
  logger.addHandler(ch)
  return logger




def get_logger():
  logger = logging.getLogger('CIRCOS')
  return logger



def get_interactions(adata, radata, sender, receivers, percent_cutoff, 
                     subtype_col, broadtype_col, ligand_receptor, allow_interactions=None):
  logger = get_logger()

  logger.info(f'Running interaction script with adata: {adata.shape}')
  logger.info(f'Running interaction script with radata: {radata.shape}')

  ligands = np.unique(ligand_receptor['ligand'].values)
  logger.info(f'Inferring interactions from {len(ligands)} ligands')

  receptors = np.unique(ligand_receptor['receptor'].values)
  logger.info(f'Inferring interactions from {len(receptors)} receptors')

  celltypes = np.unique(adata.obs[broadtype_col].values)
  logger.info(f'Using {len(celltypes)} celltypes')

  diffrx = {}
  singular_celltypes = []
  for ct in celltypes:
    logger.info(f'Getting receptor enrichment for broad cell type {ct}')

    # get broad cell type background to use:
    ct_subtypes = np.unique(radata.obs[subtype_col].values[radata.obs[broadtype_col] == ct])
    if len(ct_subtypes) == 1:
      singular_celltypes.append(ct)
      logger.info(f'Celltype {ct} has one subtype... ')
      continue

    rd = radata[radata.obs[broadtype_col] == ct, radata.var_names.isin(receptors)]
    logger.info(f'Running rank genes with adata subset {rd.shape} on column {subtype_col}')
    try:
      sc.tl.rank_genes_groups(rd, groupby=subtype_col, method='wilcoxon')
    except:
      logger.warning(f'Problems running rank genes on broad type {ct}')
      continue

    rdx = pd.DataFrame(rd.X.toarray() > 0, index=rd.obs_names, columns=rd.var_names)
    rdx['subtype'] = rd.obs[subtype_col]
    rdx = rdx.groupby('subtype').mean()

    for st in ct_subtypes:
      expressed_receptors = rdx.columns[rdx.loc[st] > percent_cutoff].tolist() # Percent expression cutoff
      df = sc.get.rank_genes_groups_df(rd, st)

      # Require a LFC and p-value
      df = df.query("logfoldchanges > 0.25 & pvals_adj < 0.05")

      df = df.loc[df.names.isin(expressed_receptors)]
      df['subtype'] = [st] * df.shape[0]
      diffrx[st] = df.copy()
      logger.info(f'Subtype {st} got {df.shape[0]} relatively active receptors')
          

  background_set = np.unique(adata.obs[adata.obs[subtype_col]==sender][broadtype_col])[-1]
  logger.info(f'Running sender ligand enrichment against background: {background_set}')

  send_ad = adata[adata.obs[broadtype_col].isin([background_set]), adata.var_names.isin(ligands)]
  sc.tl.rank_genes_groups(send_ad, groupby=subtype_col, method='wilcoxon', n_genes=send_ad.shape[1])
  sender_df = sc.get.rank_genes_groups_df(send_ad, sender)
  sender_df = sender_df.query("logfoldchanges > 0.5 & pvals_adj < 0.05")
  sender_ligands = sender_df.names.tolist()

  logger.info(f'Sender subtype {sender} has {len(sender_ligands)} highly expressed ligands')

  # Track the matched ligand-receptors 
  interactions = {}
  for receiver in receivers:
    logger.info(f'Tracking interactions for {sender} --> {receiver}')
    if receiver not in diffrx.keys():
      logger.warn(f'receiver type {receiver} not in diffrx keys. It will be excluded from now on.')
      continue

    receptor_set = diffrx[receiver].names.values

    logger.debug(receiver)
    logger.debug(f'{receptor_set}')

    matched = ligand_receptor.loc[ligand_receptor.receptor.isin( receptor_set )]
    ligand_set = np.unique(matched['ligand'].values)
    lx = adata[:, adata.var_names.isin(ligand_set)]

    dx = pd.DataFrame(lx.X.toarray() > 0, index=adata.obs_names, columns=lx.var_names)
    dx[subtype_col] = adata.obs[subtype_col]

    pct = dx.groupby(subtype_col).mean() > percent_cutoff

    hits = pct.columns[pct.loc[sender]].tolist()
    logger.debug(f'hit expressed ligands {hits}')

    hits = [h for h in hits if h in sender_ligands]
    logger.debug(f'hit up-regulated ligands {hits}')

    channels = []
    for h in hits:
      rr = matched.loc[matched.ligand == h].receptor.values
      for r in rr:
        if r in receptor_set:
          channels.append(f'{h}_{r}')
                
    logger.debug(f'hit channels ligands {channels}')

    if allow_interactions is not None:
      channels = sorted([c for c in channels if c in allow_interactions])
      logger.info(f'hit allowed ligands {channels}')

    logger.info(f'{receiver} {len(channels)} channels')
    interactions[receiver] = channels
    # logger.info(f'{sender} --> {receiver} {channels} {len(channels)}')

  return interactions



def get_receptor_color(rdxe, receptor, receiver, cmap, contrast=False):
  # Change color scale
  # max_val = np.max(rdxe.loc[:, receptor].values)
  max_val = np.max(rdxe.values)
  min_val = np.min(rdxe.values)
  val = rdxe.loc[receiver, receptor]
  if contrast:
    if val < 0:
      bins = np.linspace(min_val, 0, 10)
      b = np.digitize(val, bins, right=True)
      color = cmap[0][b]
    else:
      bins = np.linspace(0, max_val, 10)
      b = np.digitize(val, bins, right=True)
      color = cmap[1][b]
  else:
    bins = np.linspace(min_val, max_val, 10)
    b = np.digitize(val, bins, right=True)
    color = cmap[b]
  color = f','.join([f'{c}' for c in color])
  return val, color
  


# https://stackoverflow.com/a/29643643
def hex2rgb(h):
  h = h.lstrip('#')
  return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))



def write_sender_karyotype(f, sender, semi_circle, total_ticks, color_palette):
  # write the sending semi-circle
  logger = get_logger()
  color = ','.join([f'{v}' for v in hex2rgb(color_palette[sender])])
  line = f'chr - {sender}_s {sender}_s {semi_circle} {total_ticks} {color}\n'
  logger.info(line.strip())
  f.write(line)



blues = [tuple(int(ch * 255) for ch in c) for c in sns.color_palette('Blues', 10)]
def write_receptor_karyotype(interactions, rdxe, f, hlf, txtf, start, semi_circle, 
                             color_palette, cmap=blues, contrast=False):
  logger = get_logger()

  # Also keep track of where to place ligands
  ligand_order = []
  
  # And coordinates for the links
  receptor_coords = {}
  
  total_receptors_with_repeats = 0
  for receiver, channels in interactions.items():
    receiver_receptors = set([x.split('_')[1] for x in channels])
    total_receptors_with_repeats += len(receiver_receptors)

  logger.info(f'Total receptors with repeats: {total_receptors_with_repeats}')
  area_per_receptor = int(semi_circle / total_receptors_with_repeats)
  logger.info(f'Area per receptor: {area_per_receptor}')

  celltype_start = start
  receptor_start = start
  for receiver, channels in interactions.items():
    if len(channels) == 0:
      logger.warn(f'Receiver {receiver} no matched ligands')
    
    receptor_coords[receiver] = {}
    
    receiver_receptors = [x.split('_')[1] for x in channels]
    ligs = [x.split('_')[0] for x in channels]
    printed_receptors = []

    for ch in channels:
      ligand, receptor = ch.split('_')
      if ligand not in ligand_order:
        ligand_order.append(ligand)
          
      if receptor in printed_receptors:
        continue
      else:
        printed_receptors.append(receptor)
          
      receptor_end = receptor_start + area_per_receptor
      
      # Track the receptor coordinates on this receiver
      receptor_coords[receiver][receptor] = (receptor_start, receptor_end)
      val, color = get_receptor_color(rdxe, receptor, receiver, cmap, contrast=contrast)

      hl = f'{receiver} {receptor_start} {receptor_end} fill_color={color}\n'
      hlf.write(hl)
      
      txt = f'{receiver} {receptor_start} {receptor_end} {receptor} color={color}\n'
      txtf.write(txt)

      logger.info(f'{receiver} {receptor}: {txt.strip()} (val={val:3.3f})')
      receptor_start = receptor_end

    # The value of receptor start is now where our current cell type should stop
    color = ','.join([f'{v}' for v in hex2rgb(color_palette[receiver])])
    celltype_line = f'chr - {receiver} {receiver} {celltype_start} {receptor_start} {color}\n'
    logger.info(celltype_line.strip())
    f.write(celltype_line)
    # start for the next cell type
    celltype_start = receptor_start
      
  return ligand_order, receptor_coords




def get_ligand_color(sdxe, ligand, sender, cmap, contrast=False):
  # Change color scale
  max_val = np.max(sdxe.values)
  min_val = np.min(sdxe.values)
  val = sdxe.loc[sender, ligand]
  if contrast:
    if val < 0:
      bins = np.linspace(min_val, 0, 10)
      b = np.digitize(val, bins, right=True)
      color = cmap[0][b]
    else:
      bins = np.linspace(0, max_val, 10)
      b = np.digitize(val, bins, right=True)
      color = cmap[1][b]
  else:
    bins = np.linspace(min_val, max_val, 10)
    b = np.digitize(val, bins, right=True)
    color = cmap[b]
  color = f','.join([f'{c}' for c in color])
  return val, color





# Find and place ligands 
reds = [tuple(int(ch * 255) for ch in c) for c in sns.color_palette('Reds', 10)]
def write_ligands(sdxe, sender, hlf, txtf, start, total_ticks, ligand_order, 
                  cmap=reds, contrast=False):
  logger = get_logger()

  # Track ligand positions on the sender
  ligand_coords = {}
  
  n_ligands = len(ligand_order)
  logger.info(f'Writing ligands: {n_ligands}')
  borders = np.linspace(start, total_ticks, n_ligands+1, dtype=int)
  # Go in the reverse order that we wrote the receptors:
  for i, ligand in enumerate(ligand_order[::-1]):
    ligand_start = borders[i]
    ligand_end = borders[i+1]
    val, color = get_ligand_color(sdxe, ligand, sender, cmap, contrast=contrast)
    hl = f'{sender}_s {ligand_start} {ligand_end} fill_color={color}\n'
    txt = f'{sender}_s {ligand_start} {ligand_end} {ligand} color={color}\n'
    
    ligand_coords[ligand] = (ligand_start, ligand_end)

    hlf.write(hl)
    txtf.write(txt)

    logger.info(f'{sender} {ligand}: {txt.strip()} (val={val:3.3f})')
  
  return ligand_coords



def get_contrast_color(min_val, max_val, max_neg_val, min_pos_val, val, cmap):
  # Change color scale
  if val < 0:
    bins = np.linspace(min_val, max_neg_val, 10)
    b = np.digitize(val, bins, right=True)
    color = cmap[0][b]
  else:
    bins = np.linspace(min_pos_val, max_val, 10)
    b = np.digitize(val, bins, right=True)
    color = cmap[1][b]

  color = f','.join([f'{c}' for c in color])
  return color




def filter_interactions_by_contrast(interactions, sender, sdxe, rdxe, receiver_order, pct=0.5):
  logger = get_logger()

  # loop over once to get the difference distribution
  link_vals = []
  for receiver in receiver_order:
    channels = interactions[receiver]
    for ch in channels:
      ligand, receptor = ch.split('_')
      rc1 = rdxe[0].loc[receiver, receptor] 
      rc2 = rdxe[1].loc[receiver, receptor]
      sc1 = sdxe[0].loc[sender, ligand] 
      sc2 = sdxe[1].loc[sender, ligand]
      i1 = rc1 * sc1
      i2 = rc2 * sc2
      idiff = i1 - i2
      link_vals.append(idiff)

  logger.info(f'Filtering {len(link_vals)} interaction channels')
  link_vals = np.array(link_vals)

  # we want to keep things with significant deviation from 0
  q_low = np.quantile(link_vals[link_vals < 0], 1-pct)
  q_hi = np.quantile(link_vals[link_vals > 0], pct)
  keep_links = np.zeros_like(link_vals, dtype=np.bool)
  keep_links[link_vals < q_low] = True
  keep_links[link_vals > q_hi] = True

  logger.info(f'Filter ligands q_low = {q_low}')
  logger.info(f'Filter ligands q_hi = {q_hi}')
  logger.info(f'Keeping {keep_links.sum()} links')

  i = 0
  keep_interactions = {}
  for receiver in receiver_order:
    channels = interactions[receiver]
    for ch in channels:
      if keep_links[i]: 
        logger.info(f'keeping link {receiver}: {ch}')
        if receiver not in keep_interactions.keys():
          keep_interactions[receiver] = [ch]
        else:
          keep_interactions[receiver].append(ch)
      i += 1

  return keep_interactions


# sdxe and rdxe should be here length 2 lists
def draw_links_contrast(interactions, sender, linkf, sdxe, rdxe,
                        ligand_coords, receptor_coords, cmap, 
                        receiver_order=None):
  logger = get_logger()
  logger.info('Drawing links with contrast')

  if receiver_order is None:
    receiver_order = list(interactions.keys())

  # loop over once to get the difference distribution
  max_contrast = 0
  max_neg_val = -np.inf
  min_contrast = 0
  min_pos_val = np.inf
  for receiver in receiver_order:
    channels = interactions[receiver]
    for ch in channels:
      ligand, receptor = ch.split('_')
      rc1 = rdxe[0].loc[receiver, receptor] 
      rc2 = rdxe[1].loc[receiver, receptor]
      sc1 = sdxe[0].loc[sender, ligand] 
      sc2 = sdxe[1].loc[sender, ligand]
      i1 = rc1 * sc1
      i2 = rc2 * sc2
      idiff = i1 - i2
      if idiff > max_contrast:
        max_contrast = idiff
      if idiff < min_contrast:
        min_contrast = idiff
      if idiff < 0:
        if idiff > max_neg_val:
          max_neg_val = idiff
      if idiff > 0:
        if idiff < min_pos_val:
          min_pos_val = idiff
          
  logger.info(f'MIN CONTRAST {min_contrast}')
  logger.info(f'MAX CONTRAST {max_contrast}')
  logger.info(f'MAX NEG VAL {max_neg_val}')
  logger.info(f'MIN POS VAL {min_pos_val}')

  itxp = pd.DataFrame()
  link_strings = []
  link_values = []
  for receiver in receiver_order:
    channels = interactions[receiver]
    # color = ','.join([f'{v}' for v in hex2rgb(color_palette[receiver])])
    
    for ch in channels:
      ligand, receptor = ch.split('_')
      l_c = ligand_coords[ligand]
      r_c = receptor_coords[receiver][receptor]
      
      rc1 = rdxe[0].loc[receiver, receptor] 
      rc2 = rdxe[1].loc[receiver, receptor]
      sc1 = sdxe[0].loc[sender, ligand] 
      sc2 = sdxe[1].loc[sender, ligand]
      i1 = rc1 * sc1
      i2 = rc2 * sc2
      idiff = i1 - i2
      color = get_contrast_color(min_contrast, max_contrast, 
                                 max_neg_val, min_pos_val,
                                 idiff, cmap)

      itxp.loc[f'{sender}_{ligand}', f'{receiver}_{receptor}'] = idiff

      if idiff > 0:
        trx = f'{min(1-(idiff / max_contrast)**2, 0.9):3.3f}'
      else:
        trx = f'{min(1-(idiff / min_contrast)**2, 0.9):3.3f}'
      # logger.info(f'{sender}\t{ligand}\t{receiver}\t{receptor}\t{i}\t{trx}')
      l0 = l_c[0]
      l1 = l_c[1]

      r0 = r_c[0]
      r1 = r_c[1]
      
      link = f'{sender}_s {l0} {l1} {receiver} {r0} {r1} color={color},{trx}\n'
      # link = f'{sender}_s {l0} {l1} {receiver} {r0} {r1} color={color}\n'
      
      logger.info(link.strip())
      # linkf.write(link)
      link_strings.append(link)
      link_values.append(idiff)

  link_values = np.abs(np.array(link_values))
  perm = np.argsort(link_values)
  for p in perm:
    linkf.write(link_strings[p])

  return itxp


def draw_links(interactions, sender, linkf, sdxe, rdxe, 
               ligand_coords, 
               receptor_coords, 
               color_palette,
               receiver_order=None):

  logger = get_logger()

  if receiver_order is None:
    receiver_order = list(interactions.keys())

  max_interaction = 0
  for receiver in receiver_order:
    channels = interactions[receiver]
    for ch in channels:
      ligand, receptor = ch.split('_')
      rexp = rdxe.loc[receiver, receptor]
      sexp = sdxe.loc[sender, ligand]
      i = rexp * sexp
      if i > max_interaction:
        max_interaction = i
      logger.info('\t'.join([
        'itxval', 
        sender, ligand,
        receiver, receptor,
        f'{i:3.5f}'
        ])
      )
          
  for receiver in receiver_order:
    channels = interactions[receiver]
    
    color = ','.join([f'{v}' for v in hex2rgb(color_palette[receiver])])
    
    for ch in channels:
      ligand, receptor = ch.split('_')
      l_c = ligand_coords[ligand]
      r_c = receptor_coords[receiver][receptor]
      
      rexp = rdxe.loc[receiver, receptor]
      sexp = sdxe.loc[sender, ligand]
      i = rexp * sexp

      trx = f'{min(1-(i / max_interaction)**2, 0.9):3.3f}'
      # logger.info(f'{sender}\t{ligand}\t{receiver}\t{receptor}\t{i}\t{trx}')
      l0 = l_c[0]
      l1 = l_c[1]

      r0 = r_c[0]
      r1 = r_c[1]
      
      link = f'{sender}_s {l0} {l1} {receiver} {r0} {r1} color={color},{trx}\n'
      
      logger.info(link.strip())
      linkf.write(link)




