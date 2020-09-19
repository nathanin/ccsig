import numpy as np
import scanpy as sc
import logging



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


def get_interactions(adata, radata, sender, receivers, percent_cutoff, ligands,
                    subtype_col, broadtype_col, ligand_receptor):
  logger = get_logger()

  ligands = np.unique(ligand_receptor['ligand'].values)

  celltypes = np.unique(ad.obs[broadtype_col].values)

  diffrx = {}
  for ct in celltypes:
    # if ct == 'Endothelial': continue
    rd = radata[radata.obs[broadtype_col] == ct]
    sc.tl.rank_genes_groups(rd, groupby=subtype_col, method='wilcoxon')

    # get broad cell type background to use:

    ct_subtypes = np.unique(radata.obs[subtype_col].values[radata.obs[broadtype_col] == ct])

    rdx = pd.DataFrame(rd.X.toarray() > 0, index=rd.obs_names, columns=rd.var_names)
    rdx['subtype'] = rd.obs[subtype_col]
    rdx = rdx.groupby('subtype').mean()

    for st in ct_subtypes:
      expressed_receptors = rdx.columns[rdx.loc[st] > percent_cutoff].tolist() # Percent expression cutoff
      df = sc.get.rank_genes_groups_df(rd, st)
      df = df.query("logfoldchanges > 0.5")
      df = df.loc[df.names.isin(expressed_receptors)]
      df['subtype'] = [st] * df.shape[0]
      diffrx[st] = df.copy()
          

  background_set = np.unique(adata.obs[adata.obs[subtype_col]==sender][broadtype_col])[0]

  send_ad = adata[adata.obs[broadtype_col].isin(background_set), adata.var_names.isin(ligands)]
  sc.tl.rank_genes_groups(send_ad, groupby=subtype_col, method='wilcoxon', n_genes=send_ad.shape[1])
  sender_df = sc.get.rank_genes_groups_df(send_ad, sender)
  sender_df = sender_df.query("logfoldchanges > 0.5 & pvals_adj < 0.05")
  sender_ligands = sender_df.names.tolist()

  # Track the matched ligand-receptors 
  interactions = {}
  for receiver in receivers:
    # receptor_set = diffrx[receiver].names.values[:N]
    receptor_set = diffrx[receiver].names.values
    matched = ligand_receptor.loc[ligand_receptor.receptor.isin( receptor_set )]
    ligand_set = np.unique(matched['ligand'].values)
    lx = adata[:, adata.var_names.isin(ligand_set)]

    dx = pd.DataFrame(lx.X.toarray() > 0, index=adata.obs_names, columns=lx.var_names)
    dx[subtype_col] = adata.obs[subtype_col]

    pct = dx.groupby(subtype_col).mean() > percent_cutoff

    hits = pct.columns[pct.loc[sender]].tolist()
    hits = [h for h in hits if h in sender_ligands]

    channels = []
    for h in hits:
      rr = matched.loc[matched.ligand == h].receptor.values
      for r in rr:
        if r in receptor_set:
          channels.append(f'{h}_{r}')
                
    # channels = sorted([c for c in channels if c in visium_channels])
    
    interactions[receiver] = channels
    logger.info(f'{sender} --> {receiver} {channels} {len(channels)}')

    return interactions





def get_receptor_color(rdxe, receptor, receiver, cmap):
  # Change color scale
  # max_val = np.max(rdxe.loc[:, receptor].values)
  max_val = np.max(rdxe.values)
  bins = np.linspace(0, max_val, 10)
  b = np.digitize(rdxe.loc[receiver, receptor], bins, right=True)
  color = cmap[b]
  color = f','.join([f'{c}' for c in color])
  return color
  

  
blues = [tuple(int(ch * 255) for ch in c) for c in sns.color_palette('Blues', 10)]
def write_receptor_karyotype(interactions, f, hlf, txtf, start, semi_circle, 
                             cmap=blues):
  logger = get_logger()

  # Also keep track of where to place ligands
  ligand_order = []
  
  # And coordinates for the links
  receptor_coords = {}
  
  total_receptors_with_repeats = 0
  for receiver, channels in interactions.items():
    receiver_receptors = set([x.split('_')[1] for x in channels])
    total_receptors_with_repeats += len(receiver_receptors)

  logger.info(f'{total_receptors_with_repeats}')
  area_per_receptor = int(semi_circle / total_receptors_with_repeats)
  logger.info(f'{area_per_receptor}')

  celltype_start = start
  receptor_start = start
  for receiver, channels in interactions.items():
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
      
      color = get_receptor_color(rdxe, receptor, receiver, cmap)
      
      hl = f'{receiver} {receptor_start} {receptor_end} fill_color={color}\n'
      hlf.write(hl)
      
      txt = f'{receiver} {receptor_start} {receptor_end} {receptor} color={color}\n'
      txtf.write(txt)
      
      receptor_start = receptor_end
    # The value of receptor start is now where our current cell type should stop
    color = ','.join([f'{v}' for v in hex2rgb(master_palette[receiver])])
    celltype_line = f'chr - {receiver} {receiver} {celltype_start} {receptor_start} {color}\n'
    f.write(celltype_line)
    # start for the next cell type
    celltype_start = receptor_start
      
  return ligand_order, receptor_coords


def get_ligand_color(sdxe, ligand, sender, cmap):
  # Change color scale
  max_val = np.max(sdxe.values)
  bins = np.linspace(0, max_val, 10)
  b = np.digitize(sdxe.loc[sender, ligand], bins, right=True)
  color = cmap[b]
  color = f','.join([f'{c}' for c in color])
  return color


# Find and place ligands 
reds = [tuple(int(ch * 255) for ch in c) for c in sns.color_palette('Reds', 10)]
def write_ligands(hlf, txtf, start, sdxe, total_ticks, ligand_order, cmap=reds):
  logger = get_logger()

  # Track ligand positions on the sender
  ligand_coords = {}
  
  n_ligands = len(ligand_order)
  logger.info(f'{n_ligands}')
  borders = np.linspace(start, total_ticks, n_ligands+1, dtype=int)
  # Go in the reverse order that we wrote the receptors:
  for i, ligand in enumerate(ligand_order[::-1]):
    ligand_start = borders[i]
    ligand_end = borders[i+1]
    color = get_ligand_color(sdxe, ligand, sender, reds)
    hl = f'{sender} {ligand_start} {ligand_end} fill_color={color}\n'
    txt = f'{sender} {ligand_start} {ligand_end} {ligand} color={color}\n'
    
    ligand_coords[ligand] = (ligand_start, ligand_end)

    hlf.write(hl)
    txtf.write(txt)
  
  return ligand_coords



def draw_links(interactions, linkf, sdxe, rdxe, 
               ligand_coords, 
               receptor_coords, 
               receiver_order=None,
               logger=None):

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
          
  logger.info(f'{max_interaction}') 
  for receiver in receiver_order:
    channels = interactions[receiver]
    
    color = ','.join([f'{v}' for v in hex2rgb(master_palette[receiver])])
    
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
      
      link = f'{sender} {l0} {l1} {receiver} {r0} {r1} color={color},{trx}\n'
      
      logger.info(link)
      linkf.write(link)




