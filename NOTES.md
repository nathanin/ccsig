# Goals

Want a set of functions that enables the following interactions:

Want this to run on desktop but easily transfer to a job scheduler i.e. Univa

## Data

"core input"

- adata : (cells x genes) AnnData GEX
- radata : (cells x receptor) scores AnnData receptor scores (optional)

- celltypes : (cells x 1) cell type annotation
- samples : (cells x 1) sample annotation, serving as a constraint / replicates marker for interacting cells

- receptors : a list
- ligands : a list
- p_matrix : a matrix R x L of adjacencies between receptors and ligands


receptor_genelists : a directory with text-based lists of genes corresponding to downstream targets of each receptor


membrane_proteins : a list of annotated membrane-bound proteins
secreted_proteins : a list of annotated secreted proteins

----
----

## Analysis Methods

- If radata is not given , provide a way to compute receptor scores painlessly something like,

``` 
> radata = score_receptors(adata, receptor_genelists, **kwargs)

$ ccsig score_receptors /path/to/adata /path/to/receptor_genelists /out/path args
```

--- 

1. <ins>Single channel interactions with permutation testing</ins>
  - In the usual case, we have all of the core input given, and receptor scores are already calculated on our cells
  - Using a single ligand / receptor set (in the simplest case), compute the interaction potential between celltypes on this channel. 
  The calculations extend directly to include multiple channels at a time, if desired.
  - We'll want to basically run an AND gate between every cell type per sample to identify who has the strongest expression of the receptor, and who has the strongest expression of the ligand. In samples where there are celltypes which clearly (specifically ?) are activating that receptor, can we attribute the activation to a signaling partner producing ligand at above-average levels? 
  - Interactions at single cell level are subject to dropout, probably leading to under-calling
  - Interactions at the cell type level are subject to washout - an averaging effect, probably leading to over-calling
  - First do single cell interactions, then average the potentials over permutations , per cell type? 
  **Is this mathematically equal to first averaging expression, and calculating potentials from the averages?**
  
  - The algorithm is: 
    1. Compute the test (`bid`) interactions using unique combinations of a `celltype` and `sample` annotation. 
    2. Perform N permutations of the `celltype` label within each unique value of `sample` i.e. do not shuffle labels across samples, each time computing a `null` interaction value.
    3. Accept or reject specific combinations of the `bid` interactiion values based on a threshold percentile. 
    4. Output the full, unadjusted `bid` interactions, a `pvalue` for the bids corresponding to the portion of `null` interactions found with greater value than `bid`, and the (final) adjusted interaction values which passed the threshold.

  - The output for a dataset of M distinct `celltype`, `sample` combinations, an M x M pd.DataFrame where columns are interpreted as "sending" types (they bring the ligand) and rows are the "receiving" types (they bring the receptor)
  - The interaction potential value for M_j --> M_i is the convovled co-expression of the receptor activity in the M_i-th celltype, with the total ligand expression in the M_j-th sending celltype. 
  Thus, this method is best at single channel interactions and a complementary analysis is needed to resolve multi-channel interaction potentials.


```
itx, itx_unadj, itx_pval = interaction_potential(adata, radata, channels, p_matrix)
```

2. <ins>Receptor / Ligand Congruency</ins>
  - An alternative way to braodly query the interactivity between celltypes at a multi-channel level
  - In this analysis the channels are kept separate and summarized at an expressed / not-expressed level
  - Receptors are filtered by relative activity among the input cell types, optionally controlled for broad-cell category.
  An example is cell type annotation at Tcell subtype level, with a control category of all immune cells as the reference population.
    - This is done by a non-parametric differential expression test (Wilcoxon test)
  - A top-N of these receptors are kept and termed the `definitive set` of receptors for that cell type
  - Overlaps between `definitive sets` of various cell types are permitted
  - **An alternative approach** would be to perform a tiered expression-permutation test on the receptor scores first checking for a minimum percent of cells per group expressing the receptor, then performing an enrichment calcuation for that receptor relative to some background. 
  The `definitve set` of receptors would thence be all those receptors that pass these two filters.

  - The algorithm is this:
    1. Identify a `receptor_set` of definitive receptors with method A or B
    2. For each cell type, find `expressed_ligands` with a two-step expression/enrichment filter
    3. For each pair of cell types, and some form of `p_matrix`, count the number of matched receptor-ligands between the cell types. 
    This is the Receptor / Ligand congruency.
    4. Apply another permutation test to test for specifically matching cell type pairings. 
    The permutation in this case repeats steps 2 and 3. 

```

```

----
----

## Plotting Methods

Functions providing graphical feedback from the analysis


---
---
 ## Utilities

 A list of utility functions

 - Fast query a recetpor / ligand match