# Store interaction potentials between single, individual cells

Create a database of cell-cell interactions at the single-cell level

To reduce the density of interactions, we'll impose a few constraints on the cells that interact

    1. For multi-sample datasets, we constrain interaction potentials to be non-zero only between cells from the same sample, i.e. physically collected at the same time from the same tissue
    2. Considering cross-cluster / "grouping" interactions only, though this can be relaxed 


Construct a database based on the `zarr` package (https://zarr.readthedocs.io/en/stable/index.html) to store interactions as 3D arrays.
Each array has a "sending" cell type on the rows, and "receiving" cell type on the columns, and the 3rd dimension represents an available Ligand/Receptor axis.
The construction of the Ligand/Receptor axis will depend on the database used.

As in the normal workflow, the interaction potentials are the product of count-normalized log-transformed counts of the ligand and receptor in single cells.

### Notation / ordering
Maintain throughout as left-to-right ordering denoting sending (ligand) on the "left" side of arguments and keywords, and rows of matrices, and receivers (receptors) to the "right" side of keywords and the columns of matrices.

For example, the interaction between CXCL13 and CXCR5 would be denoted in the ligand/receptor annotation as `CXCL13_CXCR5`, and the matrix would be constructed, roughly, as

```
Tcell barcodes - x x x x x x
                 x x x x x x 
                 x x x x x x 
                 x x x x x x 
                 |
                 B cell barcodes
```


## Processing interactions
Processing interactions should happen from a very high level function call. Something like this:

```python
from scseq_interactions import itxHandler
import scanpy as sc
import ray

adata = sc.read_h5ad("dataset.h5ad")
radata = sc.read_h5ad("receptor_scores.h5ad")

# load numpy arrays: lr_matrix, ligands, receptors

scitx = itxHandler()
scitx.create(adata, 
             lr_matrix=lr_matrix, 
             ligands=ligands, 
             receptors=receptors,
             radata=radata)

ray.init(num_cpus=4)
scitx.calc(groupby="celltype", outbase="output/path")


```

This type of command should launch everything we need to do and construct an interactions database under the destination path. 

We need to store metadata most importantly the cell barcodes and ligand/receptor annotation for all axes of the interaction matrices we compute.

## Recalling interactions
We want to build a workflow where raw interactions are calculated once, stored on disk persistently then recalled when desired for analysis.
Clearly the design of this strategy is on two fronts: the interface used to recall data (how do we want to refer to the interaction potential matrix of a single cell, or group of cells?), and the database structure that we descend into in order to retrieve that information.

On first pass, a recall workflow like this seems like a good place to start:

```python
from scseq_interactions import itxHandler

scitx = itxHandler()
scitx.create_from_db("database/path")
I,sbc,rbc = scitx.recall(sender='CD4T', receiver='CD8T', lr='CD40L_CD40')

"""
I.shape = (n,m,1)
sbc.shape = (n,)
rbc.shape = (m,)

sbc -> numpy array sender (row) barcodes
rbc -> numpy array receiver (column) barcodes
I -> CD4T - x x x x x x
            x x x x x x 
            x x x x x x 
            x x x x x x 
            |
            CD8T
"""
```

## Storing data
Practically we're faced with the problem of constructing a database of interaction potentials. 

Denote the interaction potentials for a particular cell type pairing $(n, m)$ in a particular sample $s$ as $I_{s(n,m)}$, which is a 3-dimensional array also denoted in index notation as $I_{k,n,m}$. 
We've made the choice to store each $I_{k,n.m}$ as its own array. This is with the notion of recalling blocks of interactions between celltypes per sample, possibly subsetting out individual interactions of interest at a time.

Keeping track of the `sample_id`, and the `grouping` (i.e. `celltype`) for the current experiment, we use `zarr` to keep all data pertaining to an experiment in one place:

<p align="center">
  <img src="assets/data-structure.png">
  <!-- <img width="200" height="340" src="assets/data-structure.png"> -->
  <br>
  <b>zarr data structure</b><br>
</p>

Notice how we keep the `lr_matrix`, indices for `receptors` and `ligands` as numpy arrays of strings, and ordered cell barcodes of each cell group, which can be used to identify single cells amongst the correct axis of the interaction matrices. Thus, we store in one place everything we need to interpret the results of a run.

## Number of interactions to compute

For dataset of a single sample with $X$ cells, we have $S$ celltypes, the number of cells per celltype as $X_s$, and $\sum_{s}{X_s} = X$.
Denote the cell type pairs $\forall (n \in S , m \in S)$ ($\forall (n, m)$) and the number of celltype pairs as $_SC_2$. If there are $K$ ligand/receptor axes to consider per cell type combination, 

The number of interactions to calculate per sample is:
$
I = K * \sum_{\forall (n, m)}X_n * X_m
$

<!-- Where $_nC_r$ is the number of combinations of $r$ items from $n$ total items:
$
    |\forall (n, m)| =\ _nC_r = \frac{n!}{r!(n-r)!}
$ -->



## Significance testing with an empirical background distribution
One major difference in the single-cell preserving workflow compared with the pooling based workflow is in downstream processing of interaction potentials, including assessing the significance of interactions.

Significance can still be determined based on a set of randomly drawn background cells. 
However, the choice of background is of particular interest as illustrated in the following example. Say we have a set of CD4T cells and CD8T cells that together compose 10% of our single cell dataset. The remainder of the data are 10% myeloid cells and 80% non-immune epithelial or stromal cells. If we were to consider the whole dataset as the background distribution, then many ligands and receptors that are expressed on CD8/CD4 Tcells will return as significantly interacting, when in reality we've simply compared Tcells with non-Tcells in terms of expected gene expression.
An alternative approach is to consider the distribution of interaction potentials in all pairs that include the sender/receiver cell populations. 
A potential advantage to this approach is that we simply need to calculate all pairwise interactions, which we would do anyway, then use the uninteresting cell type combinations as the background distribution.

Ultimately, interaction significance on the single-cell level comes down to asking if the L x R product between these two cells greater than can be expected by chance if we simply pull two cells at random from the dataset.
If two cells are the only cells in the whole dataset that express the correct L/R combination (equiv. when considering receptor scores), then yes those two cells are the most likely interactors, __given the rest of the cells in this dataset__.
If a L x R product is frequently non-zero, then we fall back to our base assumption that the cells with the highest product are the most likely to be interacting on this channel. 
Any two cells with non-zero L x R have potential for this interaction to be active. 
Part of what we want to test is if the variance in L x R is associated with spatial proximity of the cell types involved. If we can find __modes__ of interaction strengths based on scSeq, then find commensurate populations of proximal / non-proximal cells in spatial data, then we'll have built a level of confidence that the interaction potential makes sense on the single-cell level.

Consider the line of questioning centered around the receptor profiles of single cells, asking whether their cell-cell signaling activity has a significant effect on the receiving cell's overall cell state.
For example we have a strong expectation that a T-cell expressing a Th17 expression profile is receiving IL17 signaling from another cell in its microenvironment.
From the receiver cell's perspective, there is an external factor that is causing ("causing" in the non-rigorous sense) the particular expression profile we observe.
The task is then to determine which cells are sending the signal.
To test for signal sending potential, we compare the interaction potential along a signaling channel with a random assortment of potentials from all cells in the dataset.
If a particular cell x cell interaction is determined to be emperically enriched relative to the rest of possible senders (keeping the receiving population static), then we have found a likely candidate for signal sending. Job done.

Interaction matrices for all groups of sending ($m$) and receiving ($n$) cells: $\{I(m,n)\}_{m\in S, n\in S}$. Refer to a particular cell x cell pair with the indices $(i,j)$ and the interaction potential of those cells as $I(m,n)_{i,j}$. 
We want to check $I(m,n)_{i,j}$ against a background of all the sending populations: $\{I(m,n)\}_{m \in S}$.

```python
scitx = itxHandler()
scitx.create_from_db(path)
scitx.sc_background_test(alpha=0.01)
```



-----

## Post-processing
So far the workflow is to:
1. calculate raw interactions for all pairwise cells, divided by group
2. perform a significance assessment on the interactions, zeroing in on highly enriched interactions
3. ??

In the third step we'll want to start plotting summaries. Some questions:
1. How many significant interactions are found across the dataset?
2. How many significant interactions per cell type pair (CTP)?
3. For each L/R channel, how many cell types is that interaction significant in?
4. For each cell type pair what are the most enriched interaction channels?
5. Are there modes of interaction potential distributions? (correlate with spatial data: are there associated / non-associated populations of these cells in situ?)


------

## Cell-cell interaction potential graph
One advantage, maybe even the main point, of building interaction potentials at the single-cell level, is to eventually create a cell-cell interaction graph of single cells. The core concept is to use the interaction congruency between single cells as an "interactivity distance" metric, and to assign nearest-neighbors based on this interaction distance. With this graph, we can perform leiden (or louvain, both graph-based) clustering and computationally predict cellular niches based on groups of co-interacting cells. Another goal is to produce a UMAP visualization of the cell-cell interaction potential where the embedding represents cells with high interactivity (low distance) being near one another. UMAP works by modeling the inter-sample weight (proximity to neighbors) from the original data manifold in the low-dimensional space. So, if we're careful about what input we provide to UMAP we can have it produce a visualization of cell-cell interactivity for us. Ideally, cells with high interactivity will lie near each other in this UMAP, and we can use it to visualize the receptor/ligand expression, subtyping, etc. as normal. 

So, the goals for this section are:
1. Clusters of cell-cell interaction, representing cell types we expect to co-locate and therefore enhance inter-cellular signaling
2. A visualization tool to explore these predictions in the scSeq data

Both of our outcomes depend on an interactivity graph constructed from a faithful measure of interactivity distance, or Ligand-Receptor congruency:

Start with a cell pair, and all the interaction channels between them. We've used $I(m,n)_{i,j}$ in the past to represent this vector, though we can drop $(m,n)$ since these distances are pairwise across all cells in a dataset, and recall that we're really talking about a vector indexed by $k$ representing individual Ligand-Receptor channels: $I_{i,j,k}$.
Depending on the absolute expression (really the detectable levels of RNA), certain interaction channels will dominate when a highly detected RNA is involved, e.g. CD44. 
To mitigate this artifact, we normalize $I_{i,j,k}$ by either imposing a significance requirement ($ \textrm{norm}(I_{i,j,k}) = I_{i,j,k} > \textrm{sig}(I_{k}) $, discussed above), or by max-scaling the raw interaction potentials ($ \textrm{norm}(I_{i,j,k}) = I_{i,j,k} / \textrm{max}(I_k)$).

$
D_{i,j} = \frac{1}{ \sum_{k}^{K}{\textrm{norm}( I_{i,j,k}} )}
$

The other thing we need is an appropriate feature matrix. We essentially want cells with congruent ligand-receptor sets to have similar features. Futhermore, even though the notion of sending and receiving signals is one-way (naturally bipartite), we can only reprsent each cell once. So, if we have $K$ ligand-receptor channels, each represented by a pair of single genes (caveats apply), then we can use the expression of either of these genes in the same cell to reprsent the "direction agnostic potential" of the cell along this LR channel:

For cell $i$ and LR channel $k$, which expands to ligand $l$ and receptor $r$, at the gene-expression level given by the gene expression counts matrix $G$:

$
X_{i,k} = G_{i,l} + G_{i,r}
$

Using the [rapids.ai cuML package](https://rapids.ai/) for the GPU-accelerated UMAP:

```python
from cuml import UMAP 

from scseq_interactions import itxHandler
import scanpy as sc

scitx = itxHandler()
scitx.create_from_db(path)
scitx.set_adata(sc.read_h5ad(adata_path))

# adjacency matrix with distances
A = scitx.build_interaction_graph(n_neighbors=10)
X = scitx.build_directionless_features()

emb = UMAP(n_neighbors=10).fit_transform(X, knn_graph=A)

```