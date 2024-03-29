<!-- # **C**ell **C**ell **SIG**naling -->

<!-- ![graph](assets/graph.svg) -->
<p align="center">
  <img width="820" height="340" src="assets/graph.svg">
  <br>
  <b>Cell-Cell signaling analysis in python</b><br>
</p>


## Installation

For now, just clone this repository and and run the pipeline scripts. 

```bash
git clone https://github.com/nathanin/ccsig
```

Notable dependencies:
- [`anndata`](https://anndata.readthedocs.io/en/latest/anndata.AnnData.html) and [`scanpy`](https://scanpy.readthedocs.io/en/stable/) to read and manipulate single cell expression data.
- standard scientific python libraries including `numpy`, `pandas`, `scipy`, `scikit-learn`, `scipy`, `matplotlib`, and `seaborn`

- [`ray`](https://github.com/ray-project/ray) for parallelization
- [`circos`](http://circos.ca/) (tested with `circos | v 0.69-8 | 15 Jun 2019 | Perl 5.022001`)

Tested on ubuntu 16.04 and 18.04.

Minimal conda config file coming soon.

---

## Pipelines

A brief description of the individual piplines is provided here.
To run each pipeline, locate the `*_run.py` script and run it. 
The inputs listed here are things that probably need to be gathered or prepared beforehand.
For more details see the README within each pipeline directory.

---

### Build Gene Sets Pipeline

Find genes downstream of a starting receptor using a weighted directed protein interaction graph, and a weighted, directed gene regulatory graph.

inputs | format
-------|--------
list of receptors | newline delimited list receptors to find genesets for
weighted protein-protein interaction graph | a table with columns for sender, receiver, and weight, use gene names
weighted gene regulatory network | a table with directed and weighted gene regulatory relationships

outputs | format
-------|--------
receptor gene sets | a directory populated with `*.txt` files, one per receptor, each a newline delimited listing of predicted downstream genes

---

### Receptor scoring Pipeline

Apply a gene set score across single cells by testing the average deviation from a background distribution drawn from similar cells. 

inputs | format
-------|--------
single cell gene expression | AnnData
cell type annotation | column in `obs`
receptor associated gene sets | directory of `*.txt` files with one gene list per file, newline delimited

outputs | format
-------|--------
single cell receptor activity scores | AnnData

---

### Interaction Test Pipeline

Test the interaction potential on ligand-receptor channels between cell types. 

inputs | format
-------|--------
single cell gene expression | AnnData
single cell receptor activity scores | AnnData
cell type annotation | column in `obs` indicating cell types to test
sample annotation | column in `obs` indicating individual samples
ligand-receptor interactions | a pickled dictionary: `{'receptor': ['ligand1', 'ligand2']}`

outputs | format
-------|--------
interaction potentials | pickled `pd.DataFrame`s with nonzero elements where the interaction passes a permuation test

