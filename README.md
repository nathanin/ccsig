# **C**ell **C**ell **SIG**naling

Command line scripts to perform intracellular signaling analysis.


## Installation

For now, just clone this repository and and run the pipeline scripts. 

```bash
git clone https://github.com/nathanin/ccsig
cd ccsig
pip install -r requirements.txt
```

We use and depend on `anndata` and `scanpy`.
The standard scientific python libraries we use including `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn` and `graphviz`. 
The parallelized workflows in this project were made possible by `ray`.

Tested on ubuntu 16.04 and 18.04.

## Pipelines

A brief description of the individual piplines is provided here.
To run each pipeline, locate the `*_run.py` script and run it. 
For more information see the README within each pipeline directory.

### Receptor scoring pipeline

