```
usage: query_nichenet_receptors.py [-h] [--weighted_lr_sig WEIGHTED_LR_SIG]
                                   [--weighted_gr WEIGHTED_GR] [--steps STEPS]
                                   [--n_sig_partners N_SIG_PARTNERS]
                                   receptors_fname output_dir

positional arguments:
  receptors_fname       newline-delimited list of receptors to search
  output_dir            someplace to stash the results. it will be created in
                        case it does not exist.

optional arguments:
  -h, --help            show this help message and exit
  --weighted_lr_sig WEIGHTED_LR_SIG
                        a three-column table with from, to, and weight columns
                        representing directed connections between proteins
  --weighted_gr WEIGHTED_GR
                        a three-column table with from, to, and weight columns
                        representing gene regulatory relationships
  --steps STEPS         the number of moves away from each start node to
                        consider. default=2
  --n_sig_partners N_SIG_PARTNERS
                        the maximum number of elements to take at each step.
                        default=10
```