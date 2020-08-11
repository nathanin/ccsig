#!/usr/bin/env python

import numpy as np
import pandas as pd

import scrna
import pickle

import multiprocessing
from argparse import ArgumentParser

from interaction_score_def import run_interaction_test


parser = ArgumentParser()
parser.add_argument('output_file')
parser.add_argument('-j', default=3)
ARGS = parser.parse_args()

p = multiprocessing.Pool(ARGS.j)

rl_pairs = pickle.load(open( "matched_ligands_v4_F.p", "rb" ))
receptors = list(rl_pairs.keys())

ret = p.map(run_interaction_test, receptors)
p.close()
p.join()

all_colnames = pd.Series(np.unique(
  np.concatenate([list(I_tmp.columns) + list(I_tmp.index) for I_tmp in ret])
))

# Pad I's with 0
ret = {k: v for k, v in zip(receptors, ret)}
for k, I_tmp in ret.items():
    full_I = pd.DataFrame(index=all_colnames, columns=all_colnames)
    full_I.loc[:,:] = 0
    full_I.loc[I_tmp.index, I_tmp.columns] = I_tmp
    ret[k] = full_I

ret = np.stack([v for _, v in ret.items()], axis=-1)
print(f'saving {ret.shape} --> {ARGS.output_file}')

np.savez(ARGS.output_file, ret, np.array(all_colnames), np.array(receptors))