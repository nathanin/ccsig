#!/usr/bin/env python

from itx_handler import itxHandler
import argparse
import pickle

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--database', type=str, required=True)
  parser.add_argument('--outf', type=str, required=True)
  parser.add_argument('--alpha', type=float, default=0.01)
  parser.add_argument('--disable_pbar', action='store_true', default=False)
  
  ARGS = parser.parse_args()

  scitx = itxHandler()
  scitx.create_from_db(ARGS.database)
  scitx.sc_background_test(alpha=ARGS.alpha, disable_pbar=ARGS.disable_pbar)

  print(f'{ARGS.database}: Done.\nWriting to {ARGS.outf}')
  pickle.dump(scitx.alpha_levels, open(ARGS.outf, 'wb'))