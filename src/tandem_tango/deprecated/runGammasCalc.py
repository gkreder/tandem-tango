############################################################
import sys
import os
import argparse
import numpy as np
from tqdm.auto import tqdm
import gammasCalc
############################################################
parser = argparse.ArgumentParser()
parser.add_argument("--tsv", required = True)
parser.add_argument("--verbose", action = "store_true")
args = parser.parse_args()
verbose = args.verbose
############################################################

with open(args.tsv, 'r') as f:
    lines = f.readlines()

for i_line, line in enumerate(tqdm(lines)):
    if i_line == 0:
        header = line.split('\t')
        continue
    args = line.split('\t')
    cmd_args = []
    for (h, a) in zip(header, args):
        if a not in ["", '\n']:
            cmd_args.append(f"--{h}")
            cmd_args.append(f"{a}")
    cmd_str = " ".join(cmd_args)
    if verbose:
        print(f"{cmd_str}")
    run_args = gammasCalc.get_args(cmd_str)
    gammasCalc.main(run_args)
    
