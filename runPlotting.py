import sys
import os
import numpy as np
import spectrumPlotting
import argparse
import pandas as pd
from tqdm.auto import tqdm


def arg_translate(a, v):
    # "Quasicount scaling" : 'qausiScale'
    # "mzml text"
    d = {
        "Filename" : "in_filename",
        "Spectrum polarity" : "polarity",
        "Precursor m/z" : "precursor_mz",
        "MS/MS Scan Index" : "index",
        "Output directory path" : "outdir",
        "Output filename" : "filename",
        "Spectrum starting index" : "start",
        "Raw count noise intensity cutoff" : "cutoff",
        "Number of labeled peaks" : "peaks",
        "Parent peak formula" : "formula",
        "Formula assignment m/z tolerance": "tolerance",
        "Quasicount scaling" : "quasiScale",
        "Quasicount function scale" : 'quasiX',
        "Quasicount function exponent" : 'quasiY',
        "Quasicount intensity cutoff" : 'quasiCutoff'
    }
    ret_arg = d[a.strip()]
    if ret_arg == 'in_filename':
        if v.lower().endswith('.mzml'):
            ret_arg = 'mzML'
        elif v.lower().endswith('.txt'):
            ret_arg = "text"
        else:
            sys.exit(f'Error - unrecognized file extension for input file {v}')
    ret_val = v if str(v) != 'nan' else None
    if ret_arg == "quasiScale" and ret_val:
        ret_val = {'true' : True, 'false' : False, '1.0' : True, '0.0' : False, '1' : True, '0' : False}[str(ret_val).lower()]
        ret_val = '' if ret_val else None
    if ret_arg in ["index", "peaks"] and ret_val:
        ret_val = int(ret_val)
    return((ret_arg, ret_val))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xlsx', required = True)
    parser.add_argument('--sheet_name', required = True)
    args = parser.parse_args()

    in_lines = pd.read_excel(args.xlsx, sheet_name=args.sheet_name)
    for i, row in tqdm(in_lines.iterrows(), total = len(in_lines)):
        arg_dict = {arg_translate(k, v)[0] : arg_translate(k, v)[1] for (k, v) in row.to_dict().items()}
        arg_dict = {k : v for (k,v) in arg_dict.items() if ( v  or ( k == 'quasiScale' and v == ''))}
        arg_str = []
        for k,v in arg_dict.items():
            arg_str.append(f'--{k}')
            if v != '':
                arg_str.append(f'{v}')
        parser = spectrumPlotting.create_parser()
        args = parser.parse_args(arg_str)
        spectrumPlotting.process(**vars(args))
        

if __name__ == '__main__':
    main()