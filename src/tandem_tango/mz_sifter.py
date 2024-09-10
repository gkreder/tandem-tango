import inspect
import os
from tqdm.auto import tqdm
import pandas as pd
from pyteomics import mzml as pyt_mzml
import argparse
from typing import List, Dict
import logging
from tandem_tango.utils.spectrum_operations import (
    get_quasi_counts,
    filter_spectrum_quasi,
    get_mz_match_list
)


def read_txt(spectrum_indices : str):
    with open(spectrum_indices, 'r') as f:
        sids = [x.strip() for x in f.readlines()]
    sids = [x for x in sids if x != '']
    return sids


def single_files(mzml_in : str, spectrum_indices : List[int],
                 out_dir : str, quasi_x : float,
                 quasi_y : float, quasi_cutoff : float = 2.0,
                 filter_ms2 : bool = True):
    logging.info("Running single file generation")
    os.makedirs(out_dir, exist_ok=True)
    # TODO change the implementation to get SIDs from file before calling this function
    with pyt_mzml.MzML(mzml_in) as f:
        for sid in tqdm(spectrum_indices):
            spec = f.get_by_index(sid)
            if filter_ms2 and spec['ms level'] != 2:
                logging.warning(f"Skipping spectrum {sid} as it is not MS2")
                continue
            quasi_ints = get_quasi_counts(spec, quasi_x, quasi_y)
            spec['quasi array'] = quasi_ints
            spec_filtered = filter_spectrum_quasi(spec, quasi_cutoff)
            out_file = os.path.join(out_dir, f"{sid}.tsv")
            with open(out_file, 'w') as f:
                print(f"mz\tIntensity", file=f)
                for mz, quasi_intensity in zip(spec_filtered['m/z array'], spec_filtered['quasi_array']):
                    print(f"{mz}\t{quasi_intensity}", file=f)
    logging.info("Done")

def get_out_col(spec : Dict, match_mzs : List[float], quasi_x : float, quasi_y : float,
                match_acc : float, quasi_cutoff : float = 2.0, filter_ms2 : bool = True):
    
    if filter_ms2 and spec['ms level'] != 2:
        logging.warning(f"Skipping spectrum {spec['index']} as it is not MS2")
        return [[f"MS{spec['ms level']}" for x in spec['m/z array']]]
    quasi_ints = get_quasi_counts(spec, quasi_x, quasi_y)
    spec['quasi array'] = quasi_ints
    spec_filtered = filter_spectrum_quasi(spec, quasi_cutoff)
    matched_intensities = get_mz_match_list(spectrum = spec_filtered,
                                            match_mzs = match_mzs,
                                            match_acc = match_acc,
                                            intensity_key = 'quasi array')
    return matched_intensities

def write_out_tsv(out_tsv : str, spec_idxs : List[int], mzs : List[float], out_cols : List[List[float]]):
    with open(out_tsv, 'w') as f:
        s_temp = '\t'.join([str(x) for x in spec_idxs])
        print(f"mz\t{s_temp}", file=f)
        for i_mz, mz in enumerate(mzs):
            s_temp = f"{mz}"
            for i_col, col in enumerate(out_cols):
                if i_mz < len(col):
                    s_temp += f"\t{col[i_mz]}"
                else:
                    s_temp += ""
            print(s_temp, file = f)

def mz_filtered(mzml_in : str, spec_idxs : List[int], out_tsv : str, mzs : List[float],
                tol : float, quasi_x : float, quasi_y : float, quasi_cutoff : float = 2.0,
                filter_ms2 : bool = True):
    
    logging.info("Running mz filtered generation")
    # TODO change the implementation to get SIDs from file before calling this function
    out_cols = []
    with pyt_mzml.MzML(mzml_in) as f:
        for spec_idx in spec_idxs:
            spec = f.get_by_index(spec_idx)
            if filter_ms2 and spec['ms level'] != 2:
                logging.warning(f"Skipping spectrum {spec_idx} as it is not MS2")
                continue
            out_col = get_out_col(spec = spec, match_mzs = mzs,
                                  quasi_x = quasi_x, quasi_y = quasi_y,
                                  match_acc = tol, quasi_cutoff = quasi_cutoff,
                                  filter_ms2 = filter_ms2)                            
            out_cols.append(out_col)
    write_out_tsv(out_tsv, spec_idxs, mzs, out_cols)
    df_out = pd.read_csv(out_tsv, sep = '\t')
    q_sums = df_out.replace(r"MS.* spectrum", 0.0, regex = True).drop(labels = ["m/z"], axis = 1).sum(axis = 1)
    df_out['quasiCount_Sum'] = q_sums
    df_out.t_csv(out_tsv, sep = '\t', index = False)
    logging.info("Done")

def multi_mz_filtered(tsv_in : str, out_dir : str, mzs : List[float], tol : float, quasi_x : float, 
                      quasi_y : float, quasi_cutoff : float = 2.0,
                      filter_ms2 : bool = True):
    logging.info("Running multi mz filtered generation")
    os.makedirs(out_dir, exist_ok=True)
    # TODO change implementation to get MZs from file before calling this function
    with open(tsv_in, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            mzml_in = line[0]
            spec_idxs = [int(x) for x in line[1:]]
            out_cols = []
            with pyt_mzml.MzML(mzml_in) as f_pyt:
                for spec_idx in spec_idxs:
                    spec = f_pyt.get_by_index(spec_idx)
                    out_col = get_out_col(spec = spec, match_mzs = mzs, 
                                          quasi_x = quasi_x, quasi_y = quasi_y,
                                          match_acc = tol, quasi_cutoff = quasi_cutoff, 
                                          filter_ms2 = filter_ms2)
                    out_cols.append(out_col)
            mzml_pref = ".".join(os.path.basename(mzml_in).split(".")[0 : -1])
            out_tsv = os.path.join(out_dir, f"{mzml_pref}.tsv")
            write_out_tsv(out_tsv, spec_idxs, mzs, out_cols)
            df_out = pd.read_csv(out_tsv, sep = '\t')
            q_sums = df_out.replace(r"MS.* spectrum", 0.0, regex = True).drop(labels = ["m/z"], axis = 1).sum(axis = 1)
            df_out['quasiCount_Sum'] = q_sums
            df_out.to_csv(out_tsv, sep = '\t', index = False)
    logging.info("Done")

def prep_args(fun : callable, args : Dict):
    sig = inspect.signature(fun)
    if args['no_ms2_filter']:
        fun_args['filter_ms2'] = False
    fun_args = {k : v for k, v in args.items() if k in sig.parameters}
    return fun_args

def main():

    parser = argparse.ArgumentParser(description="mz-sifter: Spectrum filtering and inspection")
    subparsers = parser.add_subparsers()
    
    # create the parser for the "tsvs" command
    parser_tsvs = subparsers.add_parser('tsvs')
    parser_tsvs.add_argument("mzml_in")
    parser_tsvs.add_argument("spectrum_indices")
    parser_tsvs.add_argument("quasi_x", type = float)
    parser_tsvs.add_argument("quasi_y", type = float)
    parser_tsvs.add_argument("out_dir")
    parser_tsvs.add_argument("--no_ms2_filter", action = 'store_true', help = 'Turn off filter checking if a spectrum is MS2 before processing')
    parser_tsvs.add_argument('--quasi_cutoff', default = 2, type = float)
    parser_tsvs.set_defaults(func=single_files)

    # create the parser for the "mzFiltered" command
    parser_mf = subparsers.add_parser('mz_filtered')
    parser_mf.add_argument("mzml_in")
    parser_mf.add_argument("spectrum_indices")
    parser_mf.add_argument("mzs")
    parser_mf.add_argument("quasi_x", type = float)
    parser_mf.add_argument("quasi_y", type = float)
    parser_mf.add_argument("out_tsv")
    parser_mf.add_argument("--no_ms2_filter", action = 'store_true', help = 'Turn off filter checking if a spectrum is MS2 before processing')
    parser_mf.add_argument('--quasi_cutoff', default = 2, type = float)
    parser_mf.add_argument("--tolerance", type = float, default = 0.01, help = 'm/z matching tolerance [default: 0.01]')
    parser_mf.set_defaults(func=mz_filtered)

    # create the parser for the "multiMzFiltered" command
    parser_mmf = subparsers.add_parser('multi_mz_filtered')
    parser_mmf.add_argument("tsv_in")
    parser_mmf.add_argument("mzs")
    parser_mmf.add_argument("quasi_x", type = float)
    parser_mmf.add_argument("quasi_y", type = float)
    parser_mmf.add_argument("out_dir")
    parser_mmf.add_argument("--no_ms2_filter", action = 'store_true', help = 'Turn off filter checking if a spectrum is MS2 before processing')
    parser_mmf.add_argument('--quasi_cutoff', default = 2, type = float)
    parser_mmf.add_argument("--tolerance", type = float, default = 0.01, help = 'm/z matching tolerance [default: 0.01]')
    parser_mmf.set_defaults(func=multi_mz_filtered)
    
    args = parser.parse_args()
    fun_args = prep_args(args.func, vars(args))
    args.func(**fun_args)

if __name__ == "__main__":
    main()
    