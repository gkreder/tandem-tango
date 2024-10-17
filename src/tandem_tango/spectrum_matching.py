################################################################################
# gk@reder.io
################################################################################
import os
import sys
import inspect
from typing import List
from pathlib import Path

import argparse
import logging
import pandas as pd

from tandem_tango.utils.input_types import (
    peak_list_input,
    formula_input,
    logging_level_input,
    suffix_input,
    bool_input,
    make_out_prefix,
)

from tandem_tango.utils.spectrum_operations import (
    get_spectra_by_indices,
    validate_spectrum_pair,
    filter_and_convert_spectrum_complete,
    validate_spectrum_counts,
    merge_spectra,
    SpectrumValidationError
)

from tandem_tango.utils.metrics import (
    calc_spectra_metrics,
    add_spectra_formulas
)

from tandem_tango.utils.reporting import (
    get_results_dfs,
    write_results_xlsx,
)

from tandem_tango.utils.plotting import summary_plots

################################################################################
# Arguments for spectrum matching
################################################################################
def get_parser():
    parser = argparse.ArgumentParser(description="Runs spectrum comparison between two spectra")

    # Required parameters
    parser.add_argument("--parent_mz", required = True, type = float, help = "Parent m/z value")

    # Mutually exclusive input groups for the first spectrum
    # input_1 = parser.add_mutually_exclusive_group(required=True)
    # input_1.add_argument("--mzml_1", type = str, help = "Path to mzML file 1")
    # input_1.add_argument("--mgf_1", type = str, help = "Path to MGF file 1")
    parser.add_argument('--file_1', type = str, help = "Path to mzML or MGF file 1")
    parser.add_argument("--index_1", required = True, type = int, help = "Index of the spectrum in mzML/mgf file 1")

    # Mutually exclusive input groups for the second spectrum
    # input_2 = parser.add_mutually_exclusive_group(required=True)
    # input_2.add_argument("--mzml_2", type = str, help = "Path to mzML file 2")
    # input_2.add_argument("--mgf_2", type = str, help = "Path to MGF file 2")
    parser.add_argument('--file_2', type = str, help = "Path to mzML or MGF file 2")
    parser.add_argument("--index_2", required = True, type = int, help = "Index of the spectrum in mzML/mgf file 2")

    parser.add_argument("--starting_index_1", default = 0, type = int, help = "The starting index (first spectrum) of the first mzML/mgf file is set to 0 by default, but may be 1 for certain vendor software such as Agilent MassHunter")
    parser.add_argument("--starting_index_2", default = 0, type = int, help = "The starting index (first spectrum) of the second mzML/mgf file is set to 0 by default, but may be 1 for certain vendor software such as Agilent MassHunter")
    
    parser.add_argument("--gain_control", default = False, type = bool_input, help = "If True, perform gain control adjustment")

    # Required parameters for spectrum matching
    parser.add_argument("--quasi_x", required = True, type = float, help = "The quasicount scaling function x value (m/z multiplicative constant)")
    parser.add_argument("--quasi_y", required = True, type = float, help = "The quasicount scaling function y value (m/z exponent; if set to 0, scaling function is a constant)")
    parser.add_argument("--R", required = True, type = float, help = "The resolution parameter R (match_accuracy = 100.0/R, resolution_clearance = 200.0/R, subformula_tolerance = 100.0/R)")
    parser.add_argument("--out_dir", required = True, type = str, help = "Output directory")
    
    # Parent formula is optional but recommended
    parser.add_argument("--parent_formula", type = formula_input)

    # Optional spectrum filtering parameters
    parser.add_argument("--abs_cutoff", type = float, default = None, help = "Absolute intensity cutoff (raw counts)")
    parser.add_argument("--rel_cutoff", type = float, default = None, help = "Relative intensity cutoff")
    parser.add_argument("--quasi_cutoff", type = float, default = 5.0, help = "Quasicount intensity cutoff")
    parser.add_argument("--min_total_peaks", type = int, default = 2, help = "Minimum number of peaks required in a spectrum")
    parser.add_argument("--min_spectrum_quasi_sum", type = float, default = 20.0, help = "Minimum quasicount intensity sum required in a spectrum")
    parser.add_argument("--exclude_peaks", type = peak_list_input, default = None, help = "List of peaks to exclude from the spectra")
    parser.add_argument("--predefined_peaks", type = peak_list_input, default = None, help = "List of predefined peaks to filter for in the spectra")

    # Chemical formula fitting parameters
    parser.add_argument('--du_min', type = float, default = -0.5, help = "The minimum allowed degrees of unsaturation for molecular formula fitting")
    
    # Optional parameters involving logging, script behavior, and plotting
    parser.add_argument("--verbosity", default = logging.INFO, type = logging_level_input, help = "Logging verbosity")
    parser.add_argument("--log_file", default = True, type = bool_input, help = "If True, save log to file")
    parser.add_argument("--log_filename", default = None, type = str, help = "Name of the log file (saved to output directory). Defaults to the out_prefix with a .log extension")
    parser.add_argument("--out_prefix", default = None, type = str, help = "Prefix to append to output files (e.g. `output` would create summary files named 'output.xlsx' in the output directory). Defaults to <filename_1>_Scan_<index_1>_vs_<filename_2>_Scan_<index_2>_<Formula/noFormula>")
    parser.add_argument("--log_plots", default = True, type = bool_input, help = "If True, create log intensity plots")
    parser.add_argument("--spectrum_suffixes", type = suffix_input, default = ['A', 'B'], help = "Comma-separated list of suffixes for the spectra (e.g. A,B treats the first spectrum as spectrum 'A' and the second as spectrum 'B')")
    parser.add_argument("--plot_prefixes", type = suffix_input, default = None, help = "Comma-separated list of spectrum prefixes for plot titles (e.g. `Spectrum 1,Spectrum 2`) will reference 'Spectrum 1' and 'Spectrum 2' in the displayed plot titles. Defaults to the base filenames of the input mzML/mgf files")

    return parser

# def prep_args(args : argparse.Namespace):
#     """Prepares the arguments for the spectrum matching function"""

#     # Get the input files and convert them to 'file' arguments instead of 'mzml'/'mgf'
#     input_files_all_formats = {k : v for k, v in vars(args).items() if k.startswith('mzml_') or k.startswith('mgf_')}
#     input_files = {f"file_{k.split('_')[-1]}" : v for k,v in input_files_all_formats.items() if v is not None}
#     if len(input_files) != 2:
#         raise ValueError("Exactly two input files must be specified")
#     modified_args_dict = {
#         **{k:v for k,v in vars(args).items() if k in inspect.signature(run_matching).parameters},
#         **input_files
#         }
#     return modified_args_dict

################################################################################
# Main spectrum matching function
################################################################################
def run_matching(file_1 : str, file_2 : str, 
                 index_1 : int, index_2 : int, parent_mz : float, 
                 quasi_x : float, quasi_y : float, R : float, 
                 out_dir : str, 
                 parent_formula : str = None,  
                 starting_index_1 : int = 0, starting_index_2 : int = 0,
                 abs_cutoff : float = None,
                 rel_cutoff : float = None, quasi_cutoff : float = None,
                 min_total_peaks : int = None, min_spectrum_quasi_sum : float = None,
                 du_min : float = -0.5, 
                 exclude_peaks : List[float] = None, predefined_peaks : List[float] = None,
                 gain_control : bool = False, log_plots : bool = True, out_prefix : str = None,
                 spectrum_suffixes : List[str] = ['A', 'B'], plot_prefixes : List[str] = None,
                 log_filename : str = None, verbosity : int = logging.INFO):
    
    # Set the output prefix if not specified
    if out_prefix is None:
        out_prefix = make_out_prefix(file_1, file_2, index_1, index_2, parent_formula)
    if log_filename is None:
        log_filename = f"{out_prefix}.log"
    if plot_prefixes is None:
        plot_prefixes = [Path(file).stem for file in [file_1, file_2]]
        

    # Set internal function parameters based on R
    match_acc = 100.0 / R
    res_clearance = 200.0 / R
    subformula_tolerance = 100.0 / R

    os.makedirs(out_dir, exist_ok = True)
    if log_filename is not None:
        logging_handlers = [logging.FileHandler(os.path.join(out_dir, log_filename), mode='w')]
        logging.basicConfig(level=verbosity, format='%(asctime)s - %(levelname)s - %(message)s',
                                handlers=logging_handlers,
                                force=True)
    
    logging.info(f"Run Matching called with arguments: {locals()}")
    logging.info("Reading spectra")
    spectra = get_spectra_by_indices([file_1, file_2], [index_1 - starting_index_1, index_2 - starting_index_2], gain_control)
    logging.info("Validating that spectra are valid and usable")
    validate_spectrum_pair(spectra)
    logging.info("Filtering and converting spectra")
    spectra_filtered = [filter_and_convert_spectrum_complete(spectrum, **{
        'abs_cutoff' : abs_cutoff, 
        'quasi_x' : quasi_x,
        'quasi_y' : quasi_y,
        'rel_cutoff' : rel_cutoff,
        'quasi_cutoff' : quasi_cutoff,
        'pdpl' : predefined_peaks,
        'exclude_peaks' : exclude_peaks,
        'match_acc' : match_acc,
        'parent_mz' : parent_mz,
        'res_clearance' : res_clearance,
        'sort_intensity' : True
    }) for spectrum in spectra]
    logging.info("Checking if spectra are valid after filtering")
    logging.debug(f"min_total_peaks: {min_total_peaks}, min_spectrum_quasi_sum: {min_spectrum_quasi_sum}")
    for i_spectrum, spectrum in enumerate(spectra_filtered):
        try: 
            validate_spectrum_counts(spectrum, min_spectrum_quasi_sum, min_total_peaks)
        except SpectrumValidationError as e:
            logging.error(f"Spectrum {[file_1, file_2][i_spectrum]} scan {[index_1, index_2][i_spectrum]} failed post-filtering validation:\n\t{e}")
            return None
    
    logging.info("Merging spectra by m/z matching")
    merged_spectrum = merge_spectra(spectra_filtered[0], spectra_filtered[1], match_acc)
    if parent_formula is not None:
        logging.info("Parent formula specified - Calculating formulas")
        formula_spectrum = add_spectra_formulas(merged_spectrum, parent_formula, subformula_tolerance, du_min, predefined_peaks)
    else:
        formula_spectrum = merged_spectrum.copy()
    logging.info("Calculating metrics")
    metrics = calc_spectra_metrics(formula_spectrum)
    logging.debug(f"Metrics:\n {metrics}")
    df_stats, df_intersection, df_union, spectra_df = get_results_dfs(spectra, metrics, parent_mz, quasi_x, quasi_y, parent_formula, spectrum_suffixes)
    logging.debug(f"Stats:\n {df_stats}")
    logging.debug(f"Intersection:\n {df_intersection}")
    logging.debug(f"Union:\n {df_union}")

    logging.info("Writing results to Excel")
    out_xlsx = os.path.join(out_dir, f"{out_prefix}.xlsx")

    # Capture function params for logging to output file
    func_params = inspect.signature(run_matching).parameters
    runtime_params = {k: v for k, v in locals().items() if k in func_params}
    df_params = pd.DataFrame(list(runtime_params.items()), columns=['Parameter', 'Value'])

    write_results_xlsx(out_xlsx, df_stats, df_intersection, df_union, spectra_df, df_params = df_params)
    logging.info("Creating summary plots")
    # The gray spectra are the original spectra that will be used in the plotting
    gray_spectra = [filter_and_convert_spectrum_complete(spectrum, **{
        'quasi_x' : quasi_x,
        'quasi_y' : quasi_y,
        'parent_mz' : parent_mz,
        'match_acc' : match_acc
    }) for spectrum in spectra]
    summary_plots(df_stats, df_intersection, df_union, gray_spectra, 
                  title_suffixes=plot_prefixes,
                  scan_indices=[index_1, index_2],
                  plot_suffixes=spectrum_suffixes,
                    out_dir=out_dir,
                    file_prefix=out_prefix,
                  log_plots=log_plots,
                  parent_mz = parent_mz,)
    logging.info("Completed spectrum matching")

################################################################################
# For command line usage
################################################################################
def main():
    """Runs spectrum comparison from the command line using passed arguments"""
    parser = get_parser()
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok = True)
    # Set up streaming log handler to show log in terminal in addition to saved log output
    logging_handlers = [logging.StreamHandler()]
    if args.log_file:
        logging_handlers.append(logging.FileHandler(os.path.join(args.out_dir, args.log_filename), mode='w'))
    logging.basicConfig(level=args.verbosity, format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=logging_handlers)
    # args_str = '\n'.join([f"\t{k} : {v}" for k,v in vars(args).items()])
    # logging.info(f"Args: {args_str}")
    
    # args_dict = prep_args(args)
    # run_matching(**args_dict)
    run_matching(**vars(args))

if __name__ == "__main__":
    main()
