################################################################################
# gk@reder.io
################################################################################
import os
import sys
import inspect
from typing import List

import argparse
import logging

from tandem_tango.utils.input_types import (
    peak_list_input,
    formula_input,
    logging_level_input,
    join_types_input,
    suffix_input,
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
    parser.add_argument("--mzml_1", required = True, type = str, help = "Path to mzML file 1")
    parser.add_argument("--mzml_2", required = True, type = str, help = "Path to mzML file 2")
    parser.add_argument("--index_1", required = True, type = int, help = "Index of the spectrum in mzML file 1")
    parser.add_argument("--index_2", required = True, type = int, help = "Index of the spectrum in mzML file 2")
    parser.add_argument("--quasi_x", required = True, type = float, help = "The quasicount scaling function x value")
    parser.add_argument("--quasi_y", required = True, type = float, help = "The quasicount scaling function y value")
    parser.add_argument("--R", required = True, type = float, help = "The R parameter (match_accuracy = 100.0/R, resolution_clearance = 200.0/R, subformula_tolerance = 100.0/R)")
    parser.add_argument("--out_dir", required = True, type = str, help = "Output directory")
    
    # Parent formula is optional but recommended
    parser.add_argument("--parent_formula", type = formula_input)
    
    # Optional parameters involving logging, script behavior, and plotting
    parser.add_argument("--verbosity", default = logging.INFO, type = logging_level_input, help = "Logging verbosity")
    parser.add_argument("--log_file", default = True, type = bool, help = "If True, save log to file")
    parser.add_argument("--log_filename", default = "spectrum_matching.log", type = str, help = "Name of the log file (saved to output directory)")
    parser.add_argument("--out_prefix", default = "spectrum_comparison", type = str, help = "Prefix to append to output files")
    parser.add_argument("--starting_index", default = 0, type = int, help = "The starting index (first spectrum) of the mzML file. This is 1 for Agilent-generated mzML files")
    parser.add_argument("--gain_control", default = False, type = bool, help = "If True, perform gain control adjustment")
    parser.add_argument("--log_plots", default = True, type = bool, help = "If True, create log intensity plots")
    parser.add_argument("--join_types", type = join_types_input, default = ["Intersection", "Union"], help = "Comma-separated list of types of spectrum joins to perform - (e.g. Intersection, Union)")
    parser.add_argument("--suffixes", type = suffix_input, default = ['A', 'B'], help = "Comma-separated list of suffixes for the spectra (e.g. A,B treats the first spectrum as spectrum 'A' and the second as spectrum 'B')")
    parser.add_argument("--plot_prefixes", type = suffix_input, default = ['Spectrum 1', 'Spectrum 2'], help = "Comma-separated list of spectrum prefixes for plot titles")
    
    # Optional spectrum filtering parameters
    parser.add_argument("--abs_cutoff", type = float, default = None, help = "Absolute intensity cutoff")
    parser.add_argument("--rel_cutoff", type = float, default = None, help = "Relative intensity cutoff")
    parser.add_argument("--quasi_cutoff", type = float, default = 5.0, help = "Quasicount intensity cutoff")
    parser.add_argument("--min_total_peaks", type = int, default = 2, help = "Minimum number of peaks required in a spectrum")
    parser.add_argument("--min_spectrum_quasi_sum", type = float, default = 20.0, help = "Minimum quasicount intensity sum required in a spectrum")
    parser.add_argument("--exclude_peaks", type = peak_list_input, default = None, help = "List of peaks to exclude from the spectra")
    parser.add_argument("--predefined_peaks", type = peak_list_input, default = None, help = "List of predefined peaks to filter for in the spectra")

    # Chemical formula fitting parameters
    parser.add_argument('--du_min', type = float, default = -0.5, help = "The DUMin value for molecular formula fitting")

    return parser

################################################################################
# Main spectrum matching function
################################################################################
def run_matching(parent_mz : float, mzml_1 : str, mzml_2 : str, 
                 index_1 : int, index_2 : int, quasi_x : float,
                 quasi_y : float, R : float, out_dir : str, 
                 parent_formula : str = None, out_prefix : str = None, 
                 starting_index : int = 0, gain_control : bool = False,
                 log_plots : bool = True, join_types : List[str] = ['Intersection', 'Union'],
                 suffixes : List[str] = ['A', 'B'], plot_prefixes : List[str] = ['Spectrum 1', 'Spectrum 2'],
                 abs_cutoff : float = None,
                 rel_cutoff : float = None, quasi_cutoff : float = None,
                 min_total_peaks : int = None, min_spectrum_quasi_sum : float = None,
                 exclude_peaks : List[float] = None, predefined_peaks : List[float] = None,
                 du_min : float = -0.5, log_filename : str = None, verbosity : int = logging.INFO):
    
    # Set internal function parameters based on R
    match_acc = 100.0 / R
    res_clearance = 200.0 / R
    subformula_tolerance = 100.0 / R

    os.makedirs(out_dir, exist_ok = True)
    if log_filename is not None:
        logging_handlers = [logging.FileHandler(os.path.join(out_dir, log_filename), mode='w')]
        logging.basicConfig(level=verbosity, format='%(asctime)s - %(levelname)s - %(message)s',
                                handlers=logging_handlers)
    logging.info("Reading spectra")
    spectra = get_spectra_by_indices([mzml_1, mzml_2], [index_1 - starting_index, index_2 - starting_index], gain_control)
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
            logging.error(f"Spectrum {[mzml_1, mzml_2][i_spectrum]} scan {[index_1, index_2][i_spectrum]} failed post-filtering validation:\n\t{e}")
            sys.exit()

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
    df_stats, df_intersection, df_union, spectra_df = get_results_dfs(spectra, metrics, parent_mz, quasi_x, quasi_y, parent_formula, suffixes)
    logging.debug(f"Stats:\n {df_stats}")
    logging.debug(f"Intersection:\n {df_intersection}")
    logging.debug(f"Union:\n {df_union}")

    logging.info("Writing results to Excel")
    out_xlsx = os.path.join(out_dir, f"{out_prefix}.xlsx")
    write_results_xlsx(out_xlsx, df_stats, df_intersection, df_union, spectra_df)
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
                  plot_suffixes=suffixes,
                    out_dir=out_dir,
                    file_prefix=out_prefix,
                  log_plots=log_plots)

################################################################################
# For command line usage
################################################################################
def main():
    parser = get_parser()
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok = True)
    logging_handlers = [logging.StreamHandler()]
    if args.log_file:
        logging_handlers.append(logging.FileHandler(os.path.join(args.out_dir, args.log_filename), mode='w'))
    logging.basicConfig(level=args.verbosity, format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=logging_handlers)
    args_str = '\n'.join([f"\t{k} : {v}" for k,v in vars(args).items()])
    logging.info(f"Args: {args_str}")
    run_matching(**{k: v for k, v in vars(args).items() if k in inspect.signature(run_matching).parameters})

if __name__ == "__main__":
    main()
