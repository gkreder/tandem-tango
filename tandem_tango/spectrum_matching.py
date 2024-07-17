import os

import argparse
import logging
import molmass
from pyteomics import mzml as pytmzml
from draft import *


################################################################################
def peak_list_input(value):
    try:
        if os.path.exists(value):
            with open(value, 'r') as f:
                lines = f.readlines()
            return [float(x.strip()) for x in lines]
        else:
            return [float(x) for x in value.split(",")]
    except:
        raise argparse.ArgumentTypeError(f"Could not parse peak list from {value}")
    
def formula_input(value):
    try:
        return molmass.Formula(value).formula
    except:
        raise argparse.ArgumentTypeError(f"Could not parse formula from {value}")
    
def get_logging_level(verbosity):
    levels = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }
    return levels.get(verbosity.lower(), logging.INFO)

def join_types_type(value):
    valid_types = ["Intersection", "Union"]
    types = [x.strip() for x in value.split(",")]
    for t in types:
        if t not in valid_types:
            raise argparse.ArgumentTypeError(f"Invalid join type {t}")
    return types

def suffix_type(value):
    suffixes = [x.strip() for x in value.split(",")]
    if len(suffixes) != 2:
        raise argparse.ArgumentTypeError(f"Two suffixes must be provided")
    return suffixes

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
    parser.add_argument("parent_formula", type = formula_input)
    
    # Optional parameters involving logging and script behavior
    parser.add_argument("--verbosity", default = "info", type = str, help = "Logging verbosity")
    parser.add_argument("--log_file", default = True, type = bool, help = "If True, save log to file")
    parser.add_argument("--log_filename", default = "spectrum_matching.log", type = str, help = "Name of the log file (saved to output directory)")
    parser.add_argument("--out_prefix", default = None, type = str, help = "Prefix to append to output files")
    parser.add_argument("--starting_index", default = 0, type = int, help = "The starting index (first spectrum) of the mzML file. This is 1 for Agilent-generated mzML files")
    parser.add_argument("--gain_control", default = False, action = "store_true", help = "If True, perform gain control adjustment")
    parser.add_argument("--log_plots", default = True, type = bool, help = "If True, create log intensity plots")
    parser.add_argument("--join_types", type = join_types_type, default = ["Intersection", "Union"], help = "Comma-separated list of types of spectrum joins to perform - (e.g. Intersection, Union)")
    parser.add_argument("--suffixes", type = suffix_type, default = ['A', 'B'], help = "Comma-separated list of suffixes for the spectra (e.g. A,B treats the first spectrum as spectrum 'A' and the second as spectrum 'B')")
    
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
def parse_arguments(parser):
    args = parser.parse_args()
    args.match_acc = 100.0 / args.R
    args.res_clearance = 200.0 / args.R
    args.subformula_tolerance = 100.0 / args.R
    return args

################################################################################
def run_matching(parent_mz : float, mzm1_1 : str, mzml_2 : str, 
                 index_1 : int, index_2 : int, quasi_x : float,
                 quasi_y : float, R : float, out_dir : str, 
                 parent_formula : str = None, out_prefix : str = None, 
                 starting_index : int = 0, gain_control : bool = False,
                 log_plots : bool = True, join_types : List[str] = ['Intersection', 'Union'],
                 suffixes : List[str] = ['A', 'B'], abs_cutoff : float = None,
                 rel_cutoff : float = None, quasi_cutoff : float = None,
                 min_total_peaks : int = None, min_spectrum_quasi_sum : float = None,
                 exclude_peaks : List[float] = None, predefined_peaks : List[float] = None,
                 du_min : float = -0.5):
    
    logging.info("Reading spectra")
    spectra = get_spectra_by_indices([mzml_1, mzml_2], [index_1 - starting_index, index_2 - starting_index], gain_control)
    logging.info("Validating that spectra are valid and usable")
    validate_spectrum_pair(spectra)
    logging.info("Filtering and converting spectra")
    filtering_kw = {'abs_cutoff' : abs_cutoff, 
      'quasi_x' : quasi_x,
      'quasi_y' : quasi_y,
      'rel_cutoff' : rel_cutoff,
      'quasi_cutoff' : quasi_cutoff,
      'pdpl' : pdpl,
      'exclude_peaks' : exclude_peaks,
      'match_acc' : match_acc,
      'parent_mz' : parent_mz,
      'res_clearance' : res_clearance,
      'sort_intensity' : True
      }
    spectra_filtered = [filter_and_convert_spectrum_complete(spectrum, **filtering_kw) for spectrum in spectra]
    
    # The gray spectra are backups of the original spectra that will be used in the plotting
    gray_filtering_kw = {
    'quasi_x' : quasi_x,
    'quasi_y' : quasi_y,
    'parent_mz' : parent_mz,
    'match_acc' : match_acc
    }
    gray_spectra = [filter_and_convert_spectrum_complete(spectrum, **gray_filtering_kw) for spectrum in spectra]

    logging.info("Merging spectra by m/z matching")
    merged_spectrum = merge_spectra(spectra_filtered[0], spectra_filtered[1], match_acc)
    if parent_formula is not None:
        logging.info("Parent formula specified - Calculating formulas")
        formula_spectrum = add_spectra_formulas(merged_spectrum, parent_formula, subformula_tolerance, du_min, pdpl)
    else:
        formula_spectrum = merged_spectrum.copy()
    logging.info("Calculating metrics")
    metrics = calc_spectra_metrics(formula_spectrum)
    df_stats, df_intersection, df_union, spectra_df = get_results_dfs(spectra, metrics, parent_mz, quasi_x, quasi_y, parent_formula, suffixes)
    write_results_xlsx("/Users/gkreder/Downloads/test.xlsx", df_stats, df_intersection, df_union, spectra_df)
    summary_plots(df_stats, df_intersection, df_union, gray_spectra, suffixes=suffixes, log_plots=log_plots)



def main():
    parser = get_parser()
    args = parse_arguments(parser)
    logging_handlers = [logging.StreamHandler()]
    if args.log_file:
        logging_handlers.append(logging.FileHandler(os.path.join(args.out_dir, args.log_filename)))
    logging.basicConfig(level=args.verbosity, format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=logging_handlers)
    logging.info(f"Args: {args}")
    run_matching(vars(args))



################################################################################

# Take out the hard-coded file paths in the run_matching function