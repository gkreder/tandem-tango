#####################################################
# gk@reder.io
#####################################################
import os
import logging
from typing import List, Dict
import copy

import pandas as pd
import pyteomics.mzml, pyteomics.mgf
import numpy as np
#####################################################
# Spectrum Filtering, Cleaning, and Merging
#####################################################
class SpectrumValidationError(Exception):
    """Custom exception for spectrum validation errors"""
    pass

def get_spectrum_by_index(file_path : str, spec_index: int, gain_control: bool = False):
    """Get a spectrum by index from an input file"""
    logging.debug(f"Getting spectrum {spec_index} from {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    ext = os.path.splitext(file_path)[1]
    if ext.lower() not in ['.mzml', '.mgf']:
        raise ValueError(f"Unsupported spectrum file extension: {ext}")
    reader = {
        '.mzml' : pyteomics.mzml.MzML,
        '.mgf' : pyteomics.mgf.MGF
    }[ext.lower()](file_path)

    if ext.lower() == '.mzml':
        spec = reader.get_by_index(spec_index)
    elif ext.lower() == '.mgf':
        for i, x in enumerate(reader):
            if i == spec_index:
                spec = x
                break
            raise ValueError(f"Could not find spectrum {spec_index} in {file_path}")
    reader.close()
    if gain_control:
        raise NotImplementedError("Gain control is not yet implemented for Pyteomics implementation")
    return spec

def get_spectra_by_indices(mzml_files : List[str], spec_indices : List[int], gain_control : bool = False):
    """Get a list of spectra by indices from a list of mz files"""
    spectra = []
    for mzml_file, spec_index in zip(mzml_files, spec_indices):
        spectra.append(get_spectrum_by_index(mzml_file, spec_index, gain_control))
    return spectra

def get_spectrum_polarity(spectrum : Dict) -> str:
    """Get the polarity of a spectrum using dictionary keys populated by Pyteomics"""
    # For mzml files read by Pyteomics
    if 'negative scan' in spectrum.keys():
        return 'Negative'
    elif 'positive scan' in spectrum.keys():
        return 'Positive'
    # For mgf files read by Pyteomics
    elif 'params' in spectrum.keys() and 'charge' in spectrum['params'].keys():
        charge_list = spectrum['params']['charge']
        if [x > 0 for x in charge_list]:
            return 'Positive'
        elif [x < 0 for x in charge_list]:
            return 'Negative'
        else:
            raise ValueError("Could not determine polarity of the mgf spectrum via Pyteomics")
    else:
        raise ValueError("Could not determine polarity of spectrum via Pyteomics")

# jrajniak Theoretically, we don't care about the MS level of the spectra: there could be a conceivable use case for comparing two
# MS1 spectra. Also, MS2+ spectra do exist, and are for our purposes equivalent to MS2 spectra.
# In the function below, I would handle this with descriptive warnings such as: 
#
# "Warning - Spectra have different MS levels: {x} for spectrum 1, {y} for spectrum 2"
#
# Also a special warning if trying to compare two MS1 spectra (which is allowable), but often likely to be a consequence of
# off-by-one index errors:
#
# "Warning - Both spectra are MS level 1"

def validate_spectrum_pair(spectra : List[Dict]) -> None:
    """Validate that a pair of spectra are the same polarity and are both MS2"""
    if len(set([get_spectrum_polarity(spec) for spec in spectra])) > 1:
        raise SpectrumValidationError("Spectra have different polarities")
    if np.all(np.array(['ms level' in spec.keys() for spec in spectra])):
        ms_levels = set([spec['ms level'] for spec in spectra])
        if len(ms_levels) > 1:
            logging.warning(f"Spectra have different MS levels: ({[x for x in ms_levels]})")
        if ms_levels == {1}:
            logging.warning(f"Both spectra are MS level 1")
        # if set([spec['ms level'] for spec in spectra]) != {2}:
        #     raise SpectrumValidationError("Spectra are not both MS2")
    else:
        logging.warning("Could not validate MS level of spectra - assuming both are MS2")
    
def validate_spectrum_counts(spectrum : Dict, min_quasi_sum : float, min_total_peaks : int) -> None:
    """Validate that a spectrum has enough peaks and total quasi count sum"""
    if len(spectrum['m/z array']) == 0:
        raise SpectrumValidationError("Spectrum has no peaks")
    if np.sum(spectrum['quasi array']) < min_quasi_sum:
        raise SpectrumValidationError(f"Spectrum quasi count sum is too low - {np.sum(spectrum['quasi array'])}")
    if len(spectrum['m/z array']) < min_total_peaks:
        raise SpectrumValidationError(f"Spectrum has too few peaks - {len(spectrum['m/z array'])}")

def get_quasi_counts(spectrum : Dict, quasi_x : float, quasi_y : float) -> np.array:
    """Returns the converted quasi counts for a given spectrum"""
    logging.debug(f"Converting spectrum intensities to quasicounts with x = {quasi_x} and y = {quasi_y}")
    quasi_counts = spectrum['intensity array'] / ( quasi_x *  ( np.power(spectrum['m/z array'], quasi_y) ) )
    return quasi_counts

def sort_spectrum_intensity(spectrum : Dict, sort_fields : List[str] = ['intensity array', 'm/z array', 'quasi array']) -> Dict:
    """Sort a spectrum by intensity (high to low)"""
    out_spectrum = copy.deepcopy(spectrum)
    sort_order = np.argsort(spectrum['intensity array'])[::-1]
    for key in sort_fields:
        if key in out_spectrum:
            out_spectrum[key] = spectrum[key][sort_order]
    return out_spectrum

def sort_spectrum_mz(spectrum : Dict, sort_fields : List[str] = ['intensity array', 'm/z array', 'quasi array']) -> Dict:
    """Sort a spectrum by m/z (low to high)"""
    out_spectrum = copy.deepcopy(spectrum)
    sort_order = np.argsort(spectrum['m/z array'])
    for key in sort_fields:
        if key in out_spectrum:
            out_spectrum[key] = spectrum[key][sort_order]
    return out_spectrum

def filter_spectrum_absolute(spectrum : Dict, abs_cutoff : float, filter_keys : List[str] = ['intensity array', 'm/z array', 'quasi array']) -> Dict:
    """Filter a spectrum by an absolute intensity cutoff along the given filter keys"""
    logging.debug(f"Filtering spectrum by absolute intensity cutoff of {abs_cutoff}")
    out_spectrum = copy.deepcopy(spectrum)
    filter = spectrum['intensity array'] >= abs_cutoff
    for key in filter_keys:
        if key in out_spectrum:
            out_spectrum[key] = spectrum[key][filter]
    return out_spectrum

def filter_spectrum_pdpl(spectrum : Dict, pdpl : List[float], match_acc : float, filter_keys : List[str] = ['intensity array', 'm/z array', 'quasi array']) -> Dict:
    """Filter a spectrum according to a pre-defined peak list (PDPL) along the given filter keys"""
    logging.debug(f"Filtering spectrum by PDPL {pdpl} with match accuracy of {match_acc}")
    out_spectrum = copy.deepcopy(spectrum)
    pdpl_array = np.array(pdpl)
    filter = np.any(np.abs(spectrum['m/z array'][:, None] - pdpl_array) <= match_acc, axis=1)
    for key in filter_keys:
        if key in out_spectrum:
            out_spectrum[key] = spectrum[key][filter]
    return out_spectrum

def get_mz_match_int(spectrum : Dict, match_mz : float, match_acc : float, intensity_key : str = "intensity array") -> float:
    """Find the intensity of the closest m/z in the spectrum within match_acc if any, None if no match. If multiple matches, returns the highest matching intensity"""
    logging.debug(f"Getting m/z match from spectrum with match accuracy of {match_acc}")
    hit_idxs = np.where(np.abs(spectrum['m/z array'] - match_mz) <= match_acc)[0]
    hits = sorted(zip(spectrum['m/z array'][hit_idxs], spectrum[intensity_key][hit_idxs]), key = lambda x : x[1], reverse = True)
    if len(hits) > 0:
        return hits[0][1]
    return None

def get_mz_match_list(spectrum : Dict, match_mzs : List[float], match_acc : float, intensity_key : str = "intensity array") -> List[float]:
    """For each mz in match_mzs finds the closest m/z in the spectrum within match_acc if any. Returns a list of the matched intensity values (None is no match)"""
    logging.debug(f"Getting m/z matches from spectrum with match accuracy of {match_acc}")
    match_mzs_array = np.array(match_mzs)
    hits = [get_mz_match_int(spectrum, x, match_acc, intensity_key) for x in match_mzs_array]
    return hits

def filter_spectrum_quasi(spectrum : Dict, quasi_cutoff : float, filter_keys : List[str] = ['intensity array', 'm/z array', 'quasi array']) -> Dict:
    """Filter a spectrum by quasi counts along the given filter keys"""
    logging.debug(f"Filtering spectrum by quasi count cutoff of {quasi_cutoff}")
    out_spectrum = copy.deepcopy(spectrum)
    filter = spectrum['quasi array'] >= quasi_cutoff
    for key in filter_keys:
        if key in out_spectrum:
            out_spectrum[key] = spectrum[key][filter]
    return out_spectrum

def filter_spectrum_parent_mz(spectrum : Dict, parent_mz : float, match_acc : float, filter_keys : List[str] = ['intensity array', 'm/z array', 'quasi array']) -> Dict:
    """Filter a spectrum along the given filter keys leaving only fragments with m/z <= parent m/z"""
    logging.debug(f"Filtering spectrum by parent m/z {parent_mz} with match accuracy of {match_acc}")
    out_spectrum = copy.deepcopy(spectrum)
    filter = spectrum['m/z array'] <= (parent_mz+match_acc)
    for key in filter_keys:
        if key in out_spectrum:
            out_spectrum[key] = spectrum[key][filter]
    return out_spectrum

def filter_spectrum_peak_exclusion(spectrum : Dict, exclude_peaks : List[float], match_acc : float, keep_exact = False, filter_keys : List[str] = ['intensity array', 'm/z array', 'quasi array']) -> Dict:
    """Filter a spectrum along the given filter keys excluding fragments with the given m/z's within match_acc.
    The keep_exact flag allows for keeping the exact m/zs in the exclusion list, throwing away only matches within match_acc"""
    logging.debug(f"Filtering spectrum by peak exclusion {exclude_peaks} (keep_exact={keep_exact}) with match accuracy of {match_acc}")
    out_spectrum = copy.deepcopy(spectrum)
    peak_exclusion_array = np.array(exclude_peaks)
    filter = ~np.any(np.abs(spectrum['m/z array'][:, None] - peak_exclusion_array) <= match_acc, axis=1)
    if keep_exact:
        # Allow exact matches to pass through the filter
        exact_matches = np.isin(spectrum['m/z array'], peak_exclusion_array)
        filter = filter | exact_matches
    filtered_mzs = out_spectrum['m/z array'][~filter]
    logging.debug(f"\tFiltering {filtered_mzs}")
    for key in filter_keys:
        if key in out_spectrum:
            out_spectrum[key] = spectrum[key][filter]
    return out_spectrum

def filter_spectrum_relative(spectrum : Dict, rel_cutoff : float, filter_keys : List[str] = ['intensity array', 'm/z array', 'quasi array']) -> Dict:
    """Filter a spectrum by a relative intensity cutoff along the given filter keys"""
    logging.debug(f"Filtering spectrum by relative intensity cutoff of {rel_cutoff}")
    out_spectrum = copy.deepcopy(spectrum)
    filter = spectrum['intensity array'] >= (rel_cutoff * np.max(spectrum['intensity array']))
    for key in filter_keys:
        if key in out_spectrum:
            out_spectrum[key] = spectrum[key][filter]
    return out_spectrum

def filter_spectrum_res_clearance(spectrum : Dict, res_clearance : float, sort_intensity : bool = True, filter_keys : List[str] = ['intensity array', 'm/z array', 'quasi array']) -> Dict:
    """Filter a spectrum by resolution clearance along the given filter keys. Loops through the fragment arrays
    and for each m/z, filters out any other fragments within res_clearance"""
    logging.debug(f"Filtering spectrum by resolution clearance of {res_clearance}")
    # Sort by intensity if requested
    if sort_intensity:
        out_spectrum = sort_spectrum_intensity(spectrum, filter_keys)
    else:
        out_spectrum = copy.deepcopy(spectrum)
    mzs_to_filter = [x for x in spectrum['m/z array']]
    while len(mzs_to_filter) > 0:
        mz = mzs_to_filter.pop(0)
        out_spectrum = filter_spectrum_peak_exclusion(out_spectrum, exclude_peaks=[mz],
                                                      match_acc=res_clearance,
                                                      filter_keys=filter_keys,
                                                      keep_exact=True)
        mzs_to_filter = [x for x in mzs_to_filter if x in out_spectrum['m/z array']]

    return out_spectrum

def filter_and_convert_spectrum_complete(spectrum : Dict, 
                             abs_cutoff : float = None,
                             quasi_x : float = None,
                             quasi_y : float = None,
                             rel_cutoff : float = None,
                             quasi_cutoff : float = None,
                             pdpl : List[float] = None,
                             exclude_peaks : List[float] = None,
                             match_acc : float = None,
                             parent_mz : float = None,
                             res_clearance : float = None,
                             sort_intensity : bool = True) -> Dict:
    """A wrapper function that filters a spectrum (and converts to quasicounts) by all filters in succession according to a set order.
    Absolute intensity filtering (if provided) is performed before quasicount conversion"""
    logging.debug(f"Filtering spectrum with filters: abs_cutoff={abs_cutoff}, quasi_x={quasi_x}, quasi_y={quasi_y}, rel_cutoff={rel_cutoff}, quasi_cutoff={quasi_cutoff}, pdpl={pdpl}, exclude_peaks={exclude_peaks}, match_acc={match_acc}, parent_mz={parent_mz}, res_clearance={res_clearance}")
    out_spectrum = copy.deepcopy(spectrum)
    # Apply filters
    if abs_cutoff is not None:
        out_spectrum = filter_spectrum_absolute(out_spectrum, abs_cutoff)
    if quasi_x is not None or quasi_y is not None:
        if not (quasi_x is not None and quasi_y is not None):
            raise ValueError("Both or none of quasi_x and quasi_y must be provided")
        out_spectrum['quasi array'] = get_quasi_counts(out_spectrum, quasi_x, quasi_y)
    if pdpl is not None:
        if match_acc is None:
            raise ValueError("PDPL provided but no match accuracy provided")
        out_spectrum = filter_spectrum_pdpl(out_spectrum, pdpl, match_acc)
    if quasi_cutoff is not None:
        out_spectrum = filter_spectrum_quasi(out_spectrum, quasi_cutoff)    
    if parent_mz is not None:
        if match_acc is None:
            raise ValueError("Parent m/z provided but no match accuracy provided")
        out_spectrum = filter_spectrum_parent_mz(out_spectrum, parent_mz, match_acc)
    if exclude_peaks is not None:
        if match_acc is None:
            raise ValueError("Excluded peaks provided but no match accuracy")
        out_spectrum = filter_spectrum_peak_exclusion(out_spectrum, exclude_peaks, match_acc)
    if rel_cutoff is not None:
        out_spectrum = filter_spectrum_relative(out_spectrum, rel_cutoff)
    if sort_intensity:
        out_spectrum = sort_spectrum_intensity(out_spectrum)
    if res_clearance is not None:
        out_spectrum = filter_spectrum_res_clearance(out_spectrum, res_clearance, sort_intensity)
            
    return out_spectrum

def merge_spectra(s1 : Dict, s2 : Dict, tolerance : float,
                  suffixes : List[str] = ["A", "B"],
                  direction : str = 'nearest',
                  keys : Dict[str, str] = {'m/z array' : 'm/z',
                                           'intensity array' : 'intensity',
                                           'quasi array' : 'quasi'},
                join_key : str = 'm/z array') -> pd.DataFrame:
    """Takes two spectra and merges them into a single DataFrame based on the join_key and match tolerance.
    Spectra are sorted by join_key before merging."""

    logging.debug(f"Merging spectra with tolerance {tolerance} using direction {direction} and join key {join_key}")
    # Add extra join key column for merging
    jk = f"{join_key}_join"
    dfs = [pd.DataFrame(dict({v : s[k] for k, v in keys.items()}, **{jk : s[join_key]})) for s in [s1, s2]]
    dfs = [df.sort_values(jk) for df in dfs]
    df = pd.merge_asof(dfs[0], dfs[1],
                       tolerance = tolerance,
                       on = jk,
                       suffixes = [f'_{x}' for x in suffixes],
                       direction = direction)
    for i_suf, suf in enumerate(suffixes):
        dft = dfs[i_suf]
        dfRem = dft[~dft[jk].isin(df[f'{keys[join_key]}_{suf}'])].rename(columns = {x : f"{x}_{suf}" for x in keys.values()})
        df = pd.concat([df, dfRem])
    df = df.drop(columns = jk).reset_index(drop = True)
    cols_sorted = [f"{k}_{x}" for k in keys.values() for x in suffixes]
    merged_spectra = pd.DataFrame(df[cols_sorted])
    return merged_spectra
