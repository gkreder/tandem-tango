from typing import Dict, List
import copy
import numpy as np
import logging
from typing import Dict, Tuple, List, Literal
import copy
import warnings

import pandas as pd
import numpy as np
import scipy.stats
from pyteomics import mzml as pytmzml

import formulaUtils

logger = logging.getLogger()

# Constants
NS = {'mzml': 'http://psi.hupo.org/ms/mzml',
      'ms_level': 'MS:1000511',
      "scan_start_time" : 'MS:1000016', # in minutes
      "isolation_window_target_mz" : 'MS:1000827'}
QUASI_CUTOFF_DEFAULT = 5
#####################################################
# Spectrum Filtering, Cleaning, and Merging
#####################################################
def get_spectrum_by_index(mzml_file: str, spec_index: int, gain_control: bool = False):
    """Get a spectrum by index from an mz file"""
    logging.debug(f"Getting spectrum {spec_index} from {mzml_file}")
    reader = pytmzml.MzML(mzml_file)
    spec = reader.get_by_index(spec_index)
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
    """Get the polarity of a spectrum"""
    if 'negative scan' in spectrum.keys():
        return 'Negative'
    elif 'positive scan' in spectrum.keys():
        return 'Positive'
    else:
        raise ValueError("Could not determine polarity of spectrum via Pyteomics")
    
def validate_spectrum_pair(spectra : List[Dict]) -> None:
    """Validate that a pair of spectra are the same polarity and both MS2"""
    if len(set([get_spectrum_polarity(spec) for spec in spectra])) > 1:
        raise ValueError("Spectra have different polarities")
    if set([spec['ms level'] for spec in spectra]) != {2}:
        raise ValueError("Spectra are not both MS2")
    
def validate_spectrum_counts(spectrum : Dict, min_quasi_sum : float, min_total_peaks : int) -> None:
    """Validate that a spectrum has enough peaks and quasi count sum"""
    if len(spectrum['m/z array']) == 0:
        raise ValueError("Spectrum has no peaks")
    if np.sum(spectrum['quasi array']) < min_quasi_sum:
        raise ValueError(f"Spectrum quasi count sum is too low - {np.sum(spectrum['quasi array'])}")
    if len(spectrum['m/z array']) < min_total_peaks:
        raise ValueError(f"Spectrum has too few peaks" - { len(spectrum['m/z array']) })

def get_quasi_counts(spectrum : Dict, quasi_x : float, quasi_y : float) -> Dict:
    """Returns the converted quasi counts for a given spectrum"""
    logging.debug(f"Converting spectrum intensities to quasicounts with x = {quasi_x} and y = {quasi_y}")
    quasi_counts = spectrum['intensity array'] / ( quasi_x *  ( np.power(spectrum['m/z array'], quasi_y) ) )
    return quasi_counts

def sort_spectrum_intensity(spectrum : Dict, sort_fields : List[str] = ['intensity array', 'm/z array', 'quasi array']) -> Dict:
    """Sort a spectrum by intensity"""
    out_spectrum = copy.deepcopy(spectrum)
    sort_order = np.argsort(spectrum['intensity array'])[::-1]
    for key in sort_fields:
        if key in out_spectrum:
            out_spectrum[key] = spectrum[key][sort_order]
    return out_spectrum

def sort_spectrum_mz(spectrum : Dict, sort_fields : List[str] = ['intensity array', 'm/z array', 'quasi array']) -> Dict:
    """Sort a spectrum by m/z"""
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
    """Filter a spectrum along the given filter keys excluding fragments with the given m/zs
    The keep_exact flag allows for keeping the exact m/zs in the exclusion list, throwing away only close hits"""
    logging.debug(f"Filtering spectrum by peak exclusion {exclude_peaks} with match accuracy of {match_acc}")
    out_spectrum = copy.deepcopy(spectrum)
    peak_exclusion_array = np.array(exclude_peaks)
    filter = ~np.any(np.abs(spectrum['m/z array'][:, None] - peak_exclusion_array) <= match_acc, axis=1)
    if keep_exact:
        # Allow exact matches to pass through the filter
        exact_matches = np.isin(spectrum['m/z array'], peak_exclusion_array)
        filter = filter | exact_matches
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
    mzs = [x for x in spectrum['m/z array']]
    for i_mz, mz in enumerate(mzs):
        out_spectrum = filter_spectrum_peak_exclusion(out_spectrum, exclude_peaks=[mz],
                                                      match_acc=res_clearance,
                                                      filter_keys=filter_keys,
                                                      keep_exact=True)
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
                       direction = direction).drop(columns = jk)
    cols_sorted = [f"{k}_{x}" for k in keys.values() for x in suffixes]
    return pd.DataFrame(df[cols_sorted])


#####################################################
# Metrics and Calculations
#####################################################

def H(x):
    h = -1 * np.sum( x * ( np.log(x) ) )
    return(h)

def calc_G2(a_i, b_i, S_A, S_B, M):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            t1 = ( a_i + b_i ) * np.log( ( a_i + b_i ) / ( S_A + S_B ) ).apply(lambda x : 0 if np.isinf(x) else x)
            t2 = a_i * np.log(a_i / S_A).apply(lambda x : 0 if np.isinf(x) else x)
            t3 = b_i * np.log(b_i / S_B).apply(lambda x : 0 if np.isinf(x) else x)
            G2 = -2 * (t1 - t2 - t3).sum()
            pval_G2 = scipy.stats.chi2.sf(G2, df = M - 1)
        return(G2, pval_G2)

def calc_D2(a_i, b_i, S_A, S_B, M):
        num = np.power( ( S_B * a_i )  - ( S_A * b_i ) , 2) 
        denom = a_i + b_i
        D2 = np.sum(num  / denom) / (S_A * S_B)
        
        pval_D2 = scipy.stats.chi2.sf(D2, df = M - 1)
        return(D2, pval_D2)

def calc_SR(a_i, b_i, S_A, S_B):
    num = ( S_B * a_i ) - ( S_A * b_i )
    denom = np.sqrt(S_A * ( S_A + S_B ) * ( a_i + b_i ))
    SR_ai = num / denom

    num = ( S_A * b_i ) - ( S_B * a_i )
    denom = np.sqrt(S_B * ( S_A + S_B ) * ( a_i + b_i ))
    SR_bi = num / denom

    return((SR_ai, SR_bi))

# Cosine Distance
def sqF(x):
    return(np.sqrt( np.power( x, 2 ).sum() ) )

def cosine_distance(p_Ai, p_Bi):
    num = (p_Ai * p_Bi).sum()
    denom = sqF(p_Ai) * sqF(p_Bi)
    csd = num / denom
    return csd

def jensen_shanon_distance(p_Ai, p_Bi, H_pA, H_pB):
    jsd = H( ( 0.5 * ( p_Ai.fillna(0.0) + p_Bi.fillna(0.0) ) ) ) - ( 0.5 * H_pA ) - ( 0.5 *  H_pB  )
    return jsd


def calc_row_metrics(row : pd.Series, quasi_sums : List[float], M : int, join_type : Literal['Union', 'Intersection'],
                     suffixes : List[str] = ['A', 'B']):
        quasi_intensities = [row[f"quasi_{s}"] for s in suffixes] # a_i and b_i
        if np.isnan(quasi_intensities).any() and (join_type == 'Intersection'):
            return None
        quasi_intensities_series = [pd.Series([x]) for x in quasi_intensities]
        D2, _ = calc_D2(*[x.fillna(0.0) for x in quasi_intensities_series],
                         *quasi_sums,
                         M)
        G2, _ = calc_G2(*[x.fillna(0.0) for x in quasi_intensities_series],
                         *quasi_sums,
                         M)
         # Calculate Standardized Residual (SR)
        SRs = calc_SR(*[x.fillna(0.0) for x in quasi_intensities_series],
                               *quasi_sums)
        SRs = [x.values[0] for x in SRs]
        return dict({f'D^2_{join_type}' : D2, f'G^2_{join_type}' : G2}, **{f'SR_{s}_{join_type}' : SR for s, SR in zip(suffixes, SRs)})
      

def calc_join_metrics(merged_spectrum : pd.DataFrame, join_type : Literal['Union', 'Intersection'],
               suffixes : List[str] = ['A', 'B'],
               keys : Dict[str, str] = {'quasi' : '(quasicounts)',
                                        'intensity' : '(raw)',},
               ):
    """Calculates the comparison metrics for two merged spectra given a join type (intersection or union)"""
    # Create a temporary DataFrame on which to calculate comparison statistics
    logging.debug(f"Calculating join metrics for {join_type} join")
    if join_type == "Union":
        dfC = merged_spectrum.copy()
    elif join_type == "Intersection":
        drop_subset = [f"m/z_{s}" for s in suffixes]
        dfC = merged_spectrum.dropna(subset = drop_subset).copy()

    metrics = {}
    metrics['M'] = dfC.shape[0]
    metrics.update({f"S_{s} {keys[k]}" : merged_spectrum[f"{k}_{s}"].sum() for k in keys.keys() for s in suffixes})
    


    quasi_intensities = [dfC[f"quasi_{s}"] for s in suffixes] # a_i and b_i
    quasi_sums = [dfC[f"quasi_{s}"].sum() for s in suffixes] # S_A and S_B


    # Calculate D^2
    D2, pval_D2 = calc_D2(*[x.fillna(value = 0.0) for x in quasi_intensities],
                          *quasi_sums,
                          metrics['M']) # a_i, b_i, S_A, S_B, M
    metrics['D^2'] = D2
    metrics['pval_D^2'] = pval_D2


    # Calculate G^2
    G2, pval_G2 = calc_G2(*[x.fillna(value = 0.0) for x in quasi_intensities],
                          *quasi_sums,
                          metrics['M']) # a_i, b_i, S_A, S_B, M
    metrics['G^2'] = G2
    metrics['pval_G^2'] = pval_G2


    ps = [quasi_intensities[i] / quasi_sums[i] for i in range(len(suffixes))] # p_Ai and p_Bi

    

    entropies = [H(p) for p in ps] # H_A and H_B
    metrics.update({f"Entropy_{s}" : e for s, e in zip(suffixes, entropies)})
    perplexities = [np.exp(h) for h in entropies] # Perplexity_A and Perplexity_B
    metrics.update({f"Perplexity_{s}" : p for s, p in zip(suffixes, perplexities)})


    # Jensen-Shannon divergence
    jsd = jensen_shanon_distance(*ps, *entropies)
    metrics['JSD'] = jsd

    # Cosine distance
    csd = cosine_distance(*ps)
    metrics['Cosine Similarity'] = csd


    dfC[f"D^2_{join_type}"] = None
    dfC[f"G^2_{join_type}"] = None
    dfC[f"SR_A_{join_type}"] = None
    dfC[f"SR_B_{join_type}"] = None


    for i_row, row in dfC.iterrows():
         row_res = calc_row_metrics(row, quasi_sums, metrics['M'], join_type)
         if row_res is not None:
            for k, v in row_res.items():
                dfC.at[i_row, k] = v
    return metrics, dfC

def calc_spectra_metrics(merged_spectrum : pd.DataFrame,
                         suffixes : List[str] = ['A', 'B'],
                         keys : Dict[str, str] = {'quasi' : '(quasicounts)',
                                                  'intensity' : '(raw)',},
                         ) -> Dict:
    """Calculates the comparison metrics for two merged spectra across join types (intersection and union)"""
    metrics = {}
    for join_type in ['Union', 'Intersection']:
        metrics_j, dfC = calc_join_metrics(merged_spectrum=merged_spectrum,
                                           join_type=join_type,
                                           suffixes=suffixes,
                                           keys=keys)
        metrics[join_type] = {}
        metrics[join_type]['metrics'] = metrics_j
        metrics[join_type]['df'] = dfC
    metrics['Jaccard'] = metrics['Intersection']['metrics']['M'] / metrics['Union']['metrics']['M']
    return metrics
    

def calc_join_formulas(mz_1 : float, mz_2 : float,
                      tolerance : float,
                      du_min : float,
                      possible_formulas : List[Tuple[str, float]]) -> Tuple[List[str], List[float], List[str]]:
    """Given two m/z values, calculates the possible formulas and their expected m/z's."""
    if np.isnan(mz_1) and np.isnan(mz_2):
        raise ValueError("Both m/z values are NaN, at least one must be valid.")
    best_formulas, th_masses, errors = formulaUtils.findBestForms(np.nanmean((mz_1, mz_2)),
                                                             possible_formulas,
                                                             toleranceDa = tolerance,
                                                             DuMin=du_min,)
    if best_formulas[0] is None:
        return None, None, None
    else:
        return best_formulas, th_masses, errors

def calc_formulas(merged_spectrum : pd.DataFrame, parent_formula : str,
                  subformula_tolerance : float,
                  du_min : float,
                  mz_col : str = 'm/z',
                  suffixes : List[str] = ['A', 'B'],) -> Tuple[List[List[str]], List[List[float]]]:
    """Calculates the formulas and expected m/z's for a merged spectrum."""
    all_formulas = formulaUtils.generateAllForms(parent_formula)
    # Zip the m/z values across the spectra
    z = zip(*merged_spectrum[[f"{mz_col}_{s}" for s in suffixes]].values.T)
    formula_finds = [calc_join_formulas(mz_a, mz_b, subformula_tolerance, du_min, all_formulas) for mz_a, mz_b in z]
    best_formulas, th_masses, errors = zip(*formula_finds)
    best_formulas_str = [", ".join([x.replace("None", "") for x in f]) if f is not None else None for f in best_formulas]
    th_masses_str = [", ".join([str(x) for x in f]) if f is not None else None for f in th_masses]
    return best_formulas_str, th_masses_str


#####################################################
# Output and Reporting
#####################################################

def create_stats_df(metrics: Dict,
                    parent_mz: float,
                    quasi_x: float,
                    quasi_y: float,
                    parent_formula: str = None) -> pd.DataFrame:

    df_stats = pd.DataFrame({'Union' : metrics['Union']['metrics'], 'Intersection' : metrics['Intersection']['metrics']})
    union_col_vals = list(df_stats['Union'].values)
    index_vals = list(df_stats.index)

    index_vals.append('Jaccard')
    union_col_vals.append(metrics['Jaccard'])

    if parent_formula:
        index_vals.append('Parent formula')
        union_col_vals.append(parent_formula)

    index_vals.append('Precursor m/z')
    union_col_vals.append(parent_mz)

    if quasi_y == 0.0:
        quasi_str = f"{quasi_x}"
    else:
        quasi_str = f"{quasi_x} x [m/z]^{quasi_y}"
    index_vals.append("Quasicount scaling function")
    union_col_vals.append(quasi_str)


    df_stats = pd.DataFrame({'Union' : union_col_vals, 'Intersection' : metrics['Intersection']['metrics']}, index=index_vals)
    return df_stats

def clean_aligned_df(df : pd.DataFrame, join_type : Literal['Intersection', 'Union'],
                     suffixes : List[str] = ['A', 'B'],) -> pd.DataFrame: 
       
    keep_cols = {'Formula' : 'Formula'}
    keep_cols.update({f'm/z_{suffix}' : f'm/z_{suffix}' for suffix in suffixes})
    keep_cols.update({f'intensity_{suffix}' : f'I_{suffix} (raw intensity)' for suffix in suffixes})
    keep_cols.update({f'quasi_{suffix}' : f'{suffix.lower()} (quasicounts)' for suffix in suffixes})
    keep_cols.update({f'{x}^2_{join_type}' : f'{x}^2_{join_type}' for x in ['D', 'G']})
    keep_cols.update({f'SR_{suffix}_{join_type}' : f'SR_{suffix}_{join_type}' for suffix in suffixes})
    df_out = df.rename(columns=keep_cols)
    df_out = df_out.drop(columns=[col for col in df_out.columns if col not in keep_cols.values()])
    # Reorders the output columns according to the order of the keep_cols dict
    df_out = pd.DataFrame(df_out[keep_cols.values()])
    return df_out

def spectrum_to_df(spectrum: Dict,
                   suffix : str = None,
                   col_map : Dict[str, str] = {
                          'm/z array' : 'm/z',
                          'intensity array' : 'I'
                   }) -> pd.DataFrame:
    vals = [spectrum[col] for col in col_map.keys()]
    df = pd.DataFrame(dict(zip(col_map.values(), vals)))
    if suffix:
        df = df.add_suffix(f"_{suffix}")
    return df

    

def write_results_xlsx(spectra : List[Dict],
                       metrics : Dict, parent_mz : float,
                       quasi_x : float, quasi_y : float, 
                       out_excel_file : str,
                       parent_formula : str = None,
                       suffixes : List[str] = ['A', 'B']) -> None:
    df_stats = create_stats_df(metrics, parent_mz, quasi_x, quasi_y, parent_formula)
    polarity = get_spectrum_polarity(spectra[0])
    df_stats.at['Polarity', 'Union'] = polarity
    
    df_intersection = clean_aligned_df(pd.DataFrame(metrics['Intersection']['df']), 'Intersection')
    df_union = clean_aligned_df(pd.DataFrame(metrics['Union']['df']), 'Union')
    spectra_df = [spectrum_to_df(spectrum, suffix) for spectrum, suffix in zip(spectra, suffixes)]
    for i_sdf, sdf in enumerate(spectra_df):
        suffix = suffixes[i_sdf]
        spectra_df[i_sdf] = sdf.rename(columns={f'I_{suffix}' : f'I_{suffix} (raw intensity)'})

    writer = pd.ExcelWriter(out_excel_file, engine='xlsxwriter')
    df_stats.to_excel(writer, sheet_name='Stats')
    df_intersection.to_excel(writer, sheet_name='Spectra_Intersection', index=False)
    df_union.to_excel(writer, sheet_name='Spectra_Union', index=False)
    for i_sdf, sdf in enumerate(spectra_df):
        suffix = suffixes[i_sdf]
        sdf.to_excel(writer, sheet_name=f'Spectrum_{suffix}_Raw', index=False)
    writer.close()



