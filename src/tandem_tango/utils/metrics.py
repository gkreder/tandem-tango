#####################################################
# gk@reder.io
#####################################################
import warnings
from typing import List, Dict, Literal
import logging

import numpy as np
import scipy.stats
import pandas as pd

from tandem_tango.utils import formula_utils

#####################################################
# Metrics and Calculations for spectrum comparison
#####################################################

def H(x : np.array):
    """Auxiliary function for calculating spectrum entropy"""
    # jrajniak
    if np.sum(x) == 0:
        return 0
    # la =  x * np.log(x) 
    # h = -1 * np.sum( x * ( np.log(x) ) )
    # If there's a zero value in x, set the log to 0 so x * log(x) = 0 instead of nan
    log_array = np.nan_to_num(np.log(x), neginf=0)
    h = -1 * np.sum(x * log_array)
    return(h)

def calc_G2(a_i : np.array, b_i : np.array, S_A : float, S_B : float, M : int):
        """Calculates the G statistic for two spectra given their intensities (a_i, b_i) and quasi-count sums (S_A, S_B)"""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # jrajniak
            t1 = ( a_i + b_i ) * np.log( ( a_i + b_i ) / ( S_A + S_B ) ).apply(lambda x : 0 if np.isinf(x) else x)
            t2 = a_i * np.log(a_i / S_A).apply(lambda x : 0 if np.isinf(x) else x)
            t3 = b_i * np.log(b_i / S_B).apply(lambda x : 0 if np.isinf(x) else x)
            G2 = -2 * (t1 - t2 - t3).sum()
            pval_G2 = scipy.stats.chi2.sf(G2, df = M - 1)
        return(G2, pval_G2)

def calc_D2(a_i : np.array, b_i : np.array, S_A : float, S_B : float, M : int):
        """Calculates the D^2 (chi-square) statistic for two spectra given their intensities (a_i, b_i) and quasi-count sums (S_A, S_B)"""
        num = np.power( ( S_B * a_i )  - ( S_A * b_i ) , 2) 
        denom = a_i + b_i
        # jrajniak
        if denom.sum() == 0 or S_A == 0 or S_B == 0:
            return 0.0, 1.0 # chi-sq = 0, p-val = 1
        D2 = np.sum(num  / denom) / (S_A * S_B)
        pval_D2 = scipy.stats.chi2.sf(D2, df = M - 1)
        return(D2, pval_D2)

def calc_SR(a_i : np.array, b_i : np.array, S_A : float, S_B : float):
    """Calculates the Standardized Residual (SR) for two spectra given their intensities (a_i, b_i) and quasi-count sums (S_A, S_B)"""
    num = ( S_B * a_i ) - ( S_A * b_i )
    denom = np.sqrt(S_A * ( S_A + S_B ) * ( a_i + b_i ))
    SR_ai = num / denom

    num = ( S_A * b_i ) - ( S_B * a_i )
    denom = np.sqrt(S_B * ( S_A + S_B ) * ( a_i + b_i ))
    SR_bi = num / denom

    return((SR_ai, SR_bi))

# Cosine Distance
def sqf(x : np.array):
    """Auxiliary function for calculating cosine distance"""
    return(np.sqrt( np.power( x, 2 ).sum() ) )

def cosine_distance(p_Ai : np.array, p_Bi : np.array):
    """Calculates the cosine distance between two spectra given their normalized intensities (p_Ai, p_Bi)"""
    num = (p_Ai * p_Bi).sum()
    denom = sqf(p_Ai) * sqf(p_Bi)
    # jrajniak
    # A nonsensical 2.0 value is returned if the denominator is 0
    if denom == 0:
        return 2.0
    csd = num / denom
    return csd

def jensen_shanon_distance(p_Ai, p_Bi, H_pA, H_pB):
    """Calculates the Jensen-Shannon divergence between two spectra given their normalized intensities (p_Ai, p_Bi) and entropies (H_pA, H_pB)"""
    jsd = H( ( 0.5 * ( p_Ai.fillna(0.0) + p_Bi.fillna(0.0) ) ) ) - ( 0.5 * H_pA ) - ( 0.5 *  H_pB  )
    return jsd

def bhattacharyya_coef(p_Ai : np.array, p_Bi : np.array):
    """Calculates the Bhattacharyya coefficient between two spectra given their normalized intensities (p_Ai, p_Bi)"""
    bcc = np.sum(np.sqrt(p_Ai.fillna(0.0) * p_Bi.fillna(0.0)))
    return bcc


def calc_row_metrics(row : pd.Series, quasi_sums : List[float], M : int, join_type : Literal['Union', 'Intersection'],
                     suffixes : List[str] = ['A', 'B']):
        """Calculates the comparison metrics for a single row of a merged spectrum given a join type (intersection or union)"""
        quasi_intensities = [row[f"quasi_{s}"] for s in suffixes] # a_i and b_i
        if np.isnan(quasi_intensities).any() and (join_type == 'Intersection'):
            return None
        quasi_intensities_series = [pd.Series([x]) for x in quasi_intensities]
        # Calculation of D^2, G^2, and Standardized Residuals (SRs)
        D2, _ = calc_D2(*[x.fillna(0.0) for x in quasi_intensities_series],
                         *quasi_sums,
                         M)
        G2, _ = calc_G2(*[x.fillna(0.0) for x in quasi_intensities_series],
                         *quasi_sums,
                         M)
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
    logging.debug(f"Calculating join metrics for {join_type} join")
    # Create a temporary DataFrame on which to calculate comparison statistics
    if join_type == "Union":
        dfC = merged_spectrum.copy()
    elif join_type == "Intersection":
        drop_subset = [f"m/z_{s}" for s in suffixes]
        dfC = merged_spectrum.dropna(subset = drop_subset).copy()

    # M is the number of fragments in the merged spectrum
    metrics = {}
    metrics['M'] = dfC.shape[0]
    # Calculate the sum of the intensities for each spectrum
    metrics.update({f"S_{s} {keys[k]}" : dfC[f"{k}_{s}"].sum() for k in keys.keys() for s in suffixes})

    # Calculate the sum of the quasi-counts for each spectrum
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

    # p_Ai and p_Bi
    ps = [quasi_intensities[i] / quasi_sums[i] for i in range(len(suffixes))] 

    
    # Entropy and Perplexity
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

    # Bhattacharyya coefficient
    bcc = bhattacharyya_coef(*ps)
    metrics['Bhattacharyya Coefficient'] = bcc

    # Create columns for the individual row metrics that will be filled in the next step
    dfC[f"D^2_{join_type}"] = None
    dfC[f"G^2_{join_type}"] = None
    dfC[f"SR_A_{join_type}"] = None
    dfC[f"SR_B_{join_type}"] = None

    # Calculate the metrics for each row in the merged spectrum
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
                      possible_formulas : List[tuple[str, float]]) -> tuple[List[str], List[float], List[str]]:
    """Given two m/z values, calculates the possible formulas and their expected m/z's."""
    if np.isnan(mz_1) and np.isnan(mz_2):
        raise ValueError("Both m/z values are NaN, at least one must be valid.")
    best_formulas, th_masses, errors = formula_utils.find_best_forms(np.nanmean((mz_1, mz_2)),
                                                             possible_formulas,
                                                             tolerance_da=tolerance,
                                                             du_min=du_min,)
    if best_formulas[0] is None:
        return None, None, None
    else:
        return best_formulas, th_masses, errors

def calc_formulas(merged_spectrum : pd.DataFrame, parent_formula : str,
                  subformula_tolerance : float,
                  du_min : float,
                  mz_col : str = 'm/z',
                  suffixes : List[str] = ['A', 'B'],) -> tuple[List[List[str]], List[List[float]]]:
    """Calculates the formulas and expected m/z's for a merged spectrum."""
    all_formulas = formula_utils.generate_all_forms(parent_formula)
    # Zip the m/z values across the spectra
    z = zip(*merged_spectrum[[f"{mz_col}_{s}" for s in suffixes]].values.T)
    formula_finds = [calc_join_formulas(mz_a, mz_b, subformula_tolerance, du_min, all_formulas) for mz_a, mz_b in z]
    best_formulas, th_masses, errors = zip(*formula_finds)
    best_formulas_str = [", ".join([x.replace("None", "") for x in f]) if f is not None else None for f in best_formulas]
    th_masses_str = [", ".join([str(x) for x in f]) if f is not None else None for f in th_masses]
    return best_formulas_str, th_masses_str

def add_spectra_formulas(merged_spectrum : pd.DataFrame, parent_formula : str, 
                         subformula_tolerance : float, du_min : float,
                         pdpl : List[float] = None) -> pd.DataFrame:
    """Adds the formulas and expected m/z's to a merged spectrum DataFrame"""
    best_formulas_str, th_masses_str = calc_formulas(merged_spectrum, parent_formula, subformula_tolerance, du_min,)
    formula_spectrum = merged_spectrum.copy()
    formula_spectrum['Formula'] = best_formulas_str
    formula_spectrum['m/z_calculated'] = th_masses_str
    # If there's no pre-defined peak list (pdpl) and a parent formula was specified drop rows with no formula
    if pdpl is None:
        logging.debug("PDPL not specified - Filtering non-formula fragments")
        formula_spectrum = formula_spectrum.dropna(subset = ['Formula'])
    return formula_spectrum