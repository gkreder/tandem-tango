#####################################################
# gk@reder.io
#####################################################
from typing import Dict, List, Literal

import pandas as pd

from tandem_tango.utils.spectrum_operations import get_spectrum_polarity

#####################################################
# Output file generation and reporting
#####################################################

def create_stats_df(metrics: Dict,
                    parent_mz: float,
                    quasi_x: float,
                    quasi_y: float,
                    polarity : str,
                    parent_formula: str = None) -> pd.DataFrame:
    """Create a DataFrame with the computed metrics of the comparison"""

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

    index_vals.append('Polarity')
    union_col_vals.append(polarity)

    if quasi_y == 0.0:
        quasi_str = f"{quasi_x}"
    else:
        quasi_str = f"{quasi_x} x [m/z]^{quasi_y}"
    index_vals.append("Quasicount scaling function")
    union_col_vals.append(quasi_str)


    df_stats = pd.DataFrame({'Union' : union_col_vals, 'Intersection' : metrics['Intersection']['metrics']}, index=index_vals)
    return df_stats

def clean_aligned_df(df : pd.DataFrame, join_type : Literal['Intersection', 'Union'],
                     suffixes : List[str] = ['A', 'B'], formula : bool = True) -> pd.DataFrame: 
    """Cleans the aligned DataFrame and renames columns to a more readable format"""
       
    keep_cols = {'Formula' : 'Formula'} if formula else {}
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
    """Converts a spectrum dictionary to a DataFrame"""
    vals = [spectrum[col] for col in col_map.keys()]
    df = pd.DataFrame(dict(zip(col_map.values(), vals)))
    if suffix:
        df = df.add_suffix(f"_{suffix}")
    return df


def get_results_dfs(spectra : List[Dict],
                       metrics : Dict, parent_mz : float,
                       quasi_x : float, quasi_y : float, 
                       parent_formula : str = None,
                       suffixes : List[str] = ['A', 'B']) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Generates the report DataFrames with the results of the spectrum comparison"""
    polarity = get_spectrum_polarity(spectra[0])
    df_stats = create_stats_df(metrics, parent_mz, quasi_x, quasi_y, polarity, parent_formula)
    df_intersection = clean_aligned_df(pd.DataFrame(metrics['Intersection']['df']), 'Intersection', formula=(parent_formula is not None))
    df_union = clean_aligned_df(pd.DataFrame(metrics['Union']['df']), 'Union', formula=(parent_formula is not None))
    spectra_df = {suffix : spectrum_to_df(spectrum, suffix) for spectrum, suffix in zip(spectra, suffixes)}
    for suffix, sdf in spectra_df.items():
        sdf = sdf.rename(columns={f'I_{suffix}' : f'I_{suffix} (raw intensity)'})
    return df_stats, df_intersection, df_union, spectra_df


def write_results_xlsx(out_excel_file : str, df_stats : pd.DataFrame, df_intersection : pd.DataFrame, 
                       df_union : pd.DataFrame, spectra_df : List[pd.DataFrame],
                       df_params : pd.DataFrame) -> None:
    """Writes the results of the comparison to an Excel file"""
    # writer = pd.ExcelWriter(out_excel_file, engine='xlsxwriter')
    writer = pd.ExcelWriter(out_excel_file, engine="openpyxl")
    df_stats.to_excel(writer, sheet_name='Stats')
    df_intersection.to_excel(writer, sheet_name='Spectra_Intersection', index=False)
    df_union.to_excel(writer, sheet_name='Spectra_Union', index=False)
    for suffix, sdf in spectra_df.items():
        sdf.to_excel(writer, sheet_name=f'Spectrum_{suffix}', index=False)
    df_params.to_excel(writer, sheet_name='Parameters', index=False)

    # Adjust the column widths
    for sheet_name in ['Stats', 'Spectra_Intersection', 'Spectra_Union'] + [f'Spectrum_{suffix}' for suffix, _ in spectra_df.items()]:
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]
        for column_cells in worksheet.columns:
            max_length = 0
            column = column_cells[0].column_letter
            for cell in column_cells:
                try:
                    if cell.value:
                        max_length = max(max_length, len(str(cell.value)))
                except:
                    pass
            adjusted_width = (max_length + 2)  # Adjust the width slightly (for better readability)
            worksheet.column_dimensions[column].width = adjusted_width

    writer.close()