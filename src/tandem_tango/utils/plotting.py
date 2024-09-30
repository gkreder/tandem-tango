#####################################################
# gk@reder.io
#####################################################
import os
from typing import List, Dict, Literal
import logging

import matplotlib.axes
import matplotlib.figure
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-interactive plotting in larger workflows
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd

from adjustText import adjust_text

#####################################################
# Base plotting functions
#####################################################

def single_plot(mzs : List[float], intensities : List[float], 
                formulas : List[str] = None, normalize : bool = True, rotation : float = 90.0, 
               side_text : str = None, fontfamily : str = "DejaVu Sans",
               fig : matplotlib.figure.Figure = None, 
               ax : matplotlib.axes.Axes = None,
               override_color : str = None,
               linewidth : float = None, alpha : float = None):
    """Plots a single MS2 spectrum with optional formula labels"""
    plt.rcParams['font.size'] = 16
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(12, 9))
    if normalize:
        intensities = intensities / intensities.max()
    linewidth_override = linewidth if linewidth else 0.75
    linewidth_def = linewidth if linewidth else 0.25
    if not alpha:
        alpha = 0.3 if not override_color else 1.0
    if override_color is None:
        vlines = ax.vlines(mzs, 0, intensities, color='black', alpha=alpha, linewidth=linewidth_override)
    else:
        vlines = ax.vlines(mzs, 0, intensities, color=override_color, alpha=alpha, linewidth=linewidth_def)
    fig.canvas.draw()

    @FuncFormatter
    def my_formatter(x, pos):
        return f"{abs(x)}"
    ax.get_yaxis().set_major_formatter(my_formatter)
    
    ylim = ax.get_ylim()
    t_adjust = (ylim[1] - ylim[0]) * 0.02
    label_cutoff = 10
    texts = []
    if np.all(formulas) is None:
        formulas = [None for x in mzs]
    package = sorted(zip(mzs, intensities, formulas), key=lambda x: x[1], reverse=True)
    for i_row in range(label_cutoff):
        if i_row >= len(package):
            break
        mz_t, int_t, formula_t = package[i_row]
        if formula_t is not None:
            texts.append(plt.text(mz_t, int_t, formula_t, ha='center', rotation=rotation))        
    if len(texts) > 0:
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black'), ha='center')
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    if side_text is not None:
        plt.text(xlim[1] + ((xlim[1] - xlim[0]) * 0.025), 0.5, side_text, fontfamily=fontfamily)
    plt.xlabel('m/z')
    if normalize:
        plt.ylabel('Relative Intensity')
    else:
        plt.ylabel('Intensity')
    return(fig, ax, vlines)

def mirror_plot(mzs_a : List[float], mzs_b : List[float],
                intensities_a : List[float], intensities_b : List[float], 
                formulas_a : List[str] = None, formulas_b : List[str] = None,
                normalize : bool = True, rotation : float = 90.0, 
               side_text : str = None, fontfamily : str = "DejaVu Sans",
               fig : matplotlib.figure.Figure = None,
               ax : matplotlib.axes.Axes = None, 
               override_color : str = None):
    """Plots two MS2 spectra in mirror-plot configuration with optional formula labels"""
    plt.rcParams['font.size'] = 16
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(12, 9))
    if normalize:
        intensities_a = intensities_a / intensities_a.max()
        intensities_b = intensities_b / intensities_b.max()

    if override_color is None:
        vlines_a = ax.vlines(mzs_a, 0, intensities_a, color='#67a9cf', alpha=0.9, linewidth=0.75)
        vlines_b = ax.vlines(mzs_b, 0, -intensities_b, color='#ef8a62', alpha=0.9, linewidth=0.75)
    else:
        vlines_a = ax.vlines(mzs_a, 0, intensities_a, color=override_color, alpha=0.3, linewidth=0.25)
        vlines_b = ax.vlines(mzs_b, 0, -intensities_b, color=override_color, alpha=0.3, linewidth=0.25)
    ax.axhline(0, 0, 1, color='black', alpha=0.75, linewidth=0.5)

    fig.canvas.draw()

    @FuncFormatter
    def my_formatter(x, pos):
        return f"{abs(x)}"
    ax.get_yaxis().set_major_formatter(my_formatter)
    
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    t_adjust = (ylim[1] - ylim[0]) * 0.02

    label_cutoff = 10
    texts = []

    if np.all(formulas_a) is None or formulas_a is None:
        formulas_a = [None for x in mzs_a]
    if np.all(formulas_b) is None or formulas_b is None:
        formulas_b = [None for x in mzs_b]
    package_a = sorted(zip(mzs_a, intensities_a, formulas_a), key=lambda x: x[1], reverse=True)
    package_b = sorted(zip(mzs_b, intensities_b, formulas_b), key=lambda x: x[1], reverse=True)
    for i_package, package in enumerate([package_a, package_b]):
        for i_row in range(label_cutoff):
            if i_row >= len(package):
                break
            mz_t, int_t, formula_t = package[i_row]
            if i_package == 1:
                int_t = -1 * int_t
            if formula_t is not None:
                texts.append(plt.text(mz_t, int_t, formula_t, ha='center', rotation=rotation))
       
    if len(texts) > 0:
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black'), ha='center')

    
    if side_text is not None:
        # ylimMax = max([abs(x) for x in ylim])
        # ylimRange = ylim[1] - ylim[0]
        # plt.text(xlim[1] + ((xlim[1] - xlim[0]) * 0.025), ylimMax - ( 0.08 * ylimRange ), side_text, fontfamily=fontfamily, verticalalignment='top')
        plt.text(1.025, 0.87, side_text, fontfamily=fontfamily, transform=ax.transAxes, verticalalignment='top')
        
        

    plt.xlabel('m/z')
    if normalize:
        plt.ylabel('Relative Intensity')
    else:
        plt.ylabel('Intensity')
    return(fig, ax)


#####################################################
# Plotting spectrum results
#####################################################

def make_side_text(df_stats : pd.DataFrame, join_type : Literal['Intersection', 'Union'],
                   text_map : Dict[str, str] = {
                        'Parent Formula' : 'Parent formula',
                       'Parent m/z' : 'Precursor m/z',
                       'M' : 'M',
                       "p-val (D^2)" : "pval_D^2",
                        "p-val (G^2)" : "pval_G^2",
                        "s_A (quasi)" : "S_A (quasicounts)",
                        "s_B (quasi)" : "S_B (quasicounts)",
                        "H(p_A)" : "Entropy_A",
                        "H(p_B)" : "Entropy_B",
                        "PP(p_A)" : "Perplexity_A",
                        "PP(p_B)" : "Perplexity_B",
                        "cos(p_A, p_B)" : "Cosine Similarity",
                        "JSD(p_A, p_B)" : "JSD",
                        "BC(p_A, p_B)" : "Bhattacharyya Coefficient",
                   }) -> str:
    """Creates a side text string for the plot based on the passed dataframe and join_type"""
    join_metrics = df_stats[join_type].dropna().to_dict()
    side_text = ""
    for k, v in text_map.items():
        if v not in join_metrics and v in df_stats['Union'].index:
                join_metrics[v] = df_stats['Union'].loc[v]
        if v in join_metrics:
            text_val = join_metrics[v]
            if isinstance(text_val, float) and v not in [text_map['Parent m/z'], 'M']:
                side_text += f"{k} : {text_val:.2e}\n"
            elif v == 'M':
                side_text += f"{k} {join_type} : {int(text_val)}\n"
            elif v == text_map['Parent m/z']:
                side_text += f"{k} : {text_val:.5f}\n"
            else:
                side_text += f"{k} : {join_metrics[v]}\n"
        else:
             logging.warning(f"Key {v} not found in {join_type} dataframe")
    if join_type == "Intersection":
        side_text += f"Jaccard : {df_stats['Union'].loc['Jaccard']:.2e}\n"
    return side_text
    
def plot_result(out_file : str, plot_title : str, df_stats,
                mzs_a : List[float],
                mzs_b : List[float],
                intensities_a : List[float],
                intensities_b : List[float],
                gray_mzs_a : List[float] = None,
                gray_mzs_b : List[float] = None,
                gray_intensities_a : List[float] = None,
                gray_intensities_b : List[float] = None,
                label_x : str = 'm/z',
                label_y : str = 'Intensity',
                join_type = Literal['Intersection', 'Union'],
                normalize : bool = True,
                suffixes : List[str] = ['A', 'B'],
                parent_mz : float = None):
    """Plots a spectrum comparison result with optional gray spectra fragments in the background"""
    
    side_text = make_side_text(df_stats, join_type)
    if gray_mzs_a is not None:
        if gray_mzs_b is None or gray_intensities_a is None or gray_intensities_b is None:
            raise ValueError("If gray spectra data are provided, all must be provided")
        fig, ax = mirror_plot(mzs_a=gray_mzs_a,
                                       mzs_b=gray_mzs_b,
                                       intensities_a=gray_intensities_a,
                                       intensities_b=gray_intensities_b, 
                                       formulas_a=None,
                                       formulas_b=None,
                                       normalize=normalize,
                                    #    side_text = None,
                                        side_text=side_text,
                                       override_color="gray")
    else:
        fig, ax = plt.subplots(figsize = (12,9))
    mirror_plot(mzs_a = mzs_a,
                            mzs_b = mzs_b,
                            intensities_a = intensities_a,
                            intensities_b = intensities_b,
                            formulas_a = None,
                            formulas_b = None,
                            normalize = normalize,
                            rotation = 90,
                            side_text = side_text,
                            fig = fig,
                            ax = ax)
    ax.set_xlim([0, ax.get_xlim()[1]])
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    ylimMax = max([abs(x) for x in ylim])
    xlimMax = max([abs(x) for x in xlim])
    ax.set_ylim([-ylimMax, ylimMax])  # Set symmetric y-limits
    # Set x-limit to be 10% larger than the parent_mz (if provided) else 10% larger than maximum m/z
    if parent_mz:
        ax.set_xlim([0, 1.1 * parent_mz])
    else:
        ax.set_xlim([0, 1.1 * xlimMax])  
    ylimRange = ylim[1] - ylim[0]
    # plt.text(xlim[1] + ( ( xlim[1] - xlim[0] ) *  0.025 ), ylimMax - ( 0.050 * ylimRange ), f"{join_type}:", fontsize = 20)
    plt.text(1.025, 0.95, f"{join_type}:", fontsize = 20, transform=ax.transAxes, verticalalignment='top')
    plt.text(0.01, 0.98, f"Spectrum {suffixes[0]}", 
         fontsize=15, fontfamily='DejaVu Sans', 
         transform=ax.transAxes, verticalalignment='top')
    plt.text(0.01, 0.02, f"Spectrum {suffixes[1]}", 
         fontsize=15, fontfamily='DejaVu Sans', 
         transform=ax.transAxes, verticalalignment='bottom')
    plt.title(f"{plot_title}")
    plt.ylabel(label_y)
    plt.xlabel(label_x)
    plt.savefig(out_file, bbox_inches = 'tight')
    plt.close()

def summary_plots(df_stats, df_intersection, df_union, gray_spectra, 
                  title_suffixes : List[str] = ['A', 'B'],
                  scan_indices : List[int] = [-1, -1],
                  plot_suffixes : List[str] = ['A', 'B'], log_plots = True,
                  file_prefix : str = 'spectrum_comparison',
                  out_dir : str = '', 
                  verbosity : int = logging.INFO,
                  logger : logging.Logger = logging.getLogger(),
                  parent_mz : float = None):
    """Generates summary plots for the passed dataframes and spectra"""
    log_transforms = [False, True] if log_plots else [False]
    for join_type, df_plot in zip(['Intersection', 'Union'], [df_intersection, df_union]):
        for log_transform in log_transforms:
            out_name = f"{os.path.join(out_dir, file_prefix)}_{join_type}{'_log' if log_transform else ''}_plot.svg"
            spectra_subtitles = [f"{title_suffixes[i]} Scan {scan_indices[i]} ({plot_suffixes[i]})" for i in range(2)]
            plot_title = f"{spectra_subtitles[0]} vs.\n {spectra_subtitles[1]} [{join_type}]"
            mzs_a = df_plot['m/z_A']
            mzs_b = df_plot['m/z_B']
            ints_a = np.log10(df_plot['a (quasicounts)']) if log_transform else df_plot['a (quasicounts)']
            ints_b = np.log10(df_plot['b (quasicounts)']) if log_transform else df_plot['b (quasicounts)']
            gray_mzs_a = gray_spectra[0]['m/z array']
            gray_mzs_b = gray_spectra[1]['m/z array']
            gray_intensities_a = np.log10(gray_spectra[0]['quasi array']) if log_transform else gray_spectra[0]['quasi array']
            gray_intensities_b = np.log10(gray_spectra[1]['quasi array']) if log_transform else gray_spectra[1]['quasi array']
            gray_plot_data = {"gray_mzs_a" : gray_mzs_a if log_plots else None, 
                              "gray_mzs_b" : gray_mzs_b if log_plots else None,
                              "gray_intensities_a" : gray_intensities_a if log_plots else None,
                              "gray_intensities_b" : gray_intensities_b if log_plots else None,
                              }

            # Turn off debug logging for the plot generation
            logger.setLevel(max(verbosity, logging.INFO))
            plot_result(
                out_file=out_name, 
                plot_title=plot_title,
                df_stats=df_stats,
                mzs_a=mzs_a,
                mzs_b=mzs_b,
                intensities_a=ints_a,
                intensities_b=ints_b,
                **gray_plot_data,
                label_x = 'm/z',
                label_y = 'Log10 absolute intensity (quasicounts)' if log_transform else 'Relative intensity (quasicounts)',
                join_type = join_type,
                suffixes = plot_suffixes,
                normalize = False if log_transform else True,
                parent_mz = parent_mz)
            logger.setLevel(verbosity)