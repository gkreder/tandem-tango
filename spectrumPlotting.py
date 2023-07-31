import argparse
import os
import sys
from pyteomics import mzml, auxiliary
import numpy as np
import pandas as pd
import warnings
import molmass
import adjustText
import matplotlib.pyplot as plt
import seaborn as sns
import spectrumMatching
import formulaUtils
import plotUtils
from matplotlib.ticker import FormatStrFormatter

def create_parser():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--mzML', help='mzML file')
    group.add_argument('--text', help='Text file with two columns')
    parser.add_argument('--polarity', choices=['Positive', 'Negative', 'Pos', 'Neg'], required=True, help='Spectrum polarity')
    parser.add_argument('--precursor_mz', required=True, help='Precursor m/z', type = float)
    parser.add_argument('--index', required='--mzML' in sys.argv, help='MS/MS scan index', type = int)
    
    parser.add_argument('--outdir', help='Output directory path')
    parser.add_argument('--filename', help='Output filename')
    parser.add_argument('--start', default=0, help='Spectrum starting index')
    parser.add_argument('--cutoff', default=0, help='Raw count noise intensity cutoff', type = float)
    parser.add_argument('--peaks', default=5, help='Number of labeled peaks', type = int)
    

    parser.add_argument('--formula', help='Parent peak formula')
    parser.add_argument('--tolerance', default=0.01, help='Formula assignment m/z tolerance', type = float)
    parser.add_argument("--DUMin", default = -0.5, help = "Physical Formula Checking, set to None for no checking")

    parser.add_argument('--quasiScale', action='store_true', help='Quasicount scaling')
    parser.add_argument('--quasiX', required="--quasiScale" in sys.argv, help='Quasicount function scale', type = float)
    parser.add_argument('--quasiY', required="--quasiScale" in sys.argv, help='Quasicount function exponent', type = float)
    parser.add_argument('--quasiCutoff', default=0, help='Quasicount intensity cutoff', type = float)
    return parser

def get_polarity(pyteomics_spec):
    if 'positive scan' in pyteomics_spec.keys():
        return("Positive")
    elif 'negative scan' in pyteomics_spec.keys():
        return('Negative')
    else:
        sys.exit(f"Error - couldnt determine the polarity of the pyteomics spectrum {pyteomics_spec}")

def get_spectrum(**kwargs):
    args = argparse.Namespace(**kwargs)
    spectrum = None
    if args.in_type == ".mzML":
        search_index = int(args.index) - int(args.start)
        with mzml.read(args.in_file) as reader:
            for i, spec in enumerate(reader):
                if int(spec['index']) == search_index:
                    spectrum = spec
                    break
        if not spectrum:
            sys.exit(f'Couldnt find spectrum with index {search_index} in file {args.in_file}')
        mzs = spectrum['m/z array']
        intensities = spectrum['intensity array']
        polarity = get_polarity(spectrum)
        if polarity != args.polarity:
            warnings.warn(f"Warning - the found spectrum polarity is {polarity} but user provided {args.polarity}", UserWarning)
    elif args.in_type == ".txt":
        with open(args.in_file, 'r') as f:
            lines = f.readlines()[1 : ] # skip the header
        mzs, intensities = zip(*[x.strip().split('\t') for x in lines])
        mzs = [float(x) for x in mzs]
        intensities = [float(x) for x in intensities]
    else:
        sys.exit(f'Error - unrecognized input file type {in_type}')
    out_spectrum = pd.DataFrame({'mz' : mzs, 'intensity' : intensities})
    return(out_spectrum)

def plot_fun(plot_spectrum, bestFormulas, log_plot = False, **kwargs):
    args = argparse.Namespace(**kwargs)
    plot_spectrum['normalized_intensity'] = plot_spectrum['intensity'] / plot_spectrum['intensity'].max()
    plot_spectrum['log_intensity'] = np.log10(plot_spectrum['intensity'])
    if log_plot:
        plot_spectrum['intensity'] = plot_spectrum['log_intensity']
    else:
        plot_spectrum['intensity'] = plot_spectrum['normalized_intensity']
    fig, ax = plt.subplots(figsize = (12,9))
    p_color = 'black' if not args.formula else "gray"
    p_lwidth = 0.75 if not args.formula else 0.25
    p_alpha = 1.0 if not args.formula else 0.3
    plotUtils.singlePlot(plot_spectrum['mz'], plot_spectrum['intensity'], bestFormulas, normalize = True, 
                         fig = fig, ax = ax, overrideColor=p_color, linewidth=p_lwidth, alpha = p_alpha)
    if args.formula:
        st = plot_spectrum.dropna(subset = ['formula']).reset_index(drop = True)
        ax.vlines(st['mz'], 0, st['intensity'], color = 'blue', alpha = 1.0, linewidth = 0.75)
    labels = []
    ax.set_xlim([0, ax.get_xlim()[1]])
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    ylimMax = max([abs(x) for x in ylim])
    ylimRange = ylim[1] - ylim[0]
    xlimRange = xlim[1] - xlim[0]
    tAdjust = ylimRange * 0.02
    ax.set_ylim(0, ylim[1] + tAdjust)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    texts = []
    for i_row, row in plot_spectrum.iterrows():
        if i_row >= args.peaks:
            break
        x = row['mz']
        y = row['intensity']
        peak_text = f"{row['mz']:.5f}"
        if args.formula and row['all_formulas']:
            peak_text += f"\n ({row['all_formulas']})"
        text = ax.annotate(f"{peak_text}", (x,y), 
                           xytext=(x, y + (ylimRange * 0.025)),
                           textcoords='data', 
                           arrowprops=dict(arrowstyle = "-", facecolor = 'gray', linestyle='dashed'), fontsize = 8)
        texts.append(text)
    adjustText.adjust_text(texts, ax=ax)
    ylabel_suf = "raw counts" if not args.quasiScale else "quasi counts"
    ylabel_pref = "Log10 absolute" if log_plot else "Relative"
    ylabel = f"{ylabel_pref} intensity ({ylabel_suf})"
    plt.ylabel(ylabel)
    return(fig, ax)


def process(**kwargs):
    args = argparse.Namespace(**kwargs)
    args.polarity = args.polarity if args.polarity in ['Positive', 'Negative'] else {'Pos' : 'Positive', 'Neg' : 'Negative'}[args.polarity]
    if args.mzML:
        args.in_file = args.mzML
    elif args.text:
        args.in_file = args.text
    if not args.outdir:
        args.outdir = str(os.path.dirname(args.in_file))
    os.makedirs(args.outdir, exist_ok = True)
    in_info = os.path.splitext(args.in_file)
    in_pref = os.path.basename(in_info[0])
    args.in_type = in_info[1]
    if not args.filename:
        args.out_pref = f"{in_pref}"
        st = f"Scan_{args.index}" if args.index else None
        if st:
            args.out_pref += f'_{st}'
    else:
        args.out_pref = args.filename
    args.spectrum = get_spectrum(**vars(args))
    args.raw_spectrum = args.spectrum.copy()
    spectrum = args.spectrum
    args.absCutoff = args.cutoff
    args.parentMZ = args.precursor_mz
    args.matchAcc = args.tolerance
    spectrum = spectrumMatching.filter_data(spectrum, "absolute", **vars(args))
    if args.quasiScale:
        spectrum = spectrumMatching.quasi_convert(spectrum, **vars(args))
        spectrum = spectrumMatching.filter_data(spectrum, "quasi", **vars(args))
    spectrum = spectrumMatching.filter_data(spectrum, 'parent_mz', **vars(args))
    bestFormulas = None
    bestFormulaMasses = None
    if args.formula:
        args.parentFormula = args.formula
        form = molmass.Formula(args.parentFormula).formula
        dum = args.DUMin
        args.DUMin = dum if type(dum) == type(1.0) else None
        charge = {'Positive' : 1, "Negative" : -1}[args.polarity]
        allForms = formulaUtils.generateAllForms(form)
        formulas = [None for x in range(len(spectrum))]
        formulaMasses = [None for x in range(len(spectrum))]
        bestFormulas = [None for x in range(len(spectrum))]
        bestFormulaMasses = [None for x in range(len(spectrum))]
        for i_mz, mz in enumerate(spectrum['mz'].values):
            bestForms, thMasses, errors = formulaUtils.findBestForms(mz, allForms, toleranceDa = args.tolerance, DuMin=args.DUMin, charge = charge)
            bestForm = bestForms[0]
            if bestForm == None:
                continue
            else:
                bestFormulas[i_mz] = bestForm
                bestFormulaMasses[i_mz] = thMasses[0]
            formulas[i_mz] = ", ".join([str(x).replace("None", "") for x in bestForms])
            formulaMasses[i_mz] = ", ".join([str(x) for x in thMasses])
        spectrum['formula'] = bestFormulas
        spectrum['formula_mass'] = bestFormulaMasses
        spectrum['all_formulas'] = formulas

    spectrum = spectrum.sort_values(by = 'intensity', ascending = False).reset_index(drop = True)
    plot_spectrum = spectrum.copy()
    if args.quasiScale:
        plot_spectrum['intensity'] = plot_spectrum['quasi']
    fig, ax = plot_fun(plot_spectrum, bestFormulas, **vars(args))
    plt.savefig(os.path.join(args.outdir, f"{args.out_pref}_plot.svg"), bbox_inches = 'tight') 
    plt.close()
    fig, ax = plot_fun(plot_spectrum, bestFormulas, log_plot=True, **vars(args))
    plt.savefig(os.path.join(args.outdir, f"{args.out_pref}_log_plot.svg"), bbox_inches = 'tight') 
    plt.close()


    stats_spectrum = spectrum.copy()
    if args.formula:
        stats_spectrum = stats_spectrum.dropna(subset = "formula")
    df1 = pd.DataFrame(columns=["Param", "Value"])
    df1.loc[len(df1)] = ['M', len(stats_spectrum)]
    s = 'raw counts' if not args.quasiScale else 'quasi counts'
    k = "intensity" if not args.quasiScale else "quasi"
    df1.loc[len(df1)] = [f"Total intensity ({s})", stats_spectrum[k].sum()]
    entropy = spectrumMatching.H(stats_spectrum[k])
    perplexity = np.exp(entropy)
    df1.loc[len(df1)] = ["Entropy", entropy]
    df1.loc[len(df1)] = ["Perplexity", perplexity]
    df1.loc[len(df1)] = ["Precursor m/z", args.precursor_mz]
    if args.formula:
        df1.loc[len(df1)] = ["Precursor formula", args.formula]
        df1.loc[len(df1)] = ["Formula assignment m/z tolerance", args.tolerance]
    df1.loc[len(df1)] = ["Polarity", args.polarity]
    if args.quasiScale:
        df1.loc[len(df1)] = ["Quasicount function scale", args.quasiX]
        df1.loc[len(df1)] = ["Quasicount function exponent", args.quasiY]

    df2 = spectrum.copy().rename(columns = {'mz' : 'Peak m/z', 'intensity' : 'Intensity (raw counts)', 'quasi' : 'Intensity (quasicounts)'}).drop(columns = ['formula_mass', 'formula', 'all_formulas'], errors = 'ignore')
    if args.formula:
        l = [';'.join([f"{form} ({mass:.5f})"]) if mass else None for form, mass in zip(bestFormulas, bestFormulaMasses)]
        df2['Peak formulas'] = l
    outExcelFile = os.path.join(args.outdir, f"{args.out_pref}.xlsx")
    writer = pd.ExcelWriter(outExcelFile, engine = 'xlsxwriter')
    df1.to_excel(writer, sheet_name = "Stats", header = False, index = False)
    df2.to_excel(writer, sheet_name = "Spectrum_Filtered", index = False)
    args.raw_spectrum.to_excel(writer, sheet_name = "Raw spectrum", index = False)
    writer.close()

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    process(**vars(args))
    
