# -------------------------------------------------------
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
from matplotlib.ticker import ScalarFormatter, FuncFormatter
import matplotlib.ticker 
# -------------------------------------------------------

def singlePlot( mzs, intensities, formulas = None, normalize = True, rotation = 90, 
               sideText = None, fontfamily = "DejaVu Sans", fig = None, ax = None, overrideColor = None, linewidth = None, alpha = None):
    plt.rcParams['font.size'] = 16
    if fig == None and ax == None:
        fig, ax = plt.subplots(figsize = (12,9))
    # plt.stem(df['mz_A'], df['intensity_A'], linefmt = 'b-', markerfmt = ' ', basefmt = ' ')
    if normalize:
        intensities = intensities / intensities.max()
    linewidthOverride = linewidth if linewidth else 0.75
    linewidthDef = linewidth if linewidth else 0.25
    if not alpha:
        alpha = 0.3 if not overrideColor else 1.0
    if overrideColor == None:
        vlines = ax.vlines(mzs, 0, intensities, color = 'black', alpha = alpha, linewidth = linewidthOverride)
    else:
        vlines = ax.vlines(mzs, 0, intensities, color = overrideColor, alpha = alpha, linewidth = linewidthDef)
    # ax.axhline(0, 0, 1, color = 'black', alpha = 0.75, linewidth = 0.5)
    fig.canvas.draw()

    # if normalize:
    @FuncFormatter
    def my_formatter(x, pos):
        # return( "{:.2e}".format(abs(x))) # for scientific notation
        return( f"{abs(x)}")
    ax.get_yaxis().set_major_formatter(my_formatter)
    
    ylim = ax.get_ylim()
    tAdjust = ( ylim[1] - ylim[0] ) * 0.02
    labelCutoff = 10
    texts = []
    if np.all(formulas) == None:
        formulas = [None for x in mzs]
    package = sorted(zip(mzs, intensities, formulas), key = lambda x : x[1], reverse = True)
    for i_row in range(labelCutoff):
        if i_row >= len(package):
            break
        mz_t, int_t, formula_t = package[i_row]
        # if iPackage == 1:
            # int_t = -1 * int_t
        if formula_t != None:
            texts.append(plt.text(mz_t, int_t, formula_t, ha = 'center', rotation = rotation))        
    if len(texts) > 0:
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black'), ha = 'center') # , add_objects = [vlinesA, vlinesB]
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    if sideText != None:
        plt.text(xlim[1] + ( ( xlim[1] - xlim[0] ) *  0.025 ), 0.5, sideText, fontfamily = fontfamily)
    plt.xlabel('m/z')
    if normalize:
        plt.ylabel('Relative Intensity')
    else:
        plt.ylabel('Intensity')
    return(fig, ax)

def mirrorPlot( mzs_a, mzs_b, intensities_a, intensities_b, formulas_a = None, formulas_b = None, normalize = True, rotation = 90, 
               sideText = None, fontfamily = "DejaVu Sans", fig = None, ax = None, overrideColor = None):
    plt.rcParams['font.size'] = 16
    if fig == None and ax == None:
        fig, ax = plt.subplots(figsize = (12,9))
    # plt.stem(df['mz_A'], df['intensity_A'], linefmt = 'b-', markerfmt = ' ', basefmt = ' ')
    if normalize:
        intensities_a = intensities_a / intensities_a.max()
        intensities_b = intensities_b / intensities_b.max()
        # intensities_a = np.array([x / max(intensities_a) for x in intensities_a])
        # intensities_b = np.array([x / max(intensities_b) for x in intensities_b])
    if overrideColor == None:
        vlinesA = ax.vlines(mzs_a, 0, intensities_a, color = '#67a9cf', alpha = 0.9, linewidth = 0.75)
        vlinesB = ax.vlines(mzs_b, 0, -intensities_b, color = '#ef8a62', alpha = 0.9, linewidth = 0.75)
    else:
        vlinesA = ax.vlines(mzs_a, 0, intensities_a, color = overrideColor, alpha = 0.3, linewidth = 0.25)
        vlinesB = ax.vlines(mzs_b, 0, -intensities_b, color = overrideColor, alpha = 0.3, linewidth = 0.25)
    ax.axhline(0, 0, 1, color = 'black', alpha = 0.75, linewidth = 0.5)

    fig.canvas.draw()

    # if normalize:
    @FuncFormatter
    def my_formatter(x, pos):
        # return( "{:.2e}".format(abs(x))) # for scientific notation
        return( f"{abs(x)}")
    ax.get_yaxis().set_major_formatter(my_formatter)
    
    ylim = ax.get_ylim()
    tAdjust = ( ylim[1] - ylim[0] ) * 0.02

    labelCutoff = 10
    texts = []

    if np.all(formulas_a) == None:
        formulas_a = [None for x in mzs_a]
    if np.all(formulas_b) == None:
        formulas_b = [None for x in mzs_b]
    packageA = sorted(zip(mzs_a, intensities_a, formulas_a), key = lambda x : x[1], reverse = True)
    packageB = sorted(zip(mzs_b, intensities_b, formulas_b), key = lambda x : x[1], reverse = True)
    for iPackage, package in enumerate([packageA, packageB]):
        for i_row in range(labelCutoff):
            if i_row >= len(package):
                break
            mz_t, int_t, formula_t = package[i_row]
            if iPackage == 1:
                int_t = -1 * int_t
            if formula_t != None:
                texts.append(plt.text(mz_t, int_t, formula_t, ha = 'center', rotation = rotation))
    # for i_row in range(labelCutoff):
    #     mz_a, int_a, formula_a = packageA[i_row]
    #     mz_b, int_b, formula_b = packageB[i_row]
        
    #     if formula_a != None:
    #         texts.append(plt.text(mz_a, int_a, formula_a, ha = 'center', rotation = rotation))
    #     if formula_b != None:
    #         texts.append(plt.text(mz_b, -int_b, formula_b, ha = 'center', rotation = rotation))
        
    if len(texts) > 0:
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black'), ha = 'center') # , add_objects = [vlinesA, vlinesB]

    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    if sideText != None:
        plt.text(xlim[1] + ( ( xlim[1] - xlim[0] ) *  0.025 ), - ( ( ylim[1] - ylim[0] ) *  0.075 ), sideText, fontfamily = fontfamily)




    plt.xlabel('m/z')
    if normalize:
        plt.ylabel('Relative Intensity')
    else:
        plt.ylabel('Intensity')
    return(fig, ax)


