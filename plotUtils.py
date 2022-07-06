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

def mirrorPlot( mzs_a, mzs_b, intensities_a, intensities_b, formulas_a = None, formulas_b = None):
    fig, ax = plt.subplots(figsize = (12,9))
    # plt.stem(df['mz_A'], df['intensity_A'], linefmt = 'b-', markerfmt = ' ', basefmt = ' ')
    vlinesA = ax.vlines(mzs_a, 0, intensities_a, color = '#67a9cf', alpha = 0.9)
    vlinesB = ax.vlines(mzs_b, 0, -intensities_b, color = '#ef8a62', alpha = 0.9)
    ax.axhline(0, 0, 1, color = 'black', alpha = 0.75, linewidth = 0.5)

    fig.canvas.draw()

    @FuncFormatter
    def my_formatter(x, pos):
        return( "{:.2e}".format(abs(x)))
    ax.get_yaxis().set_major_formatter(my_formatter)
    # ax.set_yticklabels([abs(x) for x in ax.get_yticks()])
    # ax.get_yaxis().set_major_formatter(ScalarFormatter(format_data = lambda x : abs(x)))
    # plt.ticklabel_format(axis = 'y', style = 'sci',  scilimits=(0,0))


    # ax.get_yaxis().get_major_formatter().set_scientific(True)
    # ax.get_yaxis().set_major_formatter(ScalarFormatter())
    # ax.ticklabel_format(style = 'sci', axis = 'y')
    # plt.ticklabel_format(style='sci', axis='y') # , scilimits=(0,0)


    # # y1, y2 = ax.get_ylim()
    # ax.set_ylim([1.1 * x for x in ax.get_ylim()])
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
    for i_row in range(labelCutoff):
        mz_a, int_a, formula_a = packageA[i_row]
        mz_b, int_b, formula_b = packageB[i_row]
        # if i_row == labelCutoff:
            # break
        # texts.append(plt.text(row['mz_A'], row['intensity_A'] + tAdjust, row['formula_A'], ha = 'center'))
        if formula_a != None:
            texts.append(plt.text(mz_a, int_a, formula_a, ha = 'center'))
        if formula_b != None:
            texts.append(plt.text(mz_b, -int_b, formula_b, ha = 'center'))
        # texts.append(plt.text(row['mz_B'], - row['intensity_B'] - tAdjust, row['formula_B'], ha = 'center'))
        # texts.append(plt.text(row['mz_B'], - row['intensity_B'], row['formula_B'], ha = 'center'))
        # break

    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black'), ha = 'center') # , add_objects = [vlinesA, vlinesB]



    plt.xlabel('m/z')
    plt.ylabel('Intensity')
    return(fig, ax)


