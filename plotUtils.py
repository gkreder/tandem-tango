# -------------------------------------------------------
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
from matplotlib.ticker import ScalarFormatter
# -------------------------------------------------------

def mirrorPlot( mzs_a, mzs_b, intensities_a, intensities_b, formulas_a = None, formulas_b = None):
    fig, ax = plt.subplots(figsize = (12,9))
    # plt.stem(df['mz_A'], df['intensity_A'], linefmt = 'b-', markerfmt = ' ', basefmt = ' ')
    vlinesA = ax.vlines(mzs_a, 0, intensities_b, color = '#67a9cf', alpha = 0.9)
    vlinesB = ax.vlines(mzs_b, 0, -intensities_b, color = '#ef8a62', alpha = 0.9)
    ax.axhline(0, 0, 1, color = 'black', alpha = 0.75, linewidth = 0.5)

    fig.canvas.draw()
    # labels = [item.get_text().replace('-', '').replace('−', '') for item in ax.get_yticklabels()]
    # ax.set_yticklabels(labels)
    ax.set_yticklabels([abs(x) for x in ax.get_yticks()])
    ax.get_yaxis().set_major_formatter(ScalarFormatter())
    plt.ticklabel_format(style='sci', axis='y') # , scilimits=(0,0)


    # y1, y2 = ax.get_ylim()
    ax.set_ylim([1.1 * x for x in ax.get_ylim()])
    ylim = ax.get_ylim()
    tAdjust = ( ylim[1] - ylim[0] ) * 0.02

    labelCutoff = 10
    texts = []

    if np.all(formulas_a) == None:
        formulas_a = [None for x in mzs_a]
    if np.all(formulas_b) == None:
        formulas_b = [None for x in mzs_b]
    for i_row, (mz_a, mz_b, int_a, int_b, formula_a, formula_b) in enumerate(sorted(zip(mzs_a, mzs_b, intensities_a, intensities_b, formulas_a, formulas_b), key = lambda x : x[2], reverse = True)):
        if i_row == labelCutoff:
            break
        # texts.append(plt.text(row['mz_A'], row['intensity_A'] + tAdjust, row['formula_A'], ha = 'center'))
        if formula_a != None:
            texts.append(plt.text(mz_a, int_a, formula_a, ha = 'center'))
        if formula_b != None:
            texts.append(plt.text(mz_b, -int_b, formula_b, ha = 'center'))
        # texts.append(plt.text(row['mz_B'], - row['intensity_B'] - tAdjust, row['formula_B'], ha = 'center'))
        # texts.append(plt.text(row['mz_B'], - row['intensity_B'], row['formula_B'], ha = 'center'))
        # break

    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black'), ha = 'center', add_objects = [vlinesA, vlinesB], expand_text = (1.05, 1.5))
    plt.xlabel('m/z')
    plt.ylabel('Intensity')
    return(fig, ax)


