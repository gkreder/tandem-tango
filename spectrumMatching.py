# ----------------------------------------
import sys
import os
import numpy as np
import pyopenms as pyms
from tqdm.auto import tqdm
import molmass
import pandas as pd
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
import argparse
# ---------------------------------------
import plotUtils
import formulaUtils
# ----------------------------------------
parser = argparse.ArgumentParser()

# Required Arguments
parser.add_argument("--mzml1", required = True)
parser.add_argument("--index1", required = True, type = int)
parser.add_argument("--mzml2", required = True)
parser.add_argument("--index2", required = True, type = int)
parser.add_argument("--parentMZ", required = True, type = float)
parser.add_argument("--outFile", required = True)
parser.add_argument("--outPlot", required = True)
parser.add_argument("--quasiX", required = True, type = float)
parser.add_argument("--quasiY", required = True, type = float)
# Agilent (MH) starts at 1, MsConvert starts at 0. So Agilent index of 100 = mzML index of 99
parser.add_argument("--indexFormat", choices = ['Agilent', 'MsConvert'], required = True)

# Optional with defaults
parser.add_argument("--gainControl", default = False )
parser.add_argument("--absCutoff", default = 50, type = float)
parser.add_argument("--relCutoff", default = 0, type = float)
parser.add_argument("--resClearance", default = 0.02, type = float)
parser.add_argument("--matchAcc", default = 0.01, type = float)
parser.add_argument("--pltTitle", default = "")
# Parent Formula related args
parser.add_argument("--subFormulaTol", default = 0.01, type = float)
parser.add_argument("--DUMin", default = -0.5) # can be "NONE" as well

# Optional with no defaults
parser.add_argument("--parentFormula", default = None)


args = parser.parse_args()

# ----------------------------------------

def calc_G2(a_i, b_i, S_A, S_B):
    t1 = ( a_i + b_i ) * np.log( ( a_i + b_i ) / ( S_A + S_B ) ).apply(lambda x : 0 if np.isinf(x) else x)
    t2 = a_i * np.log(a_i / S_A).apply(lambda x : 0 if np.isinf(x) else x)
    t3 = b_i * np.log(b_i / S_B).apply(lambda x : 0 if np.isinf(x) else x)
    G2 = -2 * (t1 - t2 - t3).sum()
    pval_G2 = scipy.stats.chi2.sf(G2, df = M - 1)
    return(G2, pval_G2)

def calc_D2(a_i, b_i, S_A, S_B):
    num = np.power( ( S_B * a_i)  - ( S_A * b_i) , 2) 
    denom = a_i + b_i
    D2 = np.sum(num  / denom) / (S_A * S_B)
    
    pval_D2 = scipy.stats.chi2.sf(D2, df = M - 1)
    return(D2, pval_D2)
    

# ----------------------------------------

# Matching accuracy m/z tolerance must be <= 0.5 x (resolution clearance width) or the peak matching will break
if args.matchAcc > ( args.resClearance * 0.5):
    sys.exit('Error - the matching accuracy (matchAcc) must at most 0.5 * resolution clearance width (resClearance)')

if args.quasiY > 0:
    sys.exit('Error - quasiY must be a non-positive number')


if args.indexFormat == 'Agilent':
    args.index1 = args.index1 - 1
    args.index2 = args.index2 - 1



# ----------------------------------------

data = ({}, {})
for i_d, d in enumerate(data):
    ode = pyms.OnDiscMSExperiment()
    ode.openFile([args.mzml1, args.mzml2][i_d])
    d['ode'] = ode
    spec = ode.getSpectrum([args.index1, args.index2][i_d])
    d['spec'] = spec
    polarity = spec.getInstrumentSettings().getPolarity()
    d['polarity'] = polarity
    if args.gainControl:
        injTime = spec.getAcquisitionInfo()[0].getMetaValue('MS:1000927')
        if injTime == None:
            sys.exit(f"Error - gain control set to True but spectrum {i_d + 1} has no injection time in its header")
        d['injection time'] = injTime
    mzs, ints = spec.get_peaks()
    d['mzs'] = mzs
    d['intensities'] = ints
    
# pyms.IonSource.Polarity.NEGATIVE
if len(set([d['polarity'] for d in data])) != 1:
    sys.exit('Error - the spectra polarities must match')
if set([d['spec'].getMSLevel() for d in data]) != set([2]):
    for i_d, d in enumerate(data):
        if d['spec'].getMSLevel() != 2:
            sys.exit(f'Error - spectrum {i_d + 1} has MS level {d["spec"].getMSLevel()}')




for i_d, d in enumerate(data):
    print(f"---- Prepping Spectrum {i_d + 1} ----")
    if args.gainControl:
        injTime = d['injection time']
        d['uncorrected intensities'] = d['intensities'].copy()
        d['intensities'] = d['intensities'] * injTime
    # --------------------------------------------
    # Basic spectrum clean-up and filtering
    # --------------------------------------------
    
    print("Filtering peaks....")
    # Absolute intensity cutoff
    indices = np.where(d['intensities'] >= args.absCutoff)
    d['intensities'] = d['intensities'][indices]
    d['mzs'] = d['mzs'][indices]

    # Eliminating peaks with m/z < parent peak m/z
    indices = np.where(d['mzs'] <= args.parentMZ)
    d['mzs'] = d['mzs'][indices]
    d['intensities'] = d['intensities'][indices]

    # Eliminate all paeks with normalized intensity < relative intensity cutoff
    indices = np.where((d['intensities'] / max(d['intensities'])) >= args.relCutoff)
    d['intensities'] = d['intensities'][indices]
    d['mzs'] = d['mzs'][indices]

    # Sort remaining peaks in spectrum from most to least intense
    sOrder = np.argsort(d['intensities'])[::-1]
    d['intensities'] = d['intensities'][sOrder]
    d['mzs'] = d['mzs'][sOrder]

    print("Resolution clearance....")
    # For each peak (from most intense to least), eliminate any peaks within the closed interval [m/z +- resolution clearance]
    newInts = []
    newMzs = []
    ints = np.array(d['intensities'].copy())
    mzs = np.array(d['mzs'].copy())
    i = 0
    while len(ints) > 0:
        i += 1
        mz = mzs[0]
        intensity = ints[0]
        keepIndices = np.where(np.abs(mzs - mz) > args.resClearance)
        newInts = newInts + [intensity] # + list(ints[keepIndices])
        newMzs = newMzs + [mz]  #+ list(mzs[keepIndices])
        ints = ints[keepIndices]
        mzs = mzs[keepIndices]
    d['intensities'] = np.array(newInts)
    d['mzs'] = np.array(newMzs)

    # Filtering for formulas if parent formula is specified. Keep only peaks that are assigned a subformula within the 
    # specified user tolerance. Discard all other peaks
    if args.parentFormula != None:
        print('Formula mapping...')
        newMzs = []
        newInts = []
        formulas = []
        formulaMasses = []
        for i, mz in enumerate(tqdm(d['mzs'])):
            intensity = d['intensities'][i]
            form = molmass.Formula(args.parentFormula).formula
            bestForm, thMass, error = formulaUtils.findBestForm(mz, form, toleranceDa = args.subFormulaTol)
            if bestForm == None:
                continue
            newMzs.append(mz)
            newInts.append(intensity)
            formulas.append(bestForm)
            formulaMasses.append(thMass)
        d['intensities'] = np.array(newInts)
        d['mzs'] = np.array(newMzs)
        d['formulas'] = np.array(formulas)
        d['formulaMasses'] = np.array(formulaMasses)
        
        

dfs = []
for d in data:
    df = pd.DataFrame({"mz" : d['mzs'], "intensity" : d['intensities']})
    if args.parentFormula != None:
        df['formula'] = d['formulas']
        df['m/z_calculated'] = d['formulaMasses']
    df = df.sort_values(by = 'mz').reset_index(drop = True)
    df['mz_join'] = df['mz']
    if args.parentFormula != None:
        df = df = df.dropna(subset = [f'formula'])
    dfs.append(df)


df = pd.merge_asof(dfs[0], dfs[1], tolerance = 0.005, on = 'mz_join', suffixes = ('_A', '_B'), direction = 'nearest').drop(columns = 'mz_join')

for i, suf in enumerate(['A', 'B']):
    dft = dfs[i]
    dfRem = dft[~dft['mz'].isin(df[f'mz_{suf}'])]
    dfRem = dfRem.rename(columns = {x : f"{x}_{suf}" for x in ['mz', 'intensity', 'formula', 'm/z_calculated']}).drop(columns = 'mz_join')
    df = pd.concat([df, dfRem])


# Convert all peaks to "quasicounts" by dividing the intensity by gamma_i(1 + delta)
for suf in ['A', 'B']:
    df[f"quasi_{suf}"] = df[f'intensity_{suf}'] / ( args.quasiX *  ( np.power(df[f'mz_{suf}'], args.quasiY) ) )




stats = {"Union" : {}, "Intersection" : {}}
jaccardTemp = {}
for join in stats.keys():
    if join == 'Union':
        dfC = df.fillna(value = 0.0)
    elif join == 'Intersection':
        dfC = df.dropna(subset = ['mz_A', 'mz_B'])
    
    df[f"D^2_{join}"] = None
    df[f"G^2_{join}"] = None
    

    a_i = dfC['quasi_A']
    b_i = dfC['quasi_B']
    M = dfC.shape[0]
    jaccardTemp[join] = M
    stats[join]['quasi_A'] = dfC['quasi_A'].sum()
    stats[join]['quasi_B'] = dfC['quasi_B'].sum()
    stats[join]['M'] = M

    # S_A = dfC['intensity_A'].sum()
    # S_B = dfC['intensity_B'].sum()
    S_A = dfC['quasi_A'].sum()
    S_B = dfC['quasi_B'].sum()

    stats[join]['S_A'] = S_A
    stats[join]['S_B'] = S_B

    

    # Calculate D^2
    D2, pval_D2 = calc_D2(a_i, b_i, S_A, S_B)
    stats[join]['D^2'] = D2
    stats[join]['pval_D^2'] = pval_D2

    # Calculate G^2
    G2, pval_G2 = calc_G2(a_i, b_i, S_A, S_B)
    stats[join]['G^2'] = G2
    stats[join]['pval_G^2'] = pval_G2


    # p_Ai = a_i / S_A
    # p_Bi = b_i / S_B
    p_Ai = a_i / dfC['quasi_A'].sum()
    p_Bi = b_i / dfC['quasi_B'].sum()
    

    def H(x):
        h = -1 * np.sum( x * ( np.log(x) ) )
        return(h)

    # Calculate the Spectrum Entropy and perplexity
    H_pA = H(p_Ai)
    H_pB = H(p_Bi)

    PP_pA = np.exp(H_pA)
    PP_pB = np.exp(H_pB)
    stats[join]['Entropy_A'] = H_pA
    stats[join]['Entropy_B'] = H_pB
    stats[join]['Perplexity_A'] = PP_pA
    stats[join]['Perplexity_B'] = PP_pB


    # Jensen-Shannon divergence
    sTerm = (p_Ai + p_Bi).fillna(value = 0)
    JSD = H( ( 0.5 * ( p_Ai + p_Bi ) ) ) - ( 0.5 * H_pA ) - ( 0.5 *  H_pB  ) 
    stats[join]['JSD'] = JSD

    # Cosine Distance
    def sqF(x):
        return(np.sqrt( np.power( x, 2 ).sum() ) )
    num = (p_Ai * p_Bi).sum()
    denom = sqF(p_Ai) * sqF(p_Bi)
    # CSD = 1 - ( num / denom )
    CSD = ( num / denom )
    stats[join]['Cosine Similarity'] = CSD

    for i_row, row in df.iterrows():
        a_i = pd.Series([row['quasi_A']]).fillna(value = 0.0)
        b_i = pd.Series([row['quasi_B']]).fillna(value = 0.0)
        D2, _ = calc_D2(a_i, b_i, S_A, S_B)
        G2, _ = calc_G2(a_i, b_i, S_A, S_B)
        df.at[i_row, f"D^2_{join}"] = D2
        df.at[i_row, f"G^2_{join}"] = G2


jaccard = jaccardTemp['Intersection'] / jaccardTemp['Union']
stats['Union']['Jaccard'] = jaccard 




dfOut = df.rename(columns = {'mz_A' : 'm/z_a',
'mz_B' : 'm/z_b',
"intensity_A" : "I_a",
"intensity_B" : "I_b",
"quasi_A" : "a",
"quasi_B" : "b"
})

if args.parentFormula == None:
    dfOut = dfOut[['m/z_a', 'm/z_b', 'I_a', 'I_b', 'a', 'b', "D^2_Union", "G^2_Union", "D^2_Intersection", "G^2_Intersection"]]
else:
    dfOut = dfOut[['formula_A', 'formula_B', 'm/z_a', 'm/z_b', 'I_a', 'I_b', 'a', 'b', "D^2_Union", "G^2_Union", "D^2_Intersection", "G^2_Intersection"]]


dfStats = pd.DataFrame(stats)

writer = pd.ExcelWriter(args.outFile, engine = 'xlsxwriter')
dfStats.to_excel(writer, sheet_name = "Stats")
dfOut.to_excel(writer, sheet_name = "Spectra", index = False)
writer.save()

# if args.parentFormula != None:
#     fig, ax = plotUtils.mirrorPlot(df['mz_A'], df['mz_B'], df['intensity_A'], df['intensity_B'], df['formula_A'], df['formula_B'], normalize = True)
# else:
#     fig, ax = plotUtils.mirrorPlot(df['mz_A'], df['mz_B'], df['intensity_A'], df['intensity_B'], None, None, normalize = True)
fig, ax = plotUtils.mirrorPlot(df['mz_A'], df['mz_B'], df['intensity_A'], df['intensity_B'], None, None, normalize = True)

plt.title = args.pltTitle
plt.savefig(args.outPlot, bbox_inches = 'tight') 


