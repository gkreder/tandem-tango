############################################################
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
from joblib import Parallel, delayed
import pickle as pkl
import sys
import os
import numpy as np
# import pyopenms as pyms
import warnings
############################################################
import formulaUtils
import plotUtils
############################################################

quasiCutoffDefault = 5


def run_matching(args):

    args.resClearance = 200 / args.R
    args.resClearance = float(args.resClearance)
    args.matchAcc = 100 / args.R
    args.matchAcc = float(args.matchAcc)
    args.subFormulaTol = 100 / args.R

    args.index1 = args.index1 - args.startingIndex
    args.index2 = args.index2 - args.startingIndex





    # Matching accuracy m/z tolerance must be <= 0.5 x (resolution clearance width) or the peak matching will break
    if args.matchAcc > ( args.resClearance * 0.5):
        sys.exit('Error - the matching accuracy (matchAcc) must at most 0.5 * resolution clearance width (resClearance)')

    if args.quasiY > 0:
        sys.exit('Error - quasiY must be a non-positive number')

    if str(args.DUMin).capitalize() == "None":
        args.DUMin = None

    if args.PEL != None and args.PDPL != None:
        sys.exit("\nError - can only input either a Peak exclusion list or a Predefined peak list but not both\n")

    if args.PDPL != None and args.quasiCutoff == quasiCutoffDefault:
        args.quasiCutoff = 0

    if args.PEL != None:
        with open(args.PEL, 'r') as f:
            excludePeaks = [float(x.strip()) for x in f.readlines()]

    if args.PDPL != None:
        with open(args.PDPL, 'r') as f:
            prePeaks = np.array([float(x.strip()) for x in f.readlines()])
            prePeaks = sorted(prePeaks)
            pDiffs = np.diff(prePeaks)
            badDiffs = np.where(pDiffs <= args.resClearance)[0]
            if len(badDiffs) > 0:
                sys.exit(f"\nError - Some Predefined peak list m/z values are spaced too closely for given resolution width e.g. {prePeaks[badDiffs[0]]} and {prePeaks[badDiffs[0] + 1]}\n")



    suf1 = os.path.splitext(os.path.basename(os.path.basename(args.mzml1)))[0]
    suf2 = os.path.splitext(os.path.basename(os.path.basename(args.mzml2)))[0]
    ind1 = args.index1 + args.startingIndex
    ind2 = args.index2 + args.startingIndex
    if args.parentFormula == None:
        formString = "noFormula"
    else:
        formString = "formula"
    baseOutFileName = f"{suf1}_Scan_{ind1}_vs_{suf2}_Scan_{ind2}_{formString}"

    ############################################################
    # Spectrum Filtering
    ############################################################
    # outSuff = "230206_output_quasi5_20V_V9"

    os.system(f'mkdir -p {args.outDir}')
    with open(os.path.join(args.outDir, 'argsFiltering.txt'), 'w') as f:
        print(vars(args), file = f)


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
    dataBackup = [x.copy() for x in data]
        
    # pyms.IonSource.Polarity.NEGATIVE
    if len(set([d['polarity'] for d in data])) != 1:
        sys.exit('Error - the spectra polarities must match')
    polarity = ['Negative', 'Positive'][set([d['polarity'] for d in data]).pop()]

    if set([d['spec'].getMSLevel() for d in data]) != set([2]):
        for i_d, d in enumerate(data):
            if d['spec'].getMSLevel() != 2:
                sys.exit(f'Error - spectrum {i_d + 1} has MS level {d["spec"].getMSLevel()}')



    grayData = []
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

        # Conversion to quasicounts by dividing the intensity by gamma_i(1 + delta)
        d['quasi'] = d['intensities'] / ( args.quasiX *  ( np.power(d['mzs'], args.quasiY) ) )
        grayData.append(d.copy())

        if args.PDPL != None: # Step 11 of the word template
            # Eliminating all peaks not within match accuracy of a peak in the pre-defined peak list
            diffSearch = np.min(np.abs(d['mzs'][:, np.newaxis] - prePeaks), axis = 1)
            indices = np.where(diffSearch <= args.matchAcc)
            d['mzs'] = d['mzs'][indices]
            d['intensities'] = d['intensities'][indices]
            d['quasi'] = d['quasi'][indices]
        
        if len(d['quasi']) != 0:
            # Eliminating peaks with quasicount < quasicount intensity cutoff
            indices = np.where(d['quasi'] >= args.quasiCutoff)
            d['intensities'] = d['intensities'][indices]
            d['mzs'] = d['mzs'][indices]
            d['quasi'] = d['quasi'][indices]

        if len(d['quasi']) != 0:
            # Eliminating peaks with m/z > parent peak m/z
            indices = np.where(d['mzs'] <= args.parentMZ + args.matchAcc)
            d['mzs'] = d['mzs'][indices]
            d['intensities'] = d['intensities'][indices]
            d['quasi'] = d['quasi'][indices]
            
            # For plotting purposes
            grayIndices = np.where(grayData[i_d]['mzs'] <= args.parentMZ + args.matchAcc)
            grayData[i_d]['mzs'] = grayData[i_d]['mzs'][grayIndices]
            grayData[i_d]['intensities'] = grayData[i_d]['intensities'][grayIndices]
            grayData[i_d]['quasi'] = grayData[i_d]['quasi'][grayIndices]

        
        if len(d['quasi']) != 0 and args.PEL != None and args.PDPL == None:
            # Eliminating all peaks within match accuracy of peak exclusion list peaks
            for mze in excludePeaks:
                indices = np.where(np.abs(d['mzs'] - mze) > args.matchAcc)
                d['mzs'] = d['mzs'][indices]
                d['intensities'] = d['intensities'][indices]
                d['quasi'] = d['quasi'][indices]
        
        if len(d['quasi']) != 0:
            # Eliminate all peaks with normalized intensity < relative intensity cutoff
            indices = np.where((d['intensities'] / max(d['intensities'])) >= args.relCutoff)
            d['intensities'] = d['intensities'][indices]
            d['mzs'] = d['mzs'][indices]
            d['quasi'] = d['quasi'][indices]

        if len(d['quasi']) != 0:
            # Sort remaining peaks in spectrum from most to least intense
            sOrder = np.argsort(d['intensities'])[::-1]
            d['intensities'] = d['intensities'][sOrder]
            d['mzs'] = d['mzs'][sOrder]
            d['quasi'] = d['quasi'][sOrder]

        if len(d['quasi']) != 0:
            # print("Resolution clearance....")
            # For each peak (from most intense to least), eliminate any peaks within the closed interval [m/z +- resolution clearance]
            newInts = []
            newMzs = []
            newQuasis = []
            ints = np.array(d['intensities'].copy())
            mzs = np.array(d['mzs'].copy())
            quasis = np.array(d['quasi'].copy())
            i = 0
            while len(ints) > 0:
                i += 1
                mz = mzs[0]
                intensity = ints[0]
                quasi = quasis[0]
                keepIndices = np.where(np.abs(mzs - mz) > args.resClearance)
                newInts = newInts + [intensity] # + list(ints[keepIndices])
                newMzs = newMzs + [mz]  #+ list(mzs[keepIndices])
                newQuasis = newQuasis + [quasi]
                ints = ints[keepIndices]
                mzs = mzs[keepIndices]
                quasis = quasis[keepIndices]
            d['intensities'] = np.array(newInts)
            d['mzs'] = np.array(newMzs)
            d['quasi'] = np.array(newQuasis)

        if not (len(d['quasi']) > 0 and d['quasi'].sum() >= args.minSpectrumQuasiCounts and len(d['quasi']) >= args.minTotalPeaks):
            errorFile = os.path.join(args.outDir, f"{baseOutFileName}.log")
            with open(errorFile, 'w') as f:
                if len(d['quasi']) == 0:
                    print('The number of quasicounted peaks equals 0', file = f)
                if d['quasi'].sum() < args.minSpectrumQuasiCounts:
                    print(f"The spectrum quasicount sum ({d['quasi'].sum()}) did not exceed the minimum required ({args.minSpectrumQuasiCounts})", file = f)
                if len(d['quasi']) < args.minTotalPeaks:
                    print(f"There were too few peaks ({len(d['quasi'])}) compared to the required minimum ({args.minTotalPeaks})", file = f)
            return
            # sys.exit('gkreder - I put this in here to stop empty spectra. Refer to 2022-11-25_comparisonSpcetrumFiltering.ipynb')



    ############################################################
    # P-Value Calculation
    ############################################################
    def calc_G2(a_i, b_i, S_A, S_B, M):
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



    dfs = []
    for d in data:
        df = pd.DataFrame({"mz" : d['mzs'], "intensity" : d['intensities'], 'quasi' : d['quasi']}) 
        # if args.parentFormula != None:
        #     df['formula'] = d['formulas']
        #     df['m/z_calculated'] = d['formulaMasses']
        df = df.sort_values(by = 'mz').reset_index(drop = True)
        df['mz_join'] = df['mz']
        # if args.parentFormula != None:
            # df = df = df.dropna(subset = [f'formula'])
        dfs.append(df)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        df = pd.merge_asof(dfs[0], dfs[1], tolerance = args.matchAcc, on = 'mz_join', suffixes = ('_A', '_B'), direction = 'nearest').drop(columns = 'mz_join')
        for i, suf in enumerate(['A', 'B']):
            dft = dfs[i]
            dfRem = dft[~dft['mz'].isin(df[f'mz_{suf}'])]
            dfRem = dfRem.rename(columns = {x : f"{x}_{suf}" for x in ['mz', 'intensity', 'formula', 'm/z_calculated', 'quasi']}).drop(columns = 'mz_join') #quasi?
            df = pd.concat([df, dfRem])
        df = df.reset_index(drop = True)
        # Mapping parent formulas using the new multi-formula version
        if args.parentFormula != None:
            # print('Formula mapping...')
            form = molmass.Formula(args.parentFormula).formula

            allForms = formulaUtils.generateAllForms(form)

            formulas = [None for x in range(len(df))]
            formulaMasses = [None for x in range(len(df))]
            
            for i, (mz_a, mz_b) in enumerate(df[['mz_A', 'mz_B']].values):
                if np.isnan(mz_a):
                    # bestForm, thMass, error = formulaUtils.findBestForm(mz_b, form, toleranceDa = args.subFormulaTol, DuMin=args.DUMin)
                    bestForms, thMasses, errors = formulaUtils.findBestForms(mz_b, allForms, toleranceDa = args.subFormulaTol, DuMin=args.DUMin)
                elif np.isnan(mz_b):
                    # bestForm, thMass, error = formulaUtils.findBestForm(mz_a, form, toleranceDa = args.subFormulaTol, DuMin=args.DUMin)
                    bestForms, thMasses, errors = formulaUtils.findBestForms(mz_a, allForms, toleranceDa = args.subFormulaTol, DuMin=args.DUMin)
                else:
                    # bestForm, thMass, error = formulaUtils.findBestForm(np.mean((mz_a, mz_b)), form, toleranceDa = args.subFormulaTol, DuMin=args.DUMin)
                    bestForms, thMasses, errors = formulaUtils.findBestForms(np.mean((mz_a, mz_b)), allForms, toleranceDa = args.subFormulaTol, DuMin=args.DUMin)
                bestForm = bestForms[0]
                if bestForm == None:
                    continue
                formulas[i] = ", ".join([str(x).replace("None", "") for x in bestForms])
                formulaMasses[i] = ", ".join([str(x) for x in thMasses])


            df['formula'] = np.array(formulas)
            df['m/z_calculated'] = np.array(formulaMasses)
            if args.PDPL == None:
                df = df.dropna(subset = [f'formula'])

        stats = {"Union" : {}, "Intersection" : {}}
        # stats = {"Intersection" : {}}
        dfCs = {}
        jaccardTemp = {}
        for join in stats.keys():
            if join == 'Union':
                dfC = df.copy()
            elif join == 'Intersection':
                dfC = df.dropna(subset = ['mz_A', 'mz_B']).copy()
            
            
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
            stats[join]['S_A (raw)'] = dfC['intensity_A'].sum()
            stats[join]['S_B (raw)'] = dfC['intensity_B'].sum()


            

            # Calculate D^2
            D2, pval_D2 = calc_D2(a_i.fillna(value = 0.0), b_i.fillna(value = 0.0), S_A, S_B, M)
            stats[join]['D^2'] = D2
            stats[join]['pval_D^2'] = pval_D2

            # Calculate G^2
            G2, pval_G2 = calc_G2(a_i.fillna(value = 0.0), b_i.fillna(value = 0.0), S_A, S_B, M)
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
            JSD = H( ( 0.5 * ( p_Ai.fillna(0.0) + p_Bi.fillna(0.0) ) ) ) - ( 0.5 * H_pA ) - ( 0.5 *  H_pB  ) 
            stats[join]['JSD'] = JSD

            # Cosine Distance
            def sqF(x):
                return(np.sqrt( np.power( x, 2 ).sum() ) )
            num = (p_Ai * p_Bi).sum()
            denom = sqF(p_Ai) * sqF(p_Bi)
            # CSD = 1 - ( num / denom )
            CSD = ( num / denom )
            stats[join]['Cosine Similarity'] = CSD

            dfC[f"D^2_{join}"] = None
            dfC[f"G^2_{join}"] = None
            dfC[f"SR_A_{join}"] = None
            dfC[f"SR_B_{join}"] = None

            for i_row, row in dfC.iterrows():
                a_i = row['quasi_A']
                b_i = row['quasi_B']
                if ((np.isnan(a_i)) or (np.isnan(b_i))) and (join == 'Intersection'): 
                    continue
                a_i = pd.Series([a_i])
                b_i = pd.Series([b_i])
                D2, _ = calc_D2(a_i.fillna(value = 0.0), b_i.fillna(value = 0.0), S_A, S_B, M)
                G2, _ = calc_G2(a_i.fillna(value = 0.0), b_i.fillna(value = 0.0), S_A, S_B, M)
                # Calculate Standardized Residual (SR)
                SR_ai, SR_bi = calc_SR(a_i.fillna(value = 0.0), b_i.fillna(value = 0.0), S_A, S_B)
                SR_ai = SR_ai.values[0]
                SR_bi = SR_bi.values[0]
                dfC.at[i_row, f"D^2_{join}"] = D2
                dfC.at[i_row, f"G^2_{join}"] = G2
                dfC.at[i_row, f"SR_A_{join}"] = SR_ai
                dfC.at[i_row, f"SR_B_{join}"] = SR_bi
            dfCs[join] = dfC

        jaccard = jaccardTemp['Intersection'] / jaccardTemp['Union']
        stats['Union']['Jaccard'] = jaccard 
        if args.parentFormula == None:
            pft = "Not specified"
        else:
            pft = args.parentFormula 
        stats['Union']['Precursor formula'] = pft
        stats['Union']['Polarity'] = polarity
        if args.quasiY == 0.0:
            stats['Union']['Quasicount scaling function'] = f"{args.quasiX}"
        else:
            stats['Union']['Quasicount scaling function'] = f"{args.quasiX} x [m/z]^{args.quasiY}"

    df = pd.merge(dfCs['Union'], dfCs['Intersection'], how = 'outer')


    dfOut = df.rename(columns = {'mz_A' : 'm/z_A',
    'mz_B' : 'm/z_B',
    "intensity_A" : "I_A (raw intensity)",
    "intensity_B" : "I_B (raw intensity)",
    "quasi_A" : "a (quasicounts)",
    "quasi_B" : "b (quasicounts)",
    "formula" : "Formula"
    })

    if args.parentFormula == None:
        dfOut = dfOut[['m/z_A', 'm/z_B', 'I_A (raw intensity)', 'I_B (raw intensity)', 'a (quasicounts)', 'b (quasicounts)', "D^2_Union", "G^2_Union", "D^2_Intersection", "G^2_Intersection", "SR_A_Union", "SR_A_Intersection",  "SR_B_Union", "SR_B_Intersection"]]
    else:
        # dfOut = dfOut[['formula_A', 'formula_B', 'm/z_a', 'm/z_b', 'I_a', 'I_b', 'a', 'b', "D^2_Union", "G^2_Union", "D^2_Intersection", "G^2_Intersection"]]
        dfOut = dfOut[['Formula', 'm/z_A', 'm/z_B', 'I_A (raw intensity)', 'I_B (raw intensity)', 'a (quasicounts)', 'b (quasicounts)', "D^2_Union", "G^2_Union", "D^2_Intersection", "G^2_Intersection", "SR_A_Union", "SR_A_Intersection",  "SR_B_Union", "SR_B_Intersection"]]


    dfStats = pd.DataFrame(stats)
    if len(dfCs['Intersection']) <= 1: # Filter out based on M = 0 or 1
        sys.exit("gkreder - put this in outside of loop context. See 2022-11-25_comparisonPvalCalc.ipynb for original error")
    # outRows.append([iPair, spectraMeta.loc[iSpec]['sourceFile'], 
                    # spectraMeta.iloc[iSpec]['specIndex_mzML'], 
                    # spectraMeta.loc[jSpec]['sourceFile'], 
                    # spectraMeta.iloc[jSpec]['specIndex_mzML']] + list(dfStats['Intersection'][['M', 'S_A', 'S_B', 'D^2', 'pval_D^2', 'G^2', 'pval_G^2']].values))
    # outFilePair = os.path.join(outDirInt, onp.replace('.pkl', f"_{iPair}.xlsx"))
    # writer = pd.ExcelWriter(outFilePair, engine = 'xlsxwriter')
    # dfStats.drop(labels =['quasi_A', 'quasi_B'], axis = 0).rename(index = {'S_A' : 's_A (quasicounts)', 'S_B' : 's_B (quasicounts)'}).to_excel(writer, sheet_name = "Stats")
    # dfOut.drop(labels = [x for x in dfOut.columns if "_Union" in x], axis = 1).dropna(subset = ['m/z_A', 'm/z_B']).to_excel(writer, sheet_name = "Spectra_Intersection", index = False)
    # dfOut.drop(labels = [x for x in dfOut.columns if "_Intersection" in x], axis = 1).to_excel(writer, sheet_name = "Spectra_Union", index = False)
    # # dfOut.to_excel(writer, sheet_name = "Spectra", index = False)
    # writer.save()
    outExcelFile = os.path.join(args.outDir, f"{baseOutFileName}.xlsx")

    writer = pd.ExcelWriter(outExcelFile, engine = 'xlsxwriter')
    dfStats.drop(labels =['quasi_A', 'quasi_B'], axis = 0).rename(index = {'S_A' : 's_A (quasicounts)', 'S_B' : 's_B (quasicounts)'}).to_excel(writer, sheet_name = "Stats")
    dfOut.drop(labels = [x for x in dfOut.columns if "_Union" in x], axis = 1).dropna(subset = ['m/z_A', 'm/z_B']).to_excel(writer, sheet_name = "Spectra_Intersection", index = False)
    dfOut.drop(labels = [x for x in dfOut.columns if "_Intersection" in x], axis = 1).to_excel(writer, sheet_name = "Spectra_Union", index = False)
    for isuf, suf in enumerate(['A', 'B']):
        pd.DataFrame({f"m/z_{suf}" : dataBackup[isuf]['mzs'], f"I_{suf} (raw intensity)" : dataBackup[isuf]['intensities']}).to_excel(writer, sheet_name = f"Spectrum_{suf}_Raw", index = False) 
    # dfOut.to_excel(writer, sheet_name = "Spectra", index = False)
    writer.save()


    for plotJoin in ["Union", "Intersection"]:
        if plotJoin == "Intersection":
            dfPlot = df.dropna(subset = ['mz_A', 'mz_B'])
        elif plotJoin == 'Union':
            dfPlot = df.copy()
        sideText = ""
        if args.parentFormula != None:
            sideText += f"Parent Formula : {args.parentFormula}"
        sideText += f"\nParent m/z : {args.parentMZ:.5f}"
        sideText += f"\nM {plotJoin} : {int(dfStats[plotJoin].loc['M'])}"
        sideText += f"\np-val (D^2) : {dfStats[plotJoin].loc['pval_D^2']:.2e}"
        sideText += f"\np-val (G^2) : {dfStats[plotJoin].loc['pval_G^2']:.2e}"
        sideText += f"\ns_A (quasi) : {dfStats[plotJoin].loc['quasi_A']:.2e}"
        sideText += f"\ns_B (quasi) : {dfStats[plotJoin].loc['quasi_B']:.2e}"
        sideText += f"\nH(p_A) : {dfStats[plotJoin].loc['Entropy_A']:.2e}"
        sideText += f"\nH(p_B) : {dfStats[plotJoin].loc['Entropy_B']:.2e}"
        sideText += f"\nPP(p_A) : {dfStats[plotJoin].loc['Perplexity_A']:.2e}"
        sideText += f"\nPP(p_B) : {dfStats[plotJoin].loc['Perplexity_B']:.2e}"
        sideText += f"\ncos(p_A, p_B) : {dfStats[plotJoin].loc['Cosine Similarity']:.2e}"
        sideText += f"\nJSD(p_A, p_B) : {dfStats[plotJoin].loc['JSD']:.2e}"
        if plotJoin == "Intersection":
            sideText += f"\nJaccard : {dfStats['Union'].loc['Jaccard']:.2e}"
        else:
            sideText += "\n"
        
        # Normalized 0 to 1 scale
        fig, ax = plotUtils.mirrorPlot(dfPlot['mz_A'], dfPlot['mz_B'], dfPlot['intensity_A'], dfPlot['intensity_B'], None, None, normalize = True, sideText = sideText)
        ax.set_xlim([0, ax.get_xlim()[1]])
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        ylimMax = max([abs(x) for x in ylim])
        ylimRange = ylim[1] - ylim[0]
        # plt.text(xlim[1] + ( ( xlim[1] - xlim[0] ) *  0.025 ), 0, sideText)
        plt.text(xlim[1] + ( ( xlim[1] - xlim[0] ) *  0.025 ), ylimMax - ( 0.050 * ylimRange ), f"{plotJoin}:", fontsize = 20)
        plt.text(xlim[0] + ( ( xlim[1] - xlim[0] ) *  0.01 ), ylimMax - ( .03 * ylimRange ), "Spectrum A", fontsize = 15, fontfamily = 'DejaVu Sans')
        plt.text(xlim[0] + ( ( xlim[1] - xlim[0] ) *  0.01 ), -ylimMax + ( .03 * ylimRange ), "Spectrum B", fontsize = 15, fontfamily = 'DejaVu Sans')
        # plt.text(xlim[1] + ( ( xlim[1] - xlim[0] ) *  0.025 ), 1.025, f"{plotJoin}:", fontsize = 25)
        # plt.text(xlim[0] + ( ( xlim[1] - xlim[0] ) *  0.01 ), 1.03, "Spectrum A", fontsize = 12, fontfamily = 'DejaVu Sans')
        # plt.text(xlim[0] + ( ( xlim[1] - xlim[0] ) *  0.01 ), -1.07, "Spectrum B", fontsize = 12, fontfamily = 'DejaVu Sans')
        plotFilePair = os.path.join(args.outDir, f"{baseOutFileName}_{plotJoin}_plot.svg")
        # plotFilePair = outFilePair.replace('.xlsx', f'_{plotJoin}.pdf')
        # plt.title(os.path.basename(plotFilePair).replace('.pdf', ''))
        plot_title = f"{suf1} Scan {ind1} (A) vs.\n {suf2} Scan {ind2} (B) [{plotJoin}]"
        plt.title(f"{plot_title}")
        plt.ylabel("Relative intensity (quasicounts)")
        plt.savefig(plotFilePair, bbox_inches = 'tight') 
        plt.close()

        # Absolute quasicount scale 
        dfPlot = dfPlot.loc[lambda x : (x['quasi_A'] > 1) & (x['quasi_B'] > 1)]
        gda, gdb = grayData
        fig, ax = plotUtils.mirrorPlot(gda['mzs'], gdb['mzs'], np.log10(gda['quasi']), np.log10(gdb['quasi']), None, None, normalize = False, sideText = sideText, overrideColor = "gray")
        fig, ax = plotUtils.mirrorPlot(dfPlot['mz_A'], dfPlot['mz_B'], np.log10(dfPlot['quasi_A']), 
                                    np.log10(dfPlot['quasi_B']), None, None, 
                                    normalize = False, sideText = sideText,
                                    fig = fig, ax = ax)
        ax.set_xlim([0, ax.get_xlim()[1]])
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        ylimMax = max([abs(x) for x in ylim])
        ylimRange = ylim[1] - ylim[0]
        # plt.text(xlim[1] + ( ( xlim[1] - xlim[0] ) *  0.025 ), 0, sideText)
        plt.text(xlim[1] + ( ( xlim[1] - xlim[0] ) *  0.025 ), ylimMax - ( 0.050 * ylimRange ), f"{plotJoin}:", fontsize = 20)
        plt.text(xlim[0] + ( ( xlim[1] - xlim[0] ) *  0.01 ), ylimMax - ( .03 * ylimRange ), "Spectrum A", fontsize = 15, fontfamily = 'DejaVu Sans')
        plt.text(xlim[0] + ( ( xlim[1] - xlim[0] ) *  0.01 ), -ylimMax + ( .03 * ylimRange ), "Spectrum B", fontsize = 15, fontfamily = 'DejaVu Sans')
        plotFilePair = os.path.join(args.outDir, f"{baseOutFileName}_{plotJoin}_quasicount_log_plot.svg")
        # plotFilePair = outFilePair.replace('.xlsx', f'_{plotJoin}_quasi.pdf')
        # plt.title(os.path.basename(plotFilePair).replace('.pdf', ''))
        plot_title = f"{suf1} Scan {ind1} (A) vs.\n {suf2} Scan {ind2} (B) [{plotJoin}]"
        plt.title(f"{plot_title}")
        plt.ylabel("Log10 absolute intensity (quasicounts)")
        ax.set_ylim((-ylimMax, ylimMax))
        plt.savefig(plotFilePair, bbox_inches = 'tight') 
        plt.close()



    # dfOut = pd.DataFrame(outRows, columns = ["pair_index", "mzML File 1","File 1 spectrum index (A)","mzML File 2","File 2 spectrum index (B)","M","s_A","s_B","D^2","p-val (D^2)","G^2","p-val (G^2)"])
    # # dfOut = dfOut[dfOut['M'] > 1].reset_index(drop = True)
    # dfOut.at[0, '# of spectrum pairs compared'] = len(dfOut)
    # for pvalType in ['D^2', 'G^2']:
    #     # plt.plot(dfOut['p-val (D^2)'])
    #     data = dfOut[f'p-val ({pvalType})']
    #     # scipy.stats.kstest(th, th)
    #     res = scipy.stats.kstest(data, 'uniform')
    #     kspval, ksstat = res.pvalue, res.statistic
    #     x = np.sort(data)
    #     y = 1. * np.arange(len(data)) / (len(data) - 1)
    #     th = scipy.stats.uniform().cdf(x)
    #     # th = np.arange(0, 1.01, 0.01)
    #     fig, ax = plt.subplots()
    #     plt.plot(x,y)
    #     plt.plot(x, th)
    #     title = f"{compoundRow['Compounds']} {args.CE}V {pvalType} {args.mode}"
    #     plt.title(title)
    #     plt.figtext(0.92, 0.5, f"N: {len(dfOut)}\nks-stat:{ksstat:.3f}\npval: {kspval:.3f}", fontsize=14, ha = 'left')
    #     plotFile = os.path.basename(args.inFile).replace('.pkl', f'_{pvalType.replace("^", "")}.pdf')
    #     plotFile = os.path.join(args.outDir, plotFile)
    #     plt.savefig(plotFile, bbox_inches = 'tight')
    #     plt.close()
    #     dfOut.at[0, f'K-S test {pvalType} p-val'] = kspval
    # dfOut.insert(12, None, None)
    # outFile = os.path.join(args.outDir, os.path.basename(args.inFile).replace('.pkl', '.tsv'))
    # dfOut.to_csv(outFile, sep = '\t', index = False)


def get_args(arg_string = None):

    parser = argparse.ArgumentParser()
    parser.add_argument("--mzml1", required = True)
    parser.add_argument("--index1", required = True, type = int)
    parser.add_argument("--mzml2", required = True)
    parser.add_argument("--index2", required = True, type = int)
    parser.add_argument("--quasiX", required = True, type = float)
    parser.add_argument("--quasiY", required = True, type = float)
    parser.add_argument("--absCutoff", default = 0, type = float)
    parser.add_argument("--relCutoff", default = 0, type = float)
    parser.add_argument("--DUMin", default = -0.5, type = float)
    parser.add_argument("--PEL", default = None)
    parser.add_argument("--PDPL", default = None)
    parser.add_argument("--startingIndex", default = 0, type = int)
    parser.add_argument("--R", default = 10000, type = float)
    parser.add_argument("--gainControl", default = False)
    parser.add_argument("--quasiCutoff", default = quasiCutoffDefault, type = float)
    parser.add_argument("--minSpectrumQuasiCounts", default = 20, type = float)
    parser.add_argument("--minTotalPeaks", default = 2, type = float)
    parser.add_argument("--mode", required = True)
    parser.add_argument("--outDir", required = True)
    parser.add_argument("--parentMZ", required = True, type = float)
    parser.add_argument("--parentFormula", default = None)


    if arg_string == None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(arg_string.split(" "))
    
    return(args)



if __name__ == "__main__":
    args = get_args()
    run_matching(args)





 





