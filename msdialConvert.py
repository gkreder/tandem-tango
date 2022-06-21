#--------------------------------------------
# Gabe Reder
# gkreder@gmail.com
# Converts MSDIAL Output into BatchCorr-ready formats
#--------------------------------------------
import sys
import os
import pandas as pd
import numpy as np
import argparse
import re
#--------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--inFile', required = True)
parser.add_argument('--outDir', required = True)
parser.add_argument('--sampleRegex', default = "[0-9]")
args = parser.parse_args()
#--------------------------------------------

def prepDfs(args):
    
    with open(args.inFile, 'r') as f:
        lines = f.readlines()

    headerLines = [x.split('\t') for x in lines if x.startswith('\t')]
    hIndex = headerLines[0].index('Class')
    bodyLines = [x.split('\t') for x in lines if not x.startswith('\t')]
    headerLines += [bodyLines[0]]
    headerLines = [x[hIndex : ] for x in headerLines]
    nasI = np.where(np.array([x.strip() for x in headerLines[0]]) == 'NA')[0]
    if len(nasI) > 0:
        naI = np.min(nasI)
    else:
        naI = None


    dft = pd.DataFrame(bodyLines[1 : ], columns = bodyLines[0])
    dfIntensities = pd.DataFrame(dft[dft.columns[hIndex + 1 : ]])
    if naI != None:
        dfIntensities = pd.DataFrame(dfIntensities[dfIntensities.columns[ : naI - 1]])
    dfPeakData = pd.DataFrame(dft[dft.columns[ : hIndex]])

    h = [x[0] for x in headerLines]
    h[-1] = 'Sample'
    dfMeta = pd.DataFrame(x[1 : ] for x in headerLines).T
    dfMeta.columns = h
    dfMeta = pd.DataFrame(dfMeta[dfMeta['Class'].str.strip() != 'NA'])
    dfMeta['Group'] = dfMeta['Class']
    dfMeta['Batch'] = dfMeta['Sample'].str.split('_').str[1].str.split('W').str[0].str.strip('B').astype(int)
    dfMeta['Injection'] = dfMeta['Sample'].str.split('_').str[-1]
    groups = []
    for x in dfMeta['Sample']:
        sName = x.split('_')[-2]
        m = re.match(r'[0-9]', sName[0])
        if m:
            groups.append('Sample')
        else:
            groups.append(sName)
    dfMeta['Group'] = groups
    return(dfMeta, dfPeakData, dfIntensities)


if __name__ == '__main__':
    
    dfMeta, dfPeakData, dfIntensities = prepDfs(args)
    os.system(f"mkdir -p {args.outDir}")
    inPref = os.path.splitext(os.path.basename(args.inFile))[0]
    dfMeta.to_csv(os.path.join(args.outDir, f"{inPref}_Meta.tsv"), sep = '\t', index = False)
    dfPeakData.to_csv(os.path.join(args.outDir, f"{inPref}_PeakData.tsv"), sep = '\t', index = False)
    dfIntensities.to_csv(os.path.join(args.outDir, f"{inPref}_Intensities.tsv"), sep = '\t', index = False)
    