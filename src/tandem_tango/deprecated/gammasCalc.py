###################################################
import sys
import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from pyopenms import *
import seaborn as sns
import scipy.stats
import matplotlib.pyplot as plt
from datetime import datetime
import math
import argparse
import formulaUtils
import molmass
import pickle as pkl
import shutil
###################################################
def get_args(arg_string = None):

    parser = argparse.ArgumentParser()
    parser.add_argument('--mzML', required = True, help = "Merged mzML containing all spectra of interest")
    parser.add_argument('--outDir', required = True)
    parser.add_argument('--inVoltages', required = True, help = "Input list of voltages to look at")
    parser.add_argument('--injectionTimeAdjust', required = True, choices = ['True', 'False', 'FALSE', 'TRUE'])
    parser.add_argument('--mode', required = True, choices = ['Pos', 'Neg'])
    # parser.add_argument("--parentFormula", required = True)
    parser.add_argument('--peakList', required = True, help = "curated peak list")
    parser.add_argument("--lowerPercent", type = float, default = 0.7)
    parser.add_argument("--upperPercent", type = float, default = 1.3)
    parser.add_argument("--numSlices", type = int, default = 15)
    parser.add_argument("--ticPeakCutoff", type = float, default = 0.02)
    parser.add_argument("--rawPeakCutoff", type = float, default = 0)
    parser.add_argument("--nSpectraFilter", type = int, default = 20)
    parser.add_argument("--mzTolInter", type = float, default = 0.01)
    parser.add_argument("--startScanIdx", default = None)
    parser.add_argument("--endScanIdx", default = None)
    parser.add_argument("--TIC_calculation", default = "peak_list", choices = ["peak_list", "total"], help = "Whether to calculate TIC using the peak list or the total spectrum TIC")

    if arg_string == None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(arg_string.split(" "))

    if type(args.injectionTimeAdjust) == str:
        args.injectionTimeAdjust = {"FALSE" : False, "TRUE" : True}[args.injectionTimeAdjust.upper()]
    return(args)

# parser.add_argument('--outPref', required = True, help = "Prefix for output filenames")

###################################################

def TIC(spec, df_peakList, args):
    if args.TIC_calculation == "peak_list":
        intensities = []
        for mz in df_peakList['m/z']:
            closest_peak = min(spec, key = lambda x: abs(x.getMZ() - mz) if abs(x.getMZ() - mz) < args.mzTolInter else float('inf'))
            
            if type(closest_peak) == float and closest_peak == float('inf'):
                continue
            intensities.append(closest_peak.getIntensity())
        return sum(intensities)
                               
    elif args.TIC_calculation == "total":
        return spec.calculateTIC()
    else:
        raise ValueError("Invalid TIC calculation method")


def calc_ce_column(ce, df_peakList, vMzmlDir, pl_mzs, args):
    # peak_list_fname = "/scratch/users/gkreder/media/gkreder/barracuda/data/mass_spec/meyer/210128_MS-MS_Comparison_Algorithm/210905_neg_gammas_calc/210825_QE_Dex_Neg_Peaks_formulas.tsv"
    # df_peakList = pd.read_csv(peak_list_fname, sep = '\t').sort_values(by = 'm/z').reset_index(drop = True)
    # pl_mzs = df_peakList['m/z'].values
    
    ce_out_pkl_fname = os.path.join(vMzmlDir, f"{ce}V.pkl")
    ce_out_mzml_fname = os.path.join(vMzmlDir, f"{ce}V.mzML")
        
    edges = pkl.load(open(ce_out_pkl_fname, 'rb'))
    ce_ondisc_exp = OnDiscMSExperiment()
    ce_ondisc_exp.openFile(ce_out_mzml_fname)
    ce_ns = ce_ondisc_exp.getNrSpectra()
    if args.injectionTimeAdjust:
        # ce_tics = [ce_ondisc_exp.getSpectrum(int(i)).calculateTIC() * ce_ondisc_exp.getSpectrum(int(i)).getAcquisitionInfo()[0].getMetaValue('MS:1000927') for i in range(ce_ns)]
        ce_tics = [TIC(ce_ondisc_exp.getSpectrum(int(i)), df_peakList, args) * ce_ondisc_exp.getSpectrum(int(i)).getAcquisitionInfo()[0].getMetaValue('MS:1000927') for i in range(ce_ns)]
    else:
        # ce_tics = [ce_ondisc_exp.getSpectrum(int(i)).calculateTIC() for i in range(ce_ns)]
        ce_tics = [TIC(ce_ondisc_exp.getSpectrum(int(i)), df_peakList, args) for i in range(ce_ns)]
        
    
    colnames = []
#     computeds = []
    out_cols_gammas = []
    out_cols_means = []
    out_cols_vars = []
    out_cols_avgFracs = []
    index_counts = []
    
    for ne in range(args.numSlices):
        el = edges[ne]
        er = edges[ne + 1]
        cname = f"{ce}_{el}_{er}"
        colnames.append(cname)        
        ne_indices = np.where((ce_tics >= el) & (ce_tics <= er))[0]
        index_counts.append(len(ne_indices))
        nSpectra_filter = args.nSpectraFilter
        if len(ne_indices) < nSpectra_filter:
#             computeds.append(False)
            out_cols_gammas.append([None for x in range(len(df_peakList))])
            out_cols_means.append([None for x in range(len(df_peakList))])
            out_cols_vars.append([None for x in range(len(df_peakList))])
            continue
        e_mzs = []
        e_ints = []
        pl_mzs_counts = np.zeros(len(df_peakList))
        tic_peak_cutoff = args.ticPeakCutoff
        avgFrac = []
        for ei in ne_indices:
            s = ce_ondisc_exp.getSpectrum(int(ei))
            s_mzs, s_ints = s.get_peaks()
            if args.injectionTimeAdjust:
                injection_time = s.getAcquisitionInfo()[0].getMetaValue('MS:1000927')
                s_ints = s_ints * injection_time
                # s_indices = np.where((s_ints <= (s.calculateTIC() * tic_peak_cutoff * injection_time)) & (s_ints >= args.rawPeakCutoff))[0]
                s_indices = np.where((s_ints <= (TIC(s, df_peakList, args) * tic_peak_cutoff * injection_time)) & (s_ints >= args.rawPeakCutoff))[0]
            else:
                # s_indices = np.where((s_ints <= (s.calculateTIC() * tic_peak_cutoff)) & (s_ints >= args.rawPeakCutoff))[0]
                s_indices = np.where((s_ints <= (TIC(s, df_peakList, args) * tic_peak_cutoff)) & (s_ints >= args.rawPeakCutoff))[0]
            avgFrac.append(len(s_indices) / len(s_ints))
            s_mzs = s_mzs[s_indices]
            s_ints = s_ints[s_indices]
            e_mzs.append(s_mzs)
            e_ints.append(s_ints)
            pl_mz_indices = np.where(np.min(np.abs(pl_mzs.reshape(pl_mzs.shape[0], 1) - s_mzs), axis = 1) <= args.mzTolInter)[0]
            pl_mzs_counts[pl_mz_indices] += 1
        avgFrac = sum(avgFrac) / len(avgFrac)
        # if ce == 200:
        #     import ipdb;ipdb.set_trace()
        bin_rows = []
        mean_rows = []
        var_rows = []
        avgFrac_rows = []
        # if ce == 200:
        #     import ipdb;ipdb.set_trace()
        for ri, (mz, formula) in df_peakList[['m/z', 'formula']].iterrows():
            if pl_mzs_counts[ri] != len(ne_indices):
                bin_rows.append(None)
                mean_rows.append(None)
                var_rows.append(None)
                avgFrac_rows.append(None)
                continue
            mz_ints = []
            for i_ei, ei in enumerate(ne_indices):
                s_mzs = e_mzs[i_ei]
                s_ints = e_ints[i_ei]
                mzi = np.argmin(np.abs(s_mzs - mz))
                mz_int = s_ints[mzi]
                mz_ints.append(mz_int)
            mz_ints = np.array(mz_ints)
            var = mz_ints.var()
            mean = mz_ints.mean()
            gamma = var / mean
            bin_rows.append(gamma)
            mean_rows.append(mean)
            var_rows.append(var)
            avgFrac_rows.append(avgFrac)
#         computeds.append(True)
        out_cols_gammas.append(bin_rows)
        out_cols_means.append(mean_rows)
        out_cols_vars.append(var_rows)
        out_cols_avgFracs.append(avgFrac_rows)
    return(colnames, index_counts, out_cols_gammas, out_cols_means, out_cols_vars, out_cols_avgFracs)






def main(args):
    os.system(f'mkdir -p {args.outDir}')
    os.chmod(args.outDir, 0o777)
    args.log_file = os.path.join(args.outDir, "run.log")

    
    with open(args.log_file, 'w') as f:
        print(' '.join(sys.argv), file = f)
        
    ondisc_exp = OnDiscMSExperiment()
    ondisc_exp.openFile(args.mzML)
    ns = ondisc_exp.getNrSpectra()
    tics = -1 * np.ones(ns)
    levels = -1 * np.ones(ns)
    polarities = -1 * np.ones(ns)
    voltages = -1 * np.ones(ns)
    injection_times = -1 * np.ones(ns)
    df_peakList = pd.read_csv(args.peakList, sep = '\t').sort_values(by = 'm/z').reset_index(drop = True)
    if 'formula' not in df_peakList.columns:
        df_peakList['formula'] = None
    for i in tqdm(range(ns)):
        spec = ondisc_exp.getSpectrum(i)
        # tics[i] = spec.calculateTIC()
        tics[i] = TIC(spec, df_peakList, args)
        levels[i] = spec.getMSLevel()
        polarities[i] = spec.getInstrumentSettings().getPolarity()
        if args.injectionTimeAdjust:
            injection_times[i] = spec.getAcquisitionInfo()[0].getMetaValue('MS:1000927')
        if spec.getMSLevel() > 1:
            p = spec.getPrecursors()[0]
            voltages[i] = p.getMetaValue("collision energy")

    tics_backup = tics.copy()
    if args.injectionTimeAdjust:
        for i in tqdm(range(ns)):
            tics[i] = tics[i] * injection_times[i]
            
    
    # df_voltages = pd.read_excel(args.inVoltages, header = None, names = ['Voltages'])
    df_voltages = pd.read_csv(args.inVoltages, header = None, names = ['Voltages'], sep = '\t')
    
    ces = df_voltages['Voltages'].values
    
    # parent_form = molmass.Formula(args.parentFormula).formula
    charge = {"Pos" : 1, "Neg" : -1}[args.mode]
    

    
    df = pd.DataFrame()
    # df_peakList = pd.read_excel(args.peakList).sort_values(by = 'm/z').reset_index(drop = True)
    df_peakList = pd.read_csv(args.peakList, sep = '\t').sort_values(by = 'm/z').reset_index(drop = True)
    if 'formula' not in df_peakList.columns:
        df_peakList['formula'] = None
    pl_mzs = df_peakList['m/z'].values
    
    # fdForms = [formulaUtils.findBestForm(x, parent_form, toleranceDa = 0.01, charge = charge) for x in pl_mzs]
    # df_peakList['formula'] = list(zip(*fdForms))[0]
    
    polarities_int = {'Pos' : 1, 'Neg' : 2}[args.mode]
    
    skip_ces = []
    for ce in tqdm(ces):
        ce_indices = np.where((levels == 2) & (voltages == ce) & (polarities == polarities_int))[0]
        if len(ce_indices) == 0:
            skip_ces.append(ce)
            continue
        if len(skip_ces) > 0:
            with open(args.log_file, 'a') as f:
                print(f"\nNo spectra found for the following collision energies: {skip_ces}", file = lf)
        ce_median = np.median(tics[ce_indices])
        # ce_indices_mfilt = np.where((levels == 2) & (voltages == ce) & (polarities == 1) & (tics > (7e7)) & (tics < (1.3e8)))[0]
        ce_indices_mfilt = np.where((levels == 2) & (voltages == ce) & (polarities == polarities_int) & (tics > (ce_median * args.lowerPercent)) & (tics < (ce_median * args.upperPercent)))[0]
        lcm = len(ce_indices_mfilt)
        ce_mean = np.mean(tics[ce_indices_mfilt])
        ce_sampVar = np.sum(np.power(tics[ce_indices_mfilt] - np.mean(tics[ce_indices_mfilt]), 2)) / (lcm - 1)

        ce_tic_min = tics[ce_indices_mfilt].min()
        ce_tic_max = tics[ce_indices_mfilt].max()

        # num_slices = args.numSlices
        edges = np.linspace(ce_tic_min, ce_tic_max, args.numSlices + 1)

        vMzmlDir = os.path.join(args.outDir, "voltage_filtered_mzmls")
        os.system(f"mkdir -p {vMzmlDir}")
        os.chmod(vMzmlDir, 0o777)
        ce_out_mzml_fname = os.path.join(vMzmlDir, f"{ce}V.mzML")
        ce_out_pkl_fname = os.path.join(vMzmlDir, f"{ce}V.pkl")
        pkl.dump(edges, open(ce_out_pkl_fname, 'wb'))

        write_consumer = PlainMSDataWritingConsumer(ce_out_mzml_fname)
        e = MSExperiment()
        for k in ce_indices_mfilt:
            s = ondisc_exp.getSpectrum(int(k))
            write_consumer.consumeSpectrum(s)
        del write_consumer


    output_all = [calc_ce_column(ce, df_peakList, vMzmlDir, pl_mzs, args) for ce in tqdm([x for x in ces if x not in skip_ces])]
    
    
    colnames_all = [x[0] for x in output_all]
    colnames_all = [item for sublist in colnames_all for item in sublist]

    index_counts_all = [x[1] for x in output_all]
    index_counts_all = [item for sublist in index_counts_all for item in sublist]

    output_cols_all = [x[2] for x in output_all]
    output_cols_all = [item for sublist in output_cols_all for item in sublist]

    means_cols_all = [x[3] for x in output_all]
    means_cols_all = [item for sublist in means_cols_all for item in sublist]

    vars_cols_all = [x[4] for x in output_all]
    vars_cols_all = [item for sublist in vars_cols_all for item in sublist]
    
    avgFracs_cols_all = [x[5] for x in output_all]
    avgFracs_cols_all = [item for sublist in avgFracs_cols_all for item in sublist]
    pkl.dump(avgFracs_cols_all, open(os.path.join(args.outDir, "avgFracs_cols.pkl"), 'wb'))

    df_out = pd.DataFrame(df_peakList[['m/z', 'formula']])
    # df_out = pd.DataFrame(df_peakList[['m/z']])
    ss_dict = {}
    for cname, count, col in zip(colnames_all, index_counts_all, output_cols_all):
        df_out[cname] = col
        ss_dict[cname] = count

    cnames_calc = [x for x in df_out.columns if x not in ['m/z', 'formula']]

    mean_gammas = []
    var_gammas = []
    slice_counts = []
    peak_counts = []
    mean_fracs = []
    # cnames_calc = [x for x in df_out.columns if x not in ['m/z', 'formula']]
    for i_row, row in df_out.iterrows():
        vals = row[cnames_calc].dropna().values
        if len(vals) == 0:
            mean_gammas.append(None)
            var_gammas.append(None)
            peak_counts.append(0)
            # mean_fracs.append(None)
        else:
            mean_gammas.append(vals.mean())
            var_gammas.append(vals.var())
            peak_counts.append(np.sum([ss_dict[idx] for idx in row[cnames_calc].dropna().index]))
        slice_counts.append(len(vals))

    # df_out.insert(2, "Mean γ(1+δ)", mean_gammas)
    # df_out.insert(3, "Variance γ(1+δ)", var_gammas)
    df_out.insert(2, "Mean \N{Greek Small Letter Gamma}(1+\N{Greek Small Letter Delta})", mean_gammas)
    df_out.insert(3, "Variance \N{Greek Small Letter Gamma}(1+\N{Greek Small Letter Delta})", var_gammas)
    df_out.insert(4, "Number of slices used", slice_counts)
    df_out.insert(5, "Number of peaks used", peak_counts)

    slice_sizes = ['\t\t\t\t\tSpectra in Bin'] + [str(ss_dict[c]) for c in cnames_calc]

    # cnames_voltages = [x.split('_')[0] + 'V']
    cnames_voltages = []
    prev_ce = None
    counter = 1
    for c in cnames_calc:
        cv = int(c.split('_')[0])
        if cv == prev_ce:
            counter += 1
        else:
            prev_ce = cv
            counter = 1
        cnames_voltages.append(f"{cv}V_{counter}")    

    cnames_bins = ['\t\t\t\t\tTIC bin edges']
    for c in cnames_calc:
        _, bl, br = c.split('_')
        bl = "{:.4e}".format(float(bl))
        br = "{:.4e}".format(float(br))
        cnames_bins.append(f"{bl}-{br}")

    out_fname_body = os.path.join(args.outDir, "df_gammasCalc_body.tsv")
    out_fname_header = out_fname_body.replace("_body.tsv", "_header.tsv")
    out_file_header = open(out_fname_header, 'w')
    print('\t'.join(cnames_bins), file = out_file_header)
    print('\t'.join(slice_sizes), file = out_file_header)
    out_file_header.close()

    # df_out.to_csv(out_file, sep = '\t', index = False)
    df_out.columns = list(df_out.columns[0 : 6]) + cnames_voltages
    df_out.to_csv(out_fname_body, sep = '\t', index = False, encoding='utf-16')

    os.chmod(out_fname_body, 0o777)
    
    with open(out_fname_header, 'r') as f:
        header_lines = f.read()
    with open(out_fname_body, 'r', encoding = 'utf-16') as f:
        body_lines = f.read()
    out_lines = header_lines + body_lines
    out_fname = os.path.join(args.outDir, "df_gammasCalc.tsv")
    with open(out_fname, 'w', encoding = 'utf-16') as f:
        print(out_lines, file = f)
    os.chmod(out_fname, 0o777)
    os.system(f"rm {out_fname_header}")
    os.system(f"rm {out_fname_body}")
    
    

    df_out_mv = pd.DataFrame(df_out[['m/z', 'formula', 'Mean γ(1+δ)', 'Variance γ(1+δ)', 'Number of slices used', 'Number of peaks used']])
    for i_cn, cn in enumerate(cnames_voltages):
        df_out_mv[f"{cn}_mean"] = means_cols_all[i_cn]
        df_out_mv[f"{cn}_var"] = vars_cols_all[i_cn]
    
    out_fname_mv = os.path.join(args.outDir, "df_meansVars.tsv")
    df_out_mv.to_csv(out_fname_mv, sep = '\t', index = False, encoding = 'utf-16')
    os.chmod(out_fname_mv, 0o777)
    shutil.rmtree(vMzmlDir, ignore_errors = True)


if __name__ == "__main__":
    args = get_args()
    sys.exit(args)
    main(args)