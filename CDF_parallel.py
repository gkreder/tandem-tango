############################################################
import sys
import os
import argparse
import numpy as np
from tqdm.auto import tqdm
import spectrumMatching
# import pyopenms as pyms
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import dask
from dask_jobqueue import SLURMCluster
from dask import delayed
from dask.distributed import Client
############################################################
parser = argparse.ArgumentParser()
parser.add_argument("--tsv", required = True)
parser.add_argument("--random_seed", type = int, default = 1)
parser.add_argument("--account", default = "C3SE2023-1-16", type = str)
parser.add_argument("--queue", default = "vera", type = str)
parser.add_argument('--cores', default = 4, type = int)
parser.add_argument('--memory', default = "16 GB", type = str)
parser.add_argument('--walltime', default = "05:00:00", type = str)
parser.add_argument('--interface', default = "ib0", type = str)
parser.add_argument('--lifetime', default = "4h", type = str)
parser.add_argument('--lifetime_stagger', default = "10m", type = str)
parser.add_argument('--min_workers', default = 4, type = int)
parser.add_argument('--max_workers', default = 100, type = int)
parser.add_argument('--log_directory', default = None)
parser.add_argument("--matching_results", action = 'store_true')
parser.add_argument("--log_plots", action = 'store_true')
# "/cephyr/users/reder/Vera/dask_logs"
parser.add_argument("--python", default = None)
# "/cephyr/users/reder/Vera/.conda/envs/dask/bin/python"
args = parser.parse_args()
############################################################

with open(args.tsv, 'r') as f:
    lines = f.readlines()
cmd_args_header = lines[0].split('\t')
lines = lines[1 : ]

@dask.delayed
def run_task(i_line, line):
    line_args = line.split('\t')
    cmd_args_dict = dict()
    for (h, a) in zip(cmd_args_header, line_args):
        if a not in ["", '\n']:
            cmd_args_dict[h] = a

    cmd_args_dict['parentMZ'] = cmd_args_dict['Exact m/z']
    cmd_args_dict['parentFormula'] = cmd_args_dict['Formula (w/ adduct)']
    if "inDir" in cmd_args_dict.keys():
        if "inFiles" in cmd_args_dict.keys():
            sys.exit(f"Both inDir and inFiles were provided in row {i_line + 1}. Please provide only one")
        idt = cmd_args_dict['inDir']
        cmd_args_dict['inFiles'] = ",".join([os.path.join(idt, x) for x in os.listdir(idt) if x.lower().endswith('.mzml')])
    hit_rows = spectrumMatching.scrape_spectra_hits(**cmd_args_dict)
    # hit_rows
    np.random.seed(args.random_seed)
    choices = np.random.choice(range(0, len(hit_rows)), len(hit_rows), replace=False)
    i = 0
    pairs = []
    while True:
        pairs.append((choices[i], choices[i + 1]))
        i += 2
        if i >= (len(choices) - 1):
            break
    out_rows = []
    pref = cmd_args_dict['outPrefix']
    for iPair, (iSpec, jSpec) in enumerate(tqdm(pairs)):
        iMeta = hit_rows[iSpec]
        jMeta = hit_rows[jSpec]
        drop_keys = ["Compound", "inDir", "inFiles", "Mode", "Collision Energy", 
                    "Target RT", "Begin", "End", "Adduct", "Formula (w/ adduct)",
                    "Exact m/z", "Targeted m/z", "isolationMZTol"]
        spec_matching_args_dict = {k : v for (k,v) in cmd_args_dict.items() if k not in drop_keys}
        spec_matching_args_dict['outDir'] = os.path.join(cmd_args_dict['outDir'], f"{pref}_matching")
        spec_matching_args_dict['outPrefix'] = f"{pref}_{iPair}"
        spec_matching_args_dict['mzml1'] = iMeta[0]
        spec_matching_args_dict['mzml2'] = jMeta[0]
        spec_matching_args_dict['index1'] = iMeta[2]
        spec_matching_args_dict['index2'] = jMeta[2]
        spec_matching_args_dict['startingIndex'] = 0
        spec_matching_args = " ".join([f"--{k} {v}" for (k, v) in spec_matching_args_dict.items()])
        spec_matching_args += " --silent"
        spec_matching_args += " --intersection_only"
        if not args.log_plots:
            spec_matching_args += " --no_log_plots"
        if not args.matching_results:
            spec_matching_args += " --no_matching_results"
        spec_matching_args = spectrumMatching.get_args(spec_matching_args)
        try:
            dfStats = spectrumMatching.run_matching(spec_matching_args)
        except: 
            print(f'Failed at iMeta = {iMeta} jMeta = {jMeta}')
            sys.stdout.flush()
            continue
        if dfStats is None:
            continue
        out_row = []
        out_row.append(iPair)
        out_row.append(spec_matching_args_dict['mzml1'])
        out_row.append(spec_matching_args_dict['index1'])
        out_row.append(spec_matching_args_dict['mzml2'])
        out_row.append(spec_matching_args_dict['index2'])
        out_row.append(dfStats['Intersection']['M'])
        out_row.append(dfStats['Intersection']['S_A'])
        out_row.append(dfStats['Intersection']['S_B'])
        out_row.append(dfStats['Intersection']['D^2'])
        out_row.append(dfStats['Intersection']['pval_D^2'])
        out_row.append(dfStats['Intersection']['G^2'])
        out_row.append(dfStats['Intersection']['pval_G^2'])
        out_rows.append(out_row)
        
    header = ["pair_index", 'mzML File 1','File 1 spectrum index (A)','mzML File 2','File 2 spectrum index (B)','M','S_A','S_B','D^2',
    'p-val (D^2)','G^2','p-val (G^2)']
    dfOut = pd.DataFrame(out_rows, columns = header)
    dfOut.at[0, '# of spectrum pairs compared'] = len(dfOut)
    outFile = os.path.join(cmd_args_dict['outDir'], f"{pref}.tsv")
    dfOut.to_csv(outFile, sep = '\t', index = False)
    plt.rcParams['font.family'] = 'DejaVu Sans'
    if len(out_rows) <= 1:
        return(1)
    for pvalType in ['D^2', 'G^2']:
        # plt.plot(dfOut['p-val (D^2)'])
        data = dfOut[f'p-val ({pvalType})']
        # scipy.stats.kstest(th, th)
        res = scipy.stats.kstest(data, 'uniform')
        kspval, ksstat = res.pvalue, res.statistic
        x = np.sort(data)
        y = 1. * np.arange(len(data)) / (len(data) - 1)
        x_ones = np.linspace(0, 1, 6)
        th = scipy.stats.uniform().cdf(x_ones)
        # th = np.arange(0, 1.01, 0.01)
        fig, ax = plt.subplots()
        # plt.plot(x,y, color = "blue")
        plt.step(x,y, color = "blue", linewidth = 0.5, where = 'post')
        plt.plot(x_ones, th, color = "black", linewidth = 0.5)
        title = f"{pref.replace('_', ' ')} {pvalType}"
        plt.title(title)
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))
        pltText = f"N: {len(dfOut)}\nK-S statistic:{ksstat:.3f}\nK-S test p-value: {kspval:.2E}"
        plt.figtext(0.92, 0.5, pltText, fontsize=14, ha = 'left', fontfamily = 'DejaVu Sans')
        ax.set_aspect('equal')
        # plotFile = os.path.basename(args.inFile).replace('.pkl', f'_{pvalType.replace("^", "")}.pdf')
        plotFile = os.path.join(cmd_args_dict['outDir'], f"{pref}_{pvalType.replace('^', '')}.pdf")
        plt.savefig(plotFile, bbox_inches = 'tight')
        plt.close()
        dfOut.at[0, f'K-S test {pvalType} p-val'] = kspval
    dfOut.insert(len(header), None, None)
    outFile = os.path.join(cmd_args_dict['outDir'], f"{pref}.tsv")
    dfOut.to_csv(outFile, sep = '\t', index = False)
    # plt.rcParams['font.family'] = 'default'
    return(0)


kwargs = {'queue' : args.queue,
          'account' : args.account,
          'cores' : args.cores,
          'memory' : args.memory,
          'walltime' : args.walltime,
          'interface' : args.interface,
          'processes' : 1,
          # remember that lifetime must be in format e.g. 45m or 4h
        #   'worker_extra_args' : ["--lifetime", args.lifetime, "--lifetime-stagger", args.lifetime_stagger, "--lifetime-restart"]} 
        'worker_extra_args' : ["--lifetime", args.lifetime, "--lifetime-stagger", args.lifetime_stagger]} 
if args.log_directory:
    kwargs['log_directory'] = args.log_directory

cluster = SLURMCluster(**kwargs)
# cluster.adapt(minimum = args.min_workers, maximum = args.max_workers, target_duration = "4h")
cluster.adapt(minimum = args.min_workers, maximum = args.max_workers)
# cluster.scale(args.num_workers)
# if args.log_directory is None:
#     # cluster = SLURMCluster(queue=args.queue, account=args.account, cores=args.cores, memory=args.memory,
#     #                     walltime = args.walltime, interface = args.interface,
#     #                     processes = 1)
#     cluster = SLURMCluster(queue=args.queue, account=args.account, cores=args.cores, memory=args.memory,
#                         walltime = args.walltime, interface = args.interface,
#                         processes = 1, worker_extra_args = ["--lifetime", "4 hour", "--lifetime-stagger", "10m"])
# else:
#     cluster = SLURMCluster(queue=args.queue, account=args.account, cores=args.cores, memory=args.memory,
#                         walltime = args.walltime, interface = args.interface,
#                         processes = 1, worker_extra_args = ["--lifetime", "4 hour", "--lifetime-stagger", "10m"], log_directory = args.log_directory)

client = Client(cluster)
results = [run_task(i_line, line) for i_line, line in enumerate(lines)]
dask.compute(results)