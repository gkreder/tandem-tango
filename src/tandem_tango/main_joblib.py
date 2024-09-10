################################################################################
# gk@reder.io
################################################################################
import os
import sys
import argparse
import inspect
import logging
from tqdm.auto import tqdm
import gc

import pandas as pd
import numpy as np
from tandem_tango import spectrum_matching
from joblib import Parallel, delayed
from memory_profiler import profile

################################################################################
# Spectrum Matching
################################################################################
def get_spectrum_matching_args(input_file: str):
    inputs = pd.read_csv(input_file, sep='\t')
    inputs_args = inputs.to_dict(orient='records')
    inputs_args_filtered = [{k: v for k, v in x.items() if pd.notnull(v)} for x in inputs_args]
    inputs_cmd_line = [[item for k, v in x.items() for item in [f"--{k}", str(v)]] for x in inputs_args_filtered]
    return inputs_cmd_line

def run_spectrum_matching_task(run_args_cmd_line):
    try:
        parser = spectrum_matching.get_parser()
        run_args = parser.parse_args(run_args_cmd_line)
        run_args_prepped = spectrum_matching.prep_args(run_args)
        run_args_func = {k: v for k, v in run_args_prepped.items() if k in inspect.signature(spectrum_matching.run_matching).parameters}
        
        os.makedirs(run_args_func['out_dir'], exist_ok = True)
        # Set up logging within the task explicitly
        log_file = os.path.join(run_args_func['out_dir'], f"{run_args_func['out_prefix']}.log")
        # logging_handlers = [logging.FileHandler(log_file, mode='w')]
        # logging.basicConfig(level=run_args_func['verbosity'], format='%(asctime)s - %(levelname)s - %(message)s', handlers=logging_handlers)
        
        run_args_func.update({'log_filename': log_file})
        spectrum_matching.run_matching(**run_args_func)
        return True # successful run
    except Exception as e:
        print('hello!!!!!!')
        logging.error(f"Error in task: {e}")
        return False # failed run
    finally:
        gc.collect()

################################################################################
# Spectrum Matching Flow
################################################################################
def spectrum_matching_flow(input_file: str, max_workers: int, no_parallel: bool = False):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File {input_file} not found")
    elif os.path.splitext(input_file)[1] != ".tsv":
        raise ValueError(f"File {input_file} must be a tab-separated file")
    
    runs_args_cmd_line = get_spectrum_matching_args(input_file)
    
    if no_parallel:
        # Sequential execution
        for cmd in tqdm(runs_args_cmd_line, desc="Processing Spectra"):
            success = run_spectrum_matching_task(cmd)
            if not success:
                logging.warning(f"Task failed for args: {cmd}")
    # else:
    #     # Parallel execution using joblib with tqdm for progress tracking
    #     Parallel(n_jobs=max_workers, backend="loky")(
    #         delayed(run_spectrum_matching_task)(cmd) for cmd in tqdm(runs_args_cmd_line, desc="Processing Spectra")
    #     )
    else:
        # Parallel execution using joblib with tqdm for progress tracking
        results = Parallel(n_jobs=max_workers, backend="loky")(
            delayed(run_spectrum_matching_task)(cmd) for cmd in tqdm(runs_args_cmd_line, desc="Processing Spectra")
        )

        # You can now check how many tasks failed
        failed_tasks = sum(1 for result in results if not result)
        if failed_tasks > 0:
            logging.warning(f"{failed_tasks} tasks failed during execution.")


################################################################################
# Flow Runner
################################################################################
def get_parser():
    parser = argparse.ArgumentParser(description="Tandem Tango: Spectrum Matching and Processing")
    parser.add_argument('flow', choices=['spectrum_matching'], help='The pipeline to run')
    parser.add_argument('input', help='The input sample file to process')
    parser.add_argument('--max_workers', type=int, default=4, help='Maximum number of concurrent tasks (default: 4)')
    parser.add_argument('--no_parallel', action = 'store_true', help='Run the pipeline without parallel processing')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.flow == 'spectrum_matching':
        spectrum_matching_flow(args.input, max_workers=args.max_workers, no_parallel=args.no_parallel)

if __name__ == "__main__":
    main()
