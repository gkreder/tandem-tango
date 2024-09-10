################################################################################
# gk@reder.io
################################################################################
import os
import sys
import argparse
import inspect
import logging

import pandas as pd
import numpy as np
from tandem_tango import spectrum_matching
from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner

################################################################################
# Spectrum Matching
################################################################################
@task
def get_spectrum_matching_args(input_file : str):
    inputs = pd.read_csv(input_file, sep = '\t')
    inputs_args = inputs.to_dict(orient='records')
    inputs_args_filtered = [{k : v for k,v in x.items() if pd.notnull(v)} for x in inputs_args]
    inputs_cmd_line = [[item for k, v in x.items() for item in [f"--{k}", str(v)]] for x in inputs_args_filtered]
    return inputs_cmd_line

@task
def run_spectrum_matching_task(run_args_cmd_line):
    parser = spectrum_matching.get_parser()
    run_args = parser.parse_args(run_args_cmd_line)
    run_args_prepped = spectrum_matching.prep_args(run_args)
    # run_args_func = {k: v for k, v in vars(run_args).items() if k in inspect.signature(spectrum_matching.run_matching).parameters}
    run_args_func = {k: v for k, v in run_args_prepped.items() if k in inspect.signature(spectrum_matching.run_matching).parameters}
    # run_args_func.update({'log_filename' : os.path.join(run_args_func['out_dir'], f"{run_args_func['out_prefix']}.log")})

    log_file = os.path.join(run_args_func['out_dir'], f"{run_args_func['out_prefix']}.log")
    # Set up logging within the task explicitly
    logging_handlers = [logging.FileHandler(log_file, mode='w')]
    logging.basicConfig(level=run_args_func['verbosity'], format='%(asctime)s - %(levelname)s - %(message)s', handlers=logging_handlers)
    run_args_func.update({'log_filename' : log_file})
    spectrum_matching.run_matching(**run_args_func)

@flow
def spectrum_matching_flow(input_file : str, max_workers : int):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File {input_file} not found")
    elif os.path.splitext(input_file)[1] != ".tsv":
        raise ValueError(f"File {input_file} must be a tab-separated file")
    
    runs_args_cmd_line = get_spectrum_matching_args(input_file)
    with ConcurrentTaskRunner(max_workers=max_workers):
        run_spectrum_matching_task.map(runs_args_cmd_line)
    return 

################################################################################
# Flow Runner
################################################################################
def get_parser():
    parser = argparse.ArgumentParser(description="Tandem Tango: Spectrum Matching and Processing")
    parser.add_argument('flow', choices=['spectrum_matching'],help='The pipeline to run')
    parser.add_argument('input', help='The input sample file to process')
    parser.add_argument('--max_workers', type=int, default=4, help='Maximum number of concurrent tasks (default: 4)')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.flow == 'spectrum_matching':
        spectrum_matching_flow(args.input, max_workers=args.max_workers)


if __name__ == "__main__":
    main()
    

