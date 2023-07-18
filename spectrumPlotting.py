import argparse
import os
import sys
from pyteomics import mzml, auxiliary
import pandas as pd
import warnings
import spectrumMatching

def create_parser():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--mzML', help='mzML file')
    group.add_argument('--text', help='Text file with two columns')
    parser.add_argument('--polarity', choices=['Positive', 'Negative', 'Pos', 'Neg'], required=True, help='Spectrum polarity')
    parser.add_argument('--precursor_mz', required=True, help='Precursor m/z')
    parser.add_argument('--index', required='--mzML' in sys.argv, help='MS/MS scan index')
    
    parser.add_argument('--outdir', help='Output directory path')
    parser.add_argument('--filename', help='Output filename')
    parser.add_argument('--start', default=0, help='Spectrum starting index')
    parser.add_argument('--cutoff', default=0, help='Raw count noise intensity cutoff')
    parser.add_argument('--peaks', default=5, help='Number of labeled peaks')
    

    parser.add_argument('--formula', help='Parent peak formula')
    parser.add_argument('--tolerance', default=0.01, required='--formula' in sys.argv, help='Formula assignment m/z tolerance')

    parser.add_argument('--quasiscale', action='store_true', help='Quasicount scaling')
    parser.add_argument('--quasiX', required="--quasiscale" in sys.argv, help='Quasicount function scale')
    parser.add_argument('--quasiY', required="--quasiscale" in sys.argv, help='Quasicount function exponent')
    parser.add_argument('--quasicutoff', default=0, help='Quasicount intensity cutoff')
    return parser

def get_polarity(pyteomics_spec):
    if 'positive scan' in pyteomics_spec.keys():
        return("Positive")
    elif 'negative scan' in pyteomics_spec.keys():
        return('Negative')
    else:
        sys.exit(f"Error - couldnt determine the polarity of the pyteomics spectrum {pyteomics_spec}")

def get_spectrum(**kwargs):
    args = argparse.Namespace(**kwargs)
    spectrum = None
    if args.in_type == ".mzML":
        search_index = int(args.index)
        with mzml.read(args.in_file) as reader:
            for i, spec in enumerate(reader):
                if int(spec['index']) == search_index:
                    spectrum = spec
                    break
        if not spectrum:
            sys.exit(f'Couldnt find spectrum with index {search_index} in file {args.in_file}')
        mzs = spectrum['m/z array']
        intensities = spectrum['intensity array']
        polarity = get_polarity(spectrum)
        if polarity != args.polarity:
            warnings.warn(f"Warning - the found spectrum polarity is {polarity} but user provided {args.polarity}", UserWarning)
    elif args.in_type == ".txt":
        with open(args.in_file, 'r') as f:
            lines = f.readlines()[1 : ] # skip the header
        mzs, intensities = zip(*[x.strip().split('\t') for x in lines])
        mzs = [float(x) for x in mzs]
        intensities = [float(x) for x in intensities]
    else:
        sys.exit(f'Error - unrecognized input file type {in_type}')
    out_spectrum = pd.DataFrame({'mz' : mzs, 'Intensity' : intensities})
    return(out_spectrum)

def process(**kwargs):
    args = argparse.Namespace(**kwargs)
    spectrum = args.spectrum
    sys.exit('stopped here')
    # spectrum = spectrumMatching.filter_data(spectrum, '')

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    args.polarity = args.polarity if args.polarity in ['Positive', 'Negative'] else {'Pos' : 'Positive', 'Neg' : 'Negative'}[args.polarity]
    if '--mzML' in sys.argv:
        args.in_file = args.mzML
    elif '--text' in sys.argv:
        args.in_file = args.text
    if not args.outdir:
        args.outdir = str(os.path.dirname(args.in_file))
    in_info = os.path.splitext(args.in_file)
    in_pref = os.path.basename(in_info[0])
    args.in_type = in_info[1]
    if not args.filename:
        args.out_pref = f"{in_pref}"
        st = f"Scan_{args.index}" if args.index else None
        if st:
            args.out_pref += f'_{st}'
    args.spectrum = get_spectrum(**vars(args))
    process(**args)
    


    # Process arguments
    # Your code here...
