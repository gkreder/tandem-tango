import argparse
from typing import Dict, Tuple
import logging

from pyteomics import mzml as pytmzml

# Constants
NS = {'mzml': 'http://psi.hupo.org/ms/mzml',
      'ms_level': 'MS:1000511',
      "scan_start_time" : 'MS:1000016', # in minutes
      "isolation_window_target_mz" : 'MS:1000827'}
QUASI_CUTOFF_DEFAULT = 5

def parse_float(value: str) -> float:
    """Safely parse a string to float."""
    try:
        return float(value)
    except ValueError:
        logging.error(f"Could not convert {value} to float.")
        return 0.0
    
def clean_file_path(file_path: str) -> str:
    """Remove leading and trailing quotes from a file path."""
    return file_path.strip('"')


def parse_spectrum(spectrum, ns: Dict[str, str], bRT: float, eRT: float, target_isolation_window_mz: float, isolation_tol: float) -> Tuple[str, str, str, float]:
    """Parse a single spectrum element."""
    ms_level = scan_start_time = isolation_window_mz = None
    for cvParam in spectrum.findall('.//mzml:cvParam', ns):
        accession = cvParam.get('accession')
        if accession == NS['ms_level']:  # MS level
            ms_level = cvParam.get('value')
        elif accession == NS['scan_start_time']:  # Scan start time (minutes)
            scan_start_time = parse_float(cvParam.get('value'))
        elif accession == NS['isolation_window_target_mz']:  # Isolation window target m/z
            isolation_window_mz = parse_float(cvParam.get('value'))
    if ms_level == "2" and bRT <= scan_start_time <= eRT and abs(isolation_window_mz - target_isolation_window_mz) <= isolation_tol:
        return spectrum.get('id'), spectrum.get("index"), scan_start_time
    return None

def scrape_spectra_hits(**kwargs) -> List[Tuple[str, str, str, float]]:
    """
    Scrape spectra hits from mzML files based on given criteria.

    Parameters:
    - kwargs: Dictionary of arguments including inFiles, Begin, End, Targeted m/z, and isolationMZTol.

    Returns:
    - List of tuples containing file name, spectrum native id, spectrum index, and scan start time.
    """
    in_files = [clean_file_path(file) for file in kwargs['inFiles'].split(",")]
    hit_rows = []
    bRT = parse_float(kwargs['Begin'])
    eRT = parse_float(kwargs['End'])
    target_isolation_window_mz = parse_float(kwargs['Targeted m/z'])
    isolation_tol = parse_float(kwargs['isolationMZTol'])

    for file_path in tqdm(in_files):
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            for spectrum in root.findall('.//mzml:spectrum', NS):
                parsed_spectrum = parse_spectrum(spectrum, NS, bRT, eRT, target_isolation_window_mz, isolation_tol)
                if parsed_spectrum:
                    hit_rows.append((file_path, *parsed_spectrum))
        except ET.ParseError:
            logging.error(f"Error parsing file: {file_path}")
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
    return hit_rows

def quasi_convert(d, **kwargs):
    args = argparse.Namespace(**kwargs)
    # Conversion to quasicounts by dividing the intensity by gamma_i(1 + delta)
    d = d.copy()
    d['quasi'] = d['intensity'] / ( args.quasiX *  ( np.power(d['mz'], args.quasiY) ) )
    return(d)

def filter_data(d, filter, **kwargs):
    args = argparse.Namespace(**kwargs)
    d = d.copy()
    if len(d) == 0:
        return(d)
    if filter == 'absolute':
        out_d = d.loc[lambda x : x['intensity'] >= args.absCutoff]
    elif filter == "pdpl":
        diffSearch = np.min(np.abs(d['mz'][:, np.newaxis] - args.prePeaks), axis = 1)
        indices = np.where(diffSearch <= args.matchAcc)
        out_d = d.iloc[indices]
    elif filter == 'quasi':
        out_d = d.loc[lambda x : x['quasi'] >= args.quasiCutoff]
    elif filter == 'parent_mz':
        out_d = d.loc[lambda x : x['mz'] <= ( args.parentMZ + args.matchAcc )]
    elif filter == 'match_acc':
        out_d = d.copy()
        # Eliminating all peaks within match accuracy of peak exclusion list peaks
        for mze in args.excludePeaks:
            out_d = out_d.loc[lambda x : np.abs(x['mz'] - mze) > args.matchAcc]
    elif filter == 'relative':
        indices = np.where((d['intensity'] / max(d['intensity'])) >= args.relCutoff)
        out_d = d.iloc[indices]
    elif filter == 'res_clearance':
        mzs = d['mz'].copy().values
        out_d = d.copy()
        for mz in mzs:
            if mz not in out_d['mz'].values:
                continue
            # out_d = out_d.loc[lambda x : np.abs(x['mz'] - mz) > args.resClearance]    
            out_d = pd.concat([out_d.loc[lambda x : x['mz'] == mz], out_d.loc[lambda x : np.abs(x['mz'] - mz) > args.resClearance]]).reset_index(drop = True)
    return(out_d)

def H(x):
    h = -1 * np.sum( x * ( np.log(x) ) )
    return(h)

def get_spectrum_by_index(mzml_file: str, spec_index: int, gainControl: bool = False):
    """Get a spectrum by index from an mz file"""
    reader = pytmzml.MzML([mzml_file])
    spec = reader.get_by_index(spec_index)
    reader.close()
    if gainControl:
        raise NotImplementedError("Gain control is not yet implemented for Pyteomics implementation")
    return spec



spectra = get_spectrum_by_index()






        