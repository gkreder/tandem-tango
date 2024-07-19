################################################################################
# gk@reder.io
################################################################################
import logging
import argparse
import os

import molmass

################################################################################
# Auxiliary functions for parsing command-line inputs
################################################################################

def bool_input(value):
    """Parses a boolean from the passed value"""
    if value.lower() in ['true', 't', 'yes', 'y', '1']:
        return True
    elif value.lower() in ['false', 'f', 'no', 'n', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")

def peak_list_input(value : str):
    """Parses a list of peaks from the passed value assuming the value is either a file path or a comma-separated list of floats"""
    try:
        if os.path.exists(value):
            with open(value, 'r') as f:
                lines = f.readlines()
            return [float(x.strip()) for x in lines]
        else:
            return [float(x) for x in value.split(",")]
    except:
        raise argparse.ArgumentTypeError(f"Could not parse peak list from {value}")
    
def formula_input(value):
    """Parses a chemical formula from the passed value using the molmass library"""
    try:
        return molmass.Formula(value).formula
    except:
        raise argparse.ArgumentTypeError(f"Could not parse formula from {value}")
    
def get_logging_level(verbosity):
    """Returns the logging level corresponding to the passed verbosity"""
    levels = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }
    return levels.get(verbosity.lower(), logging.INFO)

def logging_level_input(value):
    """Parses a logging level from the passed value"""
    level = get_logging_level(value)
    if level is None:
        raise argparse.ArgumentTypeError(f"Invalid logging level: {value}")
    return level

def join_types_input(value):
    """Parses a list of join types from the passed value, current valid types are 'Intersection' and 'Union' or a combination of both"""
    valid_types = ["Intersection", "Union"]
    types = [x.strip() for x in value.split(",")]
    for t in types:
        if t not in valid_types:
            raise argparse.ArgumentTypeError(f"Invalid join type {t}")
    return types

def suffix_input(value):
    """Parses a pair of suffixes from the passed value, checking that exactly two suffixes are provided"""
    suffixes = [x.strip() for x in value.split(",")]
    if len(suffixes) != 2:
        raise argparse.ArgumentTypeError(f"Two suffixes must be provided")
    return suffixes