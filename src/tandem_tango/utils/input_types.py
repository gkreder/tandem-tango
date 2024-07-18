################################################################################
# gk@reder.io
################################################################################
import logging
import argparse
import molmass

################################################################################
# Auxiliary functions for parsing command-line inputs
################################################################################

def peak_list_input(value):
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
    try:
        return molmass.Formula(value).formula
    except:
        raise argparse.ArgumentTypeError(f"Could not parse formula from {value}")
    
def get_logging_level(verbosity):
    levels = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }
    return levels.get(verbosity.lower(), logging.INFO)

def logging_level_input(value):
    level = get_logging_level(value)
    if level is None:
        raise argparse.ArgumentTypeError(f"Invalid logging level: {value}")
    return level

def join_types_input(value):
    valid_types = ["Intersection", "Union"]
    types = [x.strip() for x in value.split(",")]
    for t in types:
        if t not in valid_types:
            raise argparse.ArgumentTypeError(f"Invalid join type {t}")
    return types

def suffix_input(value):
    suffixes = [x.strip() for x in value.split(",")]
    if len(suffixes) != 2:
        raise argparse.ArgumentTypeError(f"Two suffixes must be provided")
    return suffixes